import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __package__:
    from .bootstrap import ensure_package
else:
    from bootstrap import ensure_package

ensure_package()

from astrbot_plugin_relationship_memory.policy import validate_analysis
from astrbot_plugin_relationship_memory.store import Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "data.sqlite3"))
        self.store.ensure_profile("g1:u1", "g1", "u1", "group:g1:u1")
        self.store.ensure_profile("g1:u2", "g1", "u2", "group:g1:u2")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def analysis(self, **kw):
        result = {
            "deltas": {"affection": 3, "trust": 2, "familiarity": 1, "closeness": -2},
            "confidence": 0.9,
            "reason": "test",
            "style": "warm",
            "emotion": "happy",
            "memories": [{"content": "likes tea", "kind": "fact", "public": True}],
            "followups": [],
        }
        result.update(kw)
        return result

    def test_profile_isolation_and_message_dedup(self):
        t = 1_800_000_000
        self.assertEqual(self.store.profile("g1:u1")["affection"], 20)
        self.assertTrue(self.store.commit_turn("g1:u1", "m1", "hello", "hi", self.analysis(), t))
        self.assertFalse(
            self.store.commit_turn("g1:u1", "m1", "hello", "hi", self.analysis(), t + 2)
        )
        self.assertEqual(self.store.profile("g1:u1")["affection"], 23)
        self.assertEqual(self.store.profile("g1:u2")["affection"], 20)
        self.assertEqual(len(self.store.recent("g1:u1")), 1)

    def test_score_threshold_clamp_daily_caps_and_duplicate_text(self):
        t = 1_800_000_000
        low = self.analysis(confidence=0.69)
        self.store.commit_turn("g1:u1", "a", "low", "x", low, t)
        self.assertEqual(self.store.profile("g1:u1")["affection"], 20)
        # One-hour repeated text is still stored but does not earn another score change.
        self.store.commit_turn("g1:u1", "b", "low", "x", self.analysis(), t + 10)
        self.assertEqual(self.store.profile("g1:u1")["affection"], 20)
        for i in range(6):
            self.store.commit_turn(
                "g1:u1", f"p{i}", f"distinct {i}", "x", self.analysis(), t + 7200 + i
            )
        self.assertEqual(self.store.profile("g1:u1")["affection"], 28)  # max +8/day
        self.assertEqual(self.store.profile("g1:u1")["closeness"], 0)  # floor at zero
        self.assertEqual(self.store.profile("g1:u1")["familiarity"], 6)
        self.assertEqual(len(self.store.audit("g1:u1")), 6)

    def test_daily_limits_are_configurable_and_zero_disables_them(self):
        base = 1_800_000_000
        day = datetime.fromtimestamp(base, timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
        # No limits passed: the historical +8/day ceiling still applies.
        for i in range(5):
            self.store.commit_turn("g1:u1", f"d{i}", f"default {i}", "x", self.analysis(), base + i)
        self.assertEqual(self.store.profile("g1:u1")["affection"], 28)
        # A tighter configured cap wins over the default, per user and per day.
        for i in range(5):
            self.store.commit_turn(
                "g1:u2",
                f"c{i}",
                f"capped {i}",
                "x",
                self.analysis(),
                base + i,
                limits={"affection_gain": 2, "dimension_gain": 1},
            )
        capped = self.store.profile("g1:u2")
        self.assertEqual(capped["affection"], 22)
        self.assertEqual(capped["trust"], 11)  # starts at 10, +1/day instead of +2
        self.assertEqual(self.store.daily_usage("g1:u2", day)["used"]["affection"]["gain"], 2)
        # 0 disables the cap for that side: every requested point lands.
        self.store.ensure_profile("g1:u3", "g1", "u3", "group:g1:u3")
        for i in range(5):
            self.store.commit_turn(
                "g1:u3",
                f"u{i}",
                f"unlimited {i}",
                "x",
                self.analysis(),
                base + i,
                limits={"affection_gain": 0, "dimension_gain": 0},
            )
        self.assertEqual(self.store.profile("g1:u3")["affection"], 35)
        self.assertEqual(self.store.daily_usage("g1:u3", day)["used"]["affection"]["gain"], 15)
        # Unknown keys are ignored so a typo can never silently disable a cap.
        self.store.ensure_profile("g1:u4", "g1", "u4", "group:g1:u4")
        for i in range(5):
            self.store.commit_turn(
                "g1:u4",
                f"x{i}",
                f"typo {i}",
                "x",
                self.analysis(),
                base + i,
                limits={"affection_gain": 2, "nonsense": 0},
            )
        self.assertEqual(self.store.profile("g1:u4")["affection"], 22)

    def test_daily_negative_cap(self):
        t = 1_800_000_000
        for i in range(5):
            a = self.analysis(
                deltas={"affection": -3, "trust": -3, "familiarity": -3, "closeness": -3}
            )
            self.store.commit_turn("g1:u1", f"n{i}", f"negative {i}", "x", a, t + i)
        p = self.store.profile("g1:u1")
        self.assertEqual(p["affection"], 10)  # -10 daily limit
        self.assertEqual(p["trust"], 0)  # score floor
        self.assertEqual(p["familiarity"], 0)  # never decreases

    def test_memory_disabled_keeps_relationship_only(self):
        self.store.set_preferences("g1:u1", memory_enabled=False)
        self.store.commit_turn("g1:u1", "m1", "secret", "reply", self.analysis(), 1_800_000_000)
        p = self.store.profile("g1:u1")
        self.assertEqual(p["affection"], 23)
        self.assertEqual(p["style"], "")
        self.assertEqual(self.store.recent("g1:u1"), [])
        self.assertEqual(self.store.memories("g1:u1"), [])

    def test_low_confidence_and_empty_style_emotion_do_not_overwrite(self):
        t = 1_800_000_000
        self.store.commit_turn("g1:u1", "a", "first", "reply", self.analysis(), t)
        before = self.store.profile("g1:u1")
        self.assertEqual(before["emotion_updated_at"], t)
        blank = self.analysis(
            confidence=0.69,
            style="",
            emotion="",
            memories=[{"content": "low confidence fact"}],
            followups=[
                {"content": "do not save", "due_at": "2030-01-01T12:00:00+08:00", "public": True}
            ],
        )
        self.store.commit_turn("g1:u1", "b", "second", "reply", blank, t + 100)
        after = self.store.profile("g1:u1")
        self.assertEqual(after["style"], "warm")
        self.assertEqual(after["emotion"], "happy")
        self.assertEqual(after["emotion_updated_at"], t)
        self.assertEqual([m["content"] for m in self.store.memories("g1:u1")], ["likes tea"])
        self.assertEqual(self.store.followups("g1:u1", t + 86400), [])

    def test_preferences_persona_revision_and_default_profile_fields(self):
        p = self.store.ensure_profile("persona-key", "g", "u", "u", persona="original")
        self.assertEqual(p["last_interaction"], 0)
        self.assertEqual(
            self.store.ensure_profile("persona-key", "g", "u", "u")["persona"], "original"
        )
        rev = self.store.profile("persona-key")["revision"]
        self.store.ensure_profile("persona-key", "g", "u", "u", persona="changed")
        self.assertEqual(self.store.profile("persona-key")["revision"], rev + 1)
        rev = self.store.profile("persona-key")["revision"]
        self.store.set_preferences("persona-key", memory_enabled=False)
        self.assertEqual(self.store.profile("persona-key")["revision"], rev + 1)

    def test_personal_persona_requires_current_message_evidence_and_is_per_profile(self):
        now = 1_800_000_000
        user_message = "请记住我叫阿岚，平时住在海边。"
        valid = validate_analysis(
            {
                "confidence": 0.95,
                "user_persona": "用户叫阿岚，住在海边。",
                "persona_evidence": "我叫阿岚，平时住在海边",
            },
            now,
            user_message,
        )
        self.assertEqual(valid["user_persona"], "用户叫阿岚，住在海边。")
        foreign = validate_analysis(
            {
                "confidence": 0.95,
                "user_persona": "用户叫阿岚。",
                "persona_evidence": "用户叫阿岚",
            },
            now,
            "我朋友叫阿岚",
        )
        self.assertEqual(foreign["user_persona"], "")
        self.assertEqual(
            validate_analysis(
                {
                    "confidence": 0.95,
                    "user_persona": "无来源的人设",
                    "persona_evidence": "",
                },
                now,
                user_message,
            )["user_persona"],
            "",
        )

        self.store.ensure_profile("persona-a", "g", "u1", "umo")
        self.store.ensure_profile("persona-b", "g", "u2", "umo")
        self.assertTrue(self.store.commit_turn("persona-a", "a1", user_message, "好的", valid, now))
        self.assertEqual(self.store.profile("persona-a")["user_persona"], "用户叫阿岚，住在海边。")
        self.assertEqual(self.store.profile("persona-b")["user_persona"], "")

        # Empty means no update, and memory opt-out blocks further persona writes.
        self.store.commit_turn(
            "persona-a",
            "a2",
            "普通聊天",
            "收到",
            self.analysis(user_persona="", persona_evidence=""),
            now + 7200,
        )
        self.assertEqual(self.store.profile("persona-a")["user_persona"], "用户叫阿岚，住在海边。")
        self.store.set_preferences("persona-a", memory_enabled=False)
        self.store.commit_turn(
            "persona-a",
            "a3",
            "称呼我为小岚",
            "好",
            self.analysis(user_persona="用户叫小岚。", persona_evidence="称呼我为小岚"),
            now + 14400,
        )
        self.assertEqual(self.store.profile("persona-a")["user_persona"], "用户叫阿岚，住在海边。")
        self.store.forget("persona-a")
        self.assertEqual(self.store.profile("persona-a")["user_persona"], "")
        self.assertEqual(self.store.profile("persona-b")["user_persona"], "")

    def test_relationship_reset_clears_personal_persona_only_for_target_profile(self):
        now = 1_800_000_000
        self.store.ensure_profile("persona-a", "g", "u1", "umo")
        self.store.ensure_profile("persona-b", "g", "u2", "umo")
        for key, label in (("persona-a", "甲"), ("persona-b", "乙")):
            self.store.commit_turn(
                key,
                f"{label}-1",
                f"我叫{label}",
                "记住了",
                self.analysis(user_persona=f"用户叫{label}。", persona_evidence=f"我叫{label}"),
                now,
            )
        self.store.reset_relationship("persona-a")
        self.assertEqual(self.store.profile("persona-a")["user_persona"], "")
        self.assertEqual(self.store.profile("persona-b")["user_persona"], "用户叫乙。")

    def test_memory_dedupe_sources_replacement_and_search(self):
        t = 1_800_000_000
        self.store.commit_turn(
            "g1:u1",
            "m1",
            "tea",
            "reply",
            self.analysis(memories=[{"content": "喜欢喝绿茶", "public": True}]),
            t,
        )
        fact = self.store.memories("g1:u1")[0]
        self.assertEqual(fact["source_ids"], ["m1"])
        self.assertEqual(self.store.memories("g1:u1", "喜欢喝绿茶")[0]["id"], fact["id"])
        self.store.commit_turn(
            "g1:u1",
            "m2",
            "tea again",
            "reply",
            self.analysis(memories=[{"content": "喜欢喝绿茶", "public": True}]),
            t + 7200,
        )
        memories = self.store.memories("g1:u1", "绿茶")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["source_ids"], ["m1", "m2"])
        self.store.commit_turn(
            "g1:u1",
            "m3",
            "privacy",
            "reply",
            self.analysis(memories=[{"content": "喜欢喝绿茶", "public": False}]),
            t + 14400,
        )
        self.assertFalse(self.store.memories("g1:u1")[0]["public"])
        old_id = self.store.memories("g1:u1")[0]["id"]
        self.store.commit_turn(
            "g1:u1",
            "m4",
            "job",
            "reply",
            self.analysis(
                replace_memory_ids=[old_id], memories=[{"content": "已入职新公司", "public": False}]
            ),
            t + 21600,
        )
        self.assertEqual([m["content"] for m in self.store.memories("g1:u1")], ["已入职新公司"])

    def test_forget_preserves_message_dedup_but_erases_private_payloads(self):
        t = 1_800_000_000
        self.store.commit_turn("g1:u1", "same", "private text", "reply", self.analysis(), t)
        job = self.store.reserve_proactive("g1:u1", "p1", t + 50000, 1, 0, group_interval_minutes=0)
        self.store.finish_proactive(job, "sent", "private outbound text")
        self.store.forget("g1:u1")
        self.assertFalse(
            self.store.commit_turn(
                "g1:u1", "same", "private text", "reply", self.analysis(), t + 60000
            )
        )
        self.assertEqual(
            self.store.db.execute("SELECT reason FROM audit WHERE key='g1:u1'").fetchone()[0], ""
        )
        self.assertEqual(
            self.store.db.execute("SELECT text FROM proactive_jobs WHERE key='g1:u1'").fetchone()[
                0
            ],
            "",
        )

    def test_summary_source_ids_replacement_and_revision(self):
        t = 1_800_000_000
        self.store.commit_turn(
            "g1:u1",
            "m1",
            "fact",
            "reply",
            self.analysis(memories=[{"content": "正在面试", "public": False}]),
            t,
        )
        old_id = self.store.memories("g1:u1")[0]["id"]
        pending = self.store.pending_summaries(t + 1)[0]
        rev = self.store.profile("g1:u1")["revision"]
        self.assertTrue(
            self.store.apply_summary(
                "g1:u1",
                pending["revision"],
                [x["id"] for x in pending["turns"]],
                {
                    "summary": "job update",
                    "style": "focused",
                    "replace_memory_ids": [old_id],
                    "memories": [{"content": "已经入职", "public": True}],
                },
            )
        )
        p = self.store.profile("g1:u1")
        self.assertEqual(p["revision"], rev + 1)
        memory = self.store.memories("g1:u1")[0]
        self.assertEqual(memory["content"], "已经入职")
        self.assertEqual(memory["source_type"], "summary")
        self.assertEqual(memory["source_ids"], [str(pending["turns"][0]["id"])])
        self.assertFalse(memory["public"])  # a summary cannot promote a private fact

    def test_proactive_terminal_restart_scope_and_stats(self):
        t = datetime(2026, 1, 1, 2, tzinfo=timezone.utc).timestamp()
        key_a = '["platform-a","bot-a","room","u1"]'
        key_b = '["platform-a","bot-b","room","u2"]'
        self.store.ensure_profile(key_a, "room", "u1", key_a)
        self.store.ensure_profile(key_b, "room", "u2", key_b)
        first = self.store.reserve_proactive(
            key_a, "a1", t, 1, 0, group_limit=1, group_interval_minutes=60
        )
        self.assertIsNotNone(first)
        second = self.store.reserve_proactive(
            key_b, "b1", t, 1, 0, group_limit=1, group_interval_minutes=60
        )
        self.assertIsNotNone(second)  # another bot has a separate quota scope
        # A blocked opportunity is not consumed; it can be sampled later.
        self.assertIsNone(
            self.store.reserve_proactive(key_a, "blocked", t + 1, 1, 0, group_limit=1)
        )
        self.store.finish_proactive(first, "sent", "hello")
        self.assertIsNone(self.store.reserve_proactive(key_a, "a1", t + 2, 1, 0))
        self.assertEqual(self.store.proactive_stats(key_a, t + 3)["last_texts"], ["hello"])
        self.store.commit_turn(key_a, "reply", "answer", "response", self.analysis(), t + 4)
        self.assertEqual(self.store.proactive_stats(key_a, t + 5)["unanswered"], 0)
        reserved = self.store.reserve_proactive(
            "g1:u2", "restart", t + 100000, 1, 0, group_interval_minutes=0
        )
        restarted = Store(self.store.path)
        try:
            self.assertIsNone(
                restarted.reserve_proactive(
                    "g1:u2", "restart", t + 100001, 1, 0, group_interval_minutes=0
                )
            )
            restarted.finish_proactive(reserved, "failed")
            self.assertEqual(
                restarted.db.execute(
                    "SELECT status FROM proactive_jobs WHERE id=?", (reserved,)
                ).fetchone()[0],
                "unknown",
            )
        finally:
            restarted.close()

    def test_forget_clears_context_and_invalidates_summary_revision(self):
        t = 1_800_000_000
        self.store.commit_turn("g1:u1", "m1", "hello", "reply", self.analysis(), t)
        before = self.store.profile("g1:u1")["revision"]
        pending = self.store.pending_summaries(t + 1)[0]
        self.assertEqual(len(pending["turns"]), 1)
        self.assertEqual(self.store.forget("g1:u1", "tea"), 1)
        self.assertEqual(self.store.profile("g1:u1")["revision"], before + 1)
        self.assertEqual(self.store.recent("g1:u1"), [])
        self.assertFalse(
            self.store.apply_summary(
                "g1:u1",
                pending["revision"],
                [x["id"] for x in pending["turns"]],
                {"summary": "stale", "style": "old"},
            )
        )
        self.assertEqual(self.store.profile("g1:u1")["affection"], 23)

    def test_summary_cas_and_followup_visibility_window(self):
        t = 1_800_000_000
        due = datetime.fromtimestamp(t, timezone.utc).astimezone(timezone(timedelta(hours=8)))
        analysis = self.analysis(
            followups=[
                {
                    "content": "check in",
                    "due_at": (due - timedelta(hours=1)).isoformat(),
                    "public": True,
                },
                {
                    "content": "private",
                    "due_at": (due - timedelta(hours=1)).isoformat(),
                    "public": False,
                },
                {"content": "old", "due_at": (due - timedelta(days=8)).isoformat(), "public": True},
            ]
        )
        self.store.commit_turn("g1:u1", "m1", "hello", "reply", analysis, t)
        pending = self.store.pending_summaries(t + 1)[0]
        self.assertEqual(len(self.store.followups("g1:u1", t)), 1)
        ids = [x["id"] for x in pending["turns"]]
        self.assertTrue(
            self.store.apply_summary(
                "g1:u1",
                pending["revision"],
                ids,
                {"summary": "short recap", "style": "calm", "memories": []},
            )
        )
        self.assertFalse(
            self.store.apply_summary(
                "g1:u1", pending["revision"], ids, {"summary": "duplicate", "style": ""}
            )
        )
        self.assertEqual(self.store.pending_summaries(t + 1), [])

    def test_proactive_budget_opportunity_dedupe_and_failure_refund(self):
        # New day boundary in UTC+8; one user slot and two group slots.
        t = datetime(2026, 1, 1, 2, tzinfo=timezone.utc).timestamp()
        job = self.store.reserve_proactive("g1:u1", "o1", t, 1, 0, group_limit=2, user_limit=1)
        self.assertIsNotNone(job)
        self.assertIsNone(self.store.reserve_proactive("g1:u1", "o1", t + 1, 1, 0))
        self.assertIsNone(
            self.store.reserve_proactive("g1:u1", "o2", t + 2, 1, 0, group_limit=2, user_limit=1)
        )
        self.store.finish_proactive(job, "failed", "transport error")
        job2 = self.store.reserve_proactive("g1:u1", "o3", t + 3, 1, 0, group_limit=2, user_limit=1)
        self.assertIsNotNone(job2)
        # A probability miss is remembered and is never sampled twice.
        self.assertIsNone(
            self.store.reserve_proactive(
                "g1:u2",
                "miss",
                t + 5000,
                0,
                0,
                group_limit=4,
                user_limit=2,
                group_interval_minutes=0,
            )
        )
        self.assertIsNone(
            self.store.reserve_proactive(
                "g1:u2",
                "miss",
                t + 9000,
                1,
                0,
                group_limit=4,
                user_limit=2,
                group_interval_minutes=0,
            )
        )

    def test_profile_limits_and_relationship_reset_preserves_facts(self):
        self.store.ensure_profile("x", "g", "u", "u", persona="p" * 300)
        self.assertEqual(len(self.store.profile("x")["persona"]), 300)
        self.store.commit_turn("g1:u1", "m", "hi", "hey", self.analysis(), 1_800_000_000)
        self.store.reset_relationship("g1:u1")
        p = self.store.profile("g1:u1")
        self.assertEqual(
            (p["affection"], p["trust"], p["familiarity"], p["closeness"]), (20, 10, 0, 0)
        )
        self.assertEqual(len(self.store.memories("g1:u1")), 1)
        self.assertEqual(self.store.audit("g1:u1"), [])

    def test_tag_model_validation_priority_tombstone_and_manual_readd(self):
        t = 1_800_000_000
        items = [
            {
                "name": "爱茶",
                "category": "interest",
                "confidence": 0.9,
                "evidence": "说喜欢绿茶",
                "public": True,
            },
            {"name": "低置信", "category": "interest", "confidence": 0.79},
            {"name": "字符串置信", "category": "interest", "confidence": "0.99"},
            {"name": "布尔置信", "category": "interest", "confidence": True},
            {"name": "坏类别", "category": "other", "confidence": 0.9},
            {"name": "超长" * 13, "category": "interest", "confidence": 0.99},
            *[
                {
                    "name": f"标签{i}",
                    "category": "preference",
                    "confidence": 0.9,
                    "evidence": "e" * 250,
                }
                for i in range(8)
            ],
        ]
        self.store.commit_turn("g1:u1", "tags1", "first", "reply", self.analysis(tags=items), t)
        saved = self.store.tags("g1:u1")
        self.assertEqual(len(saved), 6)
        self.assertEqual(
            [x["name"] for x in saved], ["爱茶", "标签0", "标签1", "标签2", "标签3", "标签4"]
        )
        tag = next(x for x in saved if x["name"] == "爱茶")
        self.assertEqual(tag["source"], "model")
        self.assertEqual(tag["source_ids"], ["tags1"])
        self.assertLessEqual(len(tag["evidence"]), 200)

        rev = self.store.profile("g1:u1")["revision"]
        self.assertTrue(self.store.update_tag("g1:u1", "爱茶", "preference", True, rev))
        manual = next(x for x in self.store.tags("g1:u1") if x["name"] == "爱茶")
        self.assertEqual(manual["source"], "manual")
        rev = self.store.profile("g1:u1")["revision"]
        self.store.commit_turn(
            "g1:u1",
            "tags2",
            "update",
            "reply",
            self.analysis(
                tags=[
                    {
                        "name": "爱茶",
                        "category": "interest",
                        "confidence": 0.99,
                        "evidence": "new",
                        "public": False,
                    }
                ]
            ),
            t + 7200,
        )
        after_auto = next(x for x in self.store.tags("g1:u1") if x["name"] == "爱茶")
        self.assertEqual(after_auto["source"], "manual")
        self.assertEqual(after_auto["category"], "preference")
        self.assertTrue(after_auto["public"])
        self.assertEqual(self.store.profile("g1:u1")["revision"], rev + 1)

        self.assertTrue(
            self.store.delete_tag(
                "g1:u1", after_auto["id"], self.store.profile("g1:u1")["revision"]
            )
        )
        self.store.commit_turn(
            "g1:u1",
            "tags3",
            "again",
            "reply",
            self.analysis(
                tags=[
                    {
                        "name": "爱茶",
                        "category": "interest",
                        "confidence": 0.99,
                        "evidence": "again",
                    }
                ]
            ),
            t + 14400,
        )
        self.assertNotIn("爱茶", [x["name"] for x in self.store.tags("g1:u1")])
        self.assertTrue(self.store.update_tag("g1:u1", "爱茶", "interest", False))
        self.store.commit_turn(
            "g1:u1",
            "tags4",
            "again",
            "reply",
            self.analysis(tags=[{"name": "爱茶", "category": "boundary", "confidence": 0.99}]),
            t + 21600,
        )
        readded = next(x for x in self.store.tags("g1:u1") if x["name"] == "爱茶")
        self.assertEqual(readded["source"], "manual")
        self.assertEqual(readded["category"], "interest")

    def test_tag_limits_and_input_validation(self):
        with self.assertRaises(ValueError):
            self.store.update_tag("g1:u1", "", "interest")
        with self.assertRaises(ValueError):
            self.store.update_tag("g1:u1", "x", "unknown")
        with self.assertRaises(ValueError):
            self.store.update_tag("g1:u1", "x", "interest", "yes")
        for i in range(30):
            self.assertTrue(self.store.update_tag("g1:u1", f"tag{i}"))
        self.store.commit_turn(
            "g1:u1",
            "full",
            "full",
            "reply",
            self.analysis(
                tags=[
                    {"name": "overflow", "category": "interest", "confidence": 0.99},
                ]
            ),
            1_800_000_000,
        )
        self.assertEqual(len(self.store.tags("g1:u1")), 30)
        with self.assertRaises(ValueError):
            self.store.update_tag("g1:u1", "another")

    def test_forgotten_tags_leave_only_hash_suppression_markers(self):
        self.store.update_tag("g1:u1", "不想保存的标签", "preference")
        self.store.forget("g1:u1")
        rows = self.store.db.execute(
            "SELECT name FROM tag_tombstones WHERE key=?", ("g1:u1",)
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["name"].startswith("sha256:"))
        self.assertNotIn("不想保存的标签", rows[0]["name"])

    def test_tags_forget_preferences_and_cas(self):
        t = 1_800_000_000
        self.store.commit_turn(
            "g1:u1",
            "a",
            "tags",
            "reply",
            self.analysis(
                tags=[
                    {"name": "自动茶", "category": "interest", "confidence": 0.9},
                ]
            ),
            t,
        )
        self.store.update_tag("g1:u1", "手动茶", "preference", True)
        self.store.update_tag("g1:u1", "工作", "interest", False)
        rev = self.store.profile("g1:u1")["revision"]
        auto = next(t for t in self.store.tags("g1:u1") if t["name"] == "自动茶")
        self.assertFalse(self.store.delete_tag("g1:u1", auto["id"], expected_revision=rev - 1))
        self.assertTrue(self.store.delete_tag("g1:u1", auto["id"], expected_revision=rev))
        self.assertFalse(self.store.delete_tag("g1:u1", auto["id"]))
        self.store.commit_turn(
            "g1:u1",
            "b",
            "tag",
            "reply",
            self.analysis(
                tags=[
                    {"name": "自动茶", "category": "interest", "confidence": 0.9},
                ]
            ),
            t + 7200,
        )
        self.assertNotIn("自动茶", [t["name"] for t in self.store.tags("g1:u1")])
        # Query forgetting clears all automatic tags and only matching manual tags.
        self.store.forget("g1:u1", "茶")
        self.assertEqual([t["name"] for t in self.store.tags("g1:u1")], ["工作"])
        self.store.set_preferences("g1:u1", memory_enabled=False)
        self.store.commit_turn(
            "g1:u1",
            "c",
            "new",
            "reply",
            self.analysis(
                tags=[
                    {"name": "新自动", "category": "interest", "confidence": 0.99},
                ]
            ),
            t + 14400,
        )
        self.assertEqual([t["name"] for t in self.store.tags("g1:u1")], ["工作"])
        self.store.forget("g1:u1")
        self.assertEqual(self.store.tags("g1:u1"), [])

    def test_summary_tags_source_and_reset_impressions(self):
        t = 1_800_000_000
        self.store.commit_turn("g1:u1", "a", "hello", "reply", self.analysis(), t)
        batch = self.store.pending_summaries(t + 1)[0]
        self.assertTrue(
            self.store.apply_summary(
                "g1:u1",
                batch["revision"],
                [x["id"] for x in batch["turns"]],
                {
                    "summary": "daily",
                    "style": "",
                    "memories": [],
                    "tags": [
                        {
                            "name": "外向",
                            "category": "impression",
                            "confidence": 0.91,
                            "evidence": "主动发言",
                            "public": True,
                        },
                        {
                            "name": "沟通清晰",
                            "category": "communication",
                            "confidence": 0.9,
                            "evidence": "表达明确",
                        },
                    ],
                },
            )
        )
        tags = self.store.tags("g1:u1")
        self.assertEqual(len(tags), 2)
        self.assertTrue(all(t["source"] == "summary" for t in tags))
        self.assertTrue(all(t["source_ids"] == [str(batch["turns"][0]["id"])] for t in tags))
        self.store.reset_relationship("g1:u1")
        self.assertEqual([t["name"] for t in self.store.tags("g1:u1")], ["沟通清晰"])

    def test_dashboard_profiles_search_scope_paging_and_private_detail(self):
        t = 1_800_000_000
        key1, key2, key3 = "role-a:group1:u1", "role-a:group2:u2", "role-b:group3:u3"
        self.store.ensure_profile(
            key1, "group1", "10001", "umo1", persona="private persona", display_name="茶友"
        )
        self.store.ensure_profile(key1, "group1", "10001", "umo1", display_name="")
        self.store.ensure_profile(key2, "group2", "10002", "umo2", display_name="小林")
        self.store.ensure_profile(key3, "group3", "10003", "umo3", display_name="小王")
        self.store.update_tag(key1, "喜欢绿茶", "interest", True)
        self.store.commit_turn(
            key1,
            "m1",
            "secret private memory",
            "reply",
            self.analysis(
                memories=[{"content": "private secret content", "public": False}],
                tags=[{"name": "喜欢绿茶", "category": "interest", "confidence": 0.99}],
            ),
            t,
        )
        batch = self.store.pending_summaries(t + 1, keys=[key1])[0]
        self.assertTrue(
            self.store.apply_summary(
                key1,
                batch["revision"],
                [x["id"] for x in batch["turns"]],
                {
                    "summary": "daily detail record",
                    "style": "",
                    "memories": [],
                    "tags": [],
                },
            )
        )
        self.store.set_preferences(key1, memory_enabled=False)
        scoped = self.store.dashboard_profiles(keys=[key1, key2])
        self.assertEqual(scoped["total"], 2)
        self.assertEqual(scoped["groups"], ["group1", "group2"])
        self.assertNotIn("persona", scoped["items"][0])
        self.assertNotIn("style", scoped["items"][0])
        self.assertNotIn("emotion", scoped["items"][0])
        self.assertEqual(
            self.store.dashboard_profiles(query="10001", keys=[key1, key2])["total"], 1
        )
        self.assertEqual(self.store.dashboard_profiles(query="茶友", keys=[key1, key2])["total"], 1)
        self.assertEqual(self.store.dashboard_profiles(query="绿茶", keys=[key1, key2])["total"], 1)
        page1 = self.store.dashboard_profiles(keys=[key1, key2], offset=0, limit=1)
        page2 = self.store.dashboard_profiles(keys=[key1, key2], offset=1, limit=1)
        self.assertEqual(page1["total"], 2)
        self.assertNotEqual(page1["items"][0]["key"], page2["items"][0]["key"])
        self.assertEqual(self.store.dashboard_profiles(keys=[])["total"], 0)
        detail = self.store.dashboard_detail(key1)
        self.assertNotIn("persona", detail["profile"])
        self.assertFalse(detail["memories"][0]["public"])
        self.assertEqual(detail["memories"][0]["source_ids"], ["m1"])
        self.assertEqual(detail["tags"][0]["source"], "manual")
        self.assertEqual(detail["daily_summaries"][0]["content"], "daily detail record")
        self.assertEqual(len(detail["changes"]), 1)
        self.assertIsNone(self.store.dashboard_detail("missing"))

    def test_v01_database_migration_preserves_profiles(self):
        old_path = Path(self.tmp.name) / "v01.sqlite3"
        old = __import__("sqlite3").connect(old_path)
        old.execute("""CREATE TABLE profiles(
            key TEXT PRIMARY KEY, group_id TEXT NOT NULL, user_id TEXT NOT NULL, umo TEXT NOT NULL,
            provider_id TEXT NOT NULL DEFAULT '', persona TEXT NOT NULL DEFAULT '',
            memory_enabled INTEGER NOT NULL DEFAULT 1, proactive_enabled INTEGER NOT NULL DEFAULT 1,
            affection INTEGER NOT NULL DEFAULT 20, trust INTEGER NOT NULL DEFAULT 10,
            familiarity INTEGER NOT NULL DEFAULT 0, closeness INTEGER NOT NULL DEFAULT 0,
            style TEXT NOT NULL DEFAULT '', emotion TEXT NOT NULL DEFAULT '',
            last_interaction REAL, revision INTEGER NOT NULL DEFAULT 0
        )""")
        old.execute(
            "INSERT INTO profiles(key,group_id,user_id,umo,affection,trust,last_interaction) VALUES('legacy','g','u','umo',41,32,123)"
        )
        old.commit()
        old.close()
        migrated = Store(str(old_path))
        try:
            p = migrated.profile("legacy")
            self.assertEqual(
                (p["affection"], p["trust"], p["last_interaction"], p["display_name"]),
                (41, 32, 123, ""),
            )
            self.assertEqual(migrated.tags("legacy"), [])
            migrated.ensure_profile("legacy", "g", "u", "umo", display_name="昵称")
            self.assertEqual(migrated.profile("legacy")["display_name"], "昵称")
        finally:
            migrated.close()

    def test_group_quota_uses_actual_five_part_key_and_separates_groups(self):
        t = 1800000000
        keys = [
            json.dumps(["role", "platform", "bot", group, user])
            for group, user in [("g1", "u1"), ("g1", "u2"), ("g2", "u1")]
        ]
        for key, group, user in zip(keys, ["g1", "g1", "g2"], ["u1", "u2", "u1"]):
            self.store.ensure_profile(key, group, user, key)
        self.assertIsNotNone(self.store.reserve_proactive(keys[0], "a", t, 1, 0, group_limit=1))
        self.assertIsNone(self.store.reserve_proactive(keys[1], "b", t, 1, 0, group_limit=1))
        self.assertIsNotNone(self.store.reserve_proactive(keys[2], "c", t, 1, 0, group_limit=1))

    def test_daily_archives_do_not_overwrite_previous_days(self):
        t = datetime(2026, 10, 20, 12, tzinfo=timezone(timedelta(hours=8))).timestamp()
        for i in range(2):
            self.store.commit_turn(
                "g1:u1", f"day{i}", f"hello {i}", "reply", self.analysis(), t + i * 86400
            )
        for i in range(2):
            batch = self.store.pending_summaries(t + 3 * 86400)[0]
            self.assertEqual(len(batch["turns"]), 1)
            self.store.apply_summary(
                "g1:u1",
                batch["revision"],
                [batch["turns"][0]["id"]],
                {"summary": f"day{i}", "memories": []},
            )
        self.assertEqual(
            self.store.db.execute("SELECT COUNT(*) FROM daily_history").fetchone()[0], 2
        )
        self.store.forget("g1:u1")
        self.assertEqual(
            self.store.db.execute("SELECT COUNT(*) FROM daily_history").fetchone()[0], 0
        )

    def test_summary_cannot_restore_superseded_fact(self):
        t = 1800000000
        self.store.commit_turn(
            "g1:u1",
            "old",
            "准备面试",
            "好",
            self.analysis(memories=[{"content": "正在准备面试", "public": True}]),
            t,
        )
        old = self.store.memories("g1:u1")[0]["id"]
        self.store.commit_turn(
            "g1:u1",
            "new",
            "已经入职",
            "好",
            self.analysis(
                replace_memory_ids=[old], memories=[{"content": "已经入职", "public": True}]
            ),
            t + 7200,
        )
        batch = self.store.pending_summaries(t + 8000)[0]
        self.store.apply_summary(
            "g1:u1",
            batch["revision"],
            [x["id"] for x in batch["turns"]],
            {"memories": [{"content": "正在准备面试", "public": True}]},
        )
        self.assertEqual([m["content"] for m in self.store.memories("g1:u1")], ["已经入职"])

    def test_forget_treats_sql_wildcards_as_literal(self):
        self.store.commit_turn(
            "g1:u1",
            "a",
            "x",
            "y",
            self.analysis(
                memories=[
                    {"content": "完成50%", "public": True},
                    {"content": "其他事实", "public": True},
                ]
            ),
            1800000000,
        )
        self.assertEqual(self.store.forget("g1:u1", "50%"), 1)
        self.assertEqual([m["content"] for m in self.store.memories("g1:u1")], ["其他事实"])

    def test_reset_cannot_regenerate_old_relationship_from_turns(self):
        self.store.commit_turn("g1:u1", "a", "过去的关系", "y", self.analysis(), 1800000000)
        self.store.reset_relationship("g1:u1")
        self.assertEqual(self.store.recent("g1:u1"), [])
        self.assertEqual(self.store.pending_summaries(1800001000), [])

    def test_inactive_profiles_do_not_starve_enabled_summaries(self):
        for key in ("g1:u1", "g1:u2"):
            self.store.commit_turn(key, "a", "hi", "y", self.analysis(), 1800000000)
        batches = self.store.pending_summaries(1800001000, limit=1, keys=["g1:u2"])
        self.assertEqual([b["key"] for b in batches], ["g1:u2"])


if __name__ == "__main__":
    unittest.main()
