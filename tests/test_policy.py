import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy import (
    LEVEL_BANDS,
    TZ,
    affection_level,
    affection_query,
    affection_report,
    clean_tags,
    daily_allowance,
    level_label,
    level_of,
    natural_control,
    parse_json,
    proactive_probability,
    quiet_hour,
    validate_analysis,
)


class PolicyTests(unittest.TestCase):
    def test_tags_require_evidence_confidence_and_strict_visibility(self):
        values = [
            {
                "name": "喜欢科幻",
                "category": "interest",
                "confidence": 0.9,
                "evidence": "用户明确说喜欢科幻",
                "public": True,
            },
            {"name": "猜测的性格", "category": "impression", "confidence": 0.2, "evidence": "猜测"},
            {"name": "没有依据", "category": "interest", "confidence": 1},
            {
                "name": "私密偏好",
                "category": "preference",
                "confidence": 1,
                "evidence": "用户表达",
                "public": "false",
            },
        ]
        result = clean_tags(values)
        self.assertEqual([tag["name"] for tag in result], ["喜欢科幻", "私密偏好"])
        self.assertFalse(result[1]["public"])
        self.assertEqual(validate_analysis({"confidence": 0.2, "tags": values}, 0)["tags"], [])

    def test_invalid_model_data_cannot_change_scores(self):
        result = validate_analysis(
            {
                "confidence": "NaN",
                "deltas": {"affection": 999},
                "memories": [{"content": "假事实", "kind": "fact", "public": True}],
            },
            0,
        )
        self.assertEqual(result["deltas"], {})
        self.assertEqual(result["memories"], [])

    def test_clamps_and_rejects_ambiguous_dates(self):
        now = datetime(2026, 10, 27, tzinfo=TZ).timestamp()
        result = validate_analysis(
            {
                "confidence": 0.9,
                "deltas": {"affection": 999},
                "followups": [{"content": "考试", "public": True, "due_at": "2026-10-28T12:00:00"}],
            },
            now,
        )
        self.assertEqual(result["deltas"]["affection"], 3)
        self.assertEqual(result["followups"], [])

    def test_privacy_requires_real_boolean(self):
        result = validate_analysis(
            {"confidence": 1, "memories": [{"content": "住址", "kind": "fact", "public": "false"}]},
            0,
        )
        self.assertFalse(result["memories"][0]["public"])

    def test_explicit_control_only(self):
        self.assertEqual(natural_control("以后别记我的聊天。"), ("memory_off", ""))
        self.assertIsNone(natural_control("他说‘以后别记我的聊天’，你怎么看？"))
        self.assertEqual(natural_control("忘掉我上次说的地址"), ("forget", "地址"))

    def test_quiet_window_crosses_midnight(self):
        for hour in (0, 8, 23):
            self.assertTrue(quiet_hour(datetime(2026, 10, 27, hour, tzinfo=TZ).timestamp(), 23, 9))
        self.assertFalse(quiet_hour(datetime(2026, 10, 27, 9, tzinfo=TZ).timestamp(), 23, 9))

    def test_unanswered_lowers_probability(self):
        profile = {"closeness": 50}
        self.assertGreater(
            proactive_probability(profile, 0.1, 1), proactive_probability(profile, 0.1, 1, 3)
        )

    def test_json_rejects_nonobject_and_accepts_code_fence(self):
        self.assertEqual(parse_json('```json\n{"a":1}\n```'), {"a": 1})
        with self.assertRaises(ValueError):
            parse_json("[]")

    def test_affection_query_matches_only_whole_lookup(self):
        for message in (
            "查看好感度",
            "查询好感度",
            "查看我的好感度",
            "我的好感度",
            "好感度多少",
            "好感度是多少？",
            "看一下好感度。",
            "帮我查一下好感度",
            "现在好感度多少了",
        ):
            self.assertEqual(natural_control(message), ("affection", ""), message)
        for message in (
            "把好感度改成100",
            "好感度怎么才能提升",
            "他说‘查看好感度’是什么意思",
            "查看他的好感度",
            "好感度降低会怎样",
            "",
        ):
            self.assertIsNone(natural_control(message), message)
        self.assertFalse(affection_query("好感"))

    def test_affection_report_is_readonly_and_self_scoped(self):
        profile = {
            "affection": 42,
            "trust": 30,
            "familiarity": 18,
            "closeness": 12,
            "last_interaction": 1000,
        }
        single = affection_report(profile)
        self.assertIn("42 / 100", single)
        self.assertNotIn("信任", single)
        all_dimensions = affection_report(profile, scope="all")
        for label in ("好感 42", "信任 30", "熟悉 18", "亲近 12"):
            self.assertIn(label, all_dimensions)
        words_only = affection_report(profile, show_number=False)
        self.assertNotIn("42", words_only)
        self.assertIn(affection_level(42), words_only)
        # Never out of range, and a brand new profile is described as untouched.
        self.assertIn("100 / 100", affection_report({"affection": 999, "last_interaction": 1}))
        self.assertIn("0 / 100", affection_report({"affection": -50, "last_interaction": 1}))
        self.assertIn("还没真正聊过", affection_report({"affection": 20}))
        self.assertEqual(affection_level(0), "还不太熟，先多聊聊吧")
        self.assertEqual(affection_level(90), "和你相处很放松，也很愿意继续聊下去")

    def test_scores_are_quantised_into_levels(self):
        expected = [
            (0, 1, "陌生"),
            (24, 1, "陌生"),
            (25, 2, "认识"),
            (39, 2, "认识"),
            (40, 3, "熟悉"),
            (54, 3, "熟悉"),
            (55, 4, "朋友"),
            (69, 4, "朋友"),
            (70, 5, "亲密"),
            (84, 5, "亲密"),
            (85, 6, "羁绊"),
            (100, 6, "羁绊"),
        ]
        for score, index, name in expected:
            band = level_of(score)
            self.assertEqual(band["level"], index, score)
            self.assertEqual(band["name"], name, score)
            self.assertEqual(level_label(score), f"Lv.{index} {name}")
            self.assertTrue(band["description"])
        # Out-of-range input is clamped before quantising, never a new level.
        self.assertEqual(level_of(-5)["level"], 1)
        self.assertEqual(level_of(120)["level"], 6)
        # Bands are contiguous, descending and cover the whole 0-100 range.
        thresholds = [band[0] for band in LEVEL_BANDS]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))
        self.assertEqual(thresholds[-1], 0)
        self.assertEqual([band[1] for band in LEVEL_BANDS], [6, 5, 4, 3, 2, 1])
        self.assertEqual(len({band[2] for band in LEVEL_BANDS}), len(LEVEL_BANDS))
        # The report carries the level, and words-only mode keeps the level name.
        profile = {"affection": 42, "last_interaction": 1000}
        self.assertIn("Lv.3 熟悉", affection_report(profile))
        words_only = affection_report(profile, show_number=False)
        self.assertIn("熟悉", words_only)
        self.assertNotIn("Lv.3", words_only)

    def test_daily_allowance_zero_means_unlimited(self):
        self.assertEqual(daily_allowance(3, 0, 8), 3)
        self.assertEqual(daily_allowance(3, 6, 8), 2)  # only 2 left today
        self.assertEqual(daily_allowance(3, 8, 8), 0)  # budget exhausted
        self.assertEqual(daily_allowance(0, 0, 8), 0)
        self.assertEqual(daily_allowance(-3, 0, 10), -3)
        self.assertEqual(daily_allowance(-3, 9, 10), -1)
        self.assertEqual(daily_allowance(-3, 10, 10), 0)
        # 0 (and unset/invalid values) disable the cap for that side.
        self.assertEqual(daily_allowance(3, 99, 0), 3)
        self.assertEqual(daily_allowance(-3, 99, 0), -3)
        self.assertEqual(daily_allowance(3, 99, None), 3)
        self.assertEqual(daily_allowance(3, 99, "bad"), 3)
        self.assertEqual(daily_allowance(3, 99, -1), 3)
