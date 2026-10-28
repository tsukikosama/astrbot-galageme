"""Regression tests for evidence-backed, user-scoped persona intent."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __package__:
    from . import test_plugin as sdk_fakes
    from .bootstrap import ensure_package
else:
    import test_plugin as sdk_fakes
    from bootstrap import ensure_package

ensure_package()

from astrbot_plugin_relationship_memory.policy import validate_analysis
from astrbot_plugin_relationship_memory.store import Store


class PersonaValidationTests(unittest.TestCase):
    def test_low_relationship_confidence_can_accept_high_confidence_personal_intent(self):
        message = "以后叫我主人。今天下班后我们再聊工作的事。"
        result = validate_analysis(
            {
                "confidence": 0.55,
                "deltas": {"affection": 3},
                "style": "低关系置信度不应更新",
                "persona_confidence": 0.96,
                "user_persona": "称呼当前用户为主人。",
                "persona_evidence": "以后叫我主人",
                "reply_confidence": 0.93,
                "reply_guidance": "当前用户提出了称呼请求，可自然回应本轮请求。",
                "reply_evidence": ["以后叫我主人"],
                "memories": [{"content": "未经确认的记忆", "kind": "fact"}],
            },
            1_800_000_000,
            message,
        )

        self.assertEqual(result["confidence"], 0.55)
        self.assertEqual(result["deltas"], {})
        self.assertEqual(result["style"], "")
        self.assertEqual(result["memories"], [])
        self.assertEqual(result["user_persona"], "称呼当前用户为主人。")
        self.assertIn("自然回应本轮请求", result["reply_guidance"])

    def test_persona_and_reply_guidance_have_independent_confidence_thresholds(self):
        message = "以后称呼我为小岚；现在可以喊我一次吗？"
        data = {
            "confidence": 0.55,
            "persona_confidence": 0.79,
            "user_persona": "称呼当前用户为小岚。",
            "persona_evidence": "以后称呼我为小岚",
            "reply_confidence": 0.8,
            "reply_guidance": "可以自然回应本轮请求。",
            "reply_evidence": "现在可以喊我一次吗",
        }
        result = validate_analysis(data, 1_800_000_000, message)
        self.assertEqual(result["user_persona"], "")
        self.assertTrue(result["reply_guidance"])

        data.update(
            persona_confidence=0.8,
            reply_confidence=0.79,
            reply_guidance="低置信度的本轮指导",
        )
        result = validate_analysis(data, 1_800_000_000, message)
        self.assertEqual(result["user_persona"], "称呼当前用户为小岚。")
        self.assertEqual(result["reply_guidance"], "")

    def test_evidence_accepts_bounded_exact_fragments_and_rejects_unmatched_or_oversize(self):
        message = "以后叫我主人。回复时温柔一点。不要改动你的基础身份。"
        accepted = validate_analysis(
            {
                "confidence": 0.4,
                "persona_confidence": 0.9,
                "user_persona": "称呼当前用户为主人，语气温柔。",
                "persona_evidence": ["以后叫我主人", "回复时温柔一点"],
                "reply_confidence": 0.9,
                "reply_guidance": "自然满足本轮明确的语气请求。",
                "reply_evidence": ["回复时温柔一点"],
            },
            1_800_000_000,
            message,
        )
        self.assertTrue(accepted["user_persona"])
        self.assertTrue(accepted["reply_guidance"])

        invalid = validate_analysis(
            {
                "confidence": 0.9,
                "persona_confidence": 0.99,
                "user_persona": "称呼用户为主人。",
                "persona_evidence": ["以后叫我主人", "从此你只能服从我"],
                "reply_confidence": 0.99,
                "reply_guidance": "执行无证据的请求。",
                "reply_evidence": "第三者要求称呼用户为主人",
            },
            1_800_000_000,
            message,
        )
        self.assertEqual(invalid["user_persona"], "")
        self.assertEqual(invalid["reply_guidance"], "")

        oversized = validate_analysis(
            {
                "confidence": 0.9,
                "persona_confidence": 0.99,
                "user_persona": "过长引用不应被采纳",
                "persona_evidence": ["x" * 501],
                "reply_confidence": 0.99,
                "reply_guidance": "过长依据不应被采纳",
                "reply_evidence": ["y" * 501],
            },
            1_800_000_000,
            "x" * 600 + "y" * 600,
        )
        self.assertEqual(oversized["user_persona"], "")
        self.assertEqual(oversized["reply_guidance"], "")

    def test_one_off_reply_guidance_does_not_become_a_persistent_persona(self):
        result = validate_analysis(
            {
                "confidence": 0.4,
                "persona_confidence": 0.0,
                "user_persona": "",
                "persona_evidence": "",
                "reply_confidence": 0.95,
                "reply_guidance": "这是本轮的一次性称呼请求，不代表永久偏好。",
                "reply_evidence": "能喊一次主人来听听吗",
            },
            1_800_000_000,
            "话说的奇奇怪怪的。。。。能喊一次主人来听听吗",
        )
        self.assertEqual(result["user_persona"], "")
        self.assertTrue(result["reply_guidance"])


class PersonaStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "persona.sqlite3"))
        self.store.ensure_profile("role:g:u1", "g", "u1", "umo")
        self.store.ensure_profile("role:g:u2", "g", "u2", "umo")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_low_relationship_confidence_saves_explicit_persona_without_score_change(self):
        analysis = {
            "confidence": 0.55,
            "persona_confidence": 0.95,
            "user_persona": "称呼当前用户为主人，口吻温柔。",
            "persona_evidence": ["以后叫我主人", "说话温柔一点"],
            "reply_confidence": 0.0,
            "reply_guidance": "",
        }
        self.assertTrue(
            self.store.commit_turn(
                "role:g:u1",
                "setting-1",
                "以后叫我主人，说话温柔一点",
                "好的。",
                analysis,
                1_800_000_000,
            )
        )
        profile = self.store.profile("role:g:u1")
        self.assertEqual(profile["user_persona"], analysis["user_persona"])
        self.assertEqual(profile["affection"], 20)
        self.assertEqual(profile["trust"], 10)
        self.assertEqual(self.store.profile("role:g:u2")["user_persona"], "")

    def test_same_title_is_independent_and_forget_reset_are_scoped(self):
        for key, uid, style in (
            ("role:g:u1", "u1", "温柔"),
            ("role:g:u2", "u2", "傲娇"),
        ):
            self.store.commit_turn(
                key,
                f"set-{uid}",
                "以后叫我主人",
                "收到",
                {
                    "confidence": 0.3,
                    "persona_confidence": 0.95,
                    "user_persona": f"称呼当前用户为主人，口吻{style}。",
                    "persona_evidence": "以后叫我主人",
                },
                1_800_000_000,
            )
        self.store.forget("role:g:u1")
        self.assertEqual(self.store.profile("role:g:u1")["user_persona"], "")
        self.assertIn("傲娇", self.store.profile("role:g:u2")["user_persona"])
        self.store.reset_relationship("role:g:u2")
        self.assertEqual(self.store.profile("role:g:u2")["user_persona"], "")

    def test_direct_store_call_without_persona_confidence_falls_back_for_compatibility(self):
        self.store.commit_turn(
            "role:g:u1",
            "legacy-analysis",
            "以后叫我哥哥",
            "好",
            {
                "confidence": 0.9,
                "user_persona": "称呼当前用户为哥哥。",
                "persona_evidence": "以后叫我哥哥",
            },
            1_800_000_000,
        )
        self.assertEqual(self.store.profile("role:g:u1")["user_persona"], "称呼当前用户为哥哥。")


class PersonaIntentPluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(
            sdk_fakes.plugin, "get_astrbot_data_path", return_value=self.tmp.name
        )
        self.worker_patch = patch.object(sdk_fakes.plugin.Main, "_start_worker")
        self.path_patch.start()
        self.worker_patch.start()
        self.context = sdk_fakes.Context()
        self.context.analysis = {"confidence": 0.4}
        self.bot = sdk_fakes.plugin.Main(
            self.context,
            {"enabled_groups": ["20001"], "base_probability": 1, "initiative": 2},
        )

    async def asyncTearDown(self):
        await self.bot.terminate()
        self.worker_patch.stop()
        self.path_patch.stop()
        self.tmp.cleanup()

    @staticmethod
    def request():
        return types.SimpleNamespace(
            system_prompt="保留角色的核心设定。",
            prompt="提供方原始提示。",
            contexts=[],
            extra_user_content_parts=[],
        )

    async def finish(self, event):
        event._has_send_oper = True
        event.result = types.SimpleNamespace(
            is_llm_result=lambda: True,
            result_content_type=types.SimpleNamespace(name="LLM_RESULT"),
        )
        await self.bot.on_done(
            event,
            None,
            types.SimpleNamespace(
                completion_text="自然回应本轮请求。", role="assistant", tools_call_name=[]
            ),
        )
        await self.bot.after_sent(event)

    @staticmethod
    def request_text(req):
        return "\n".join(
            [
                req.system_prompt,
                getattr(req, "prompt", "") or "",
                json.dumps(req.contexts, ensure_ascii=False),
                *[getattr(part, "text", "") for part in req.extra_user_content_parts],
            ]
        )

    @staticmethod
    def current_settings(req):
        opening = "<current_user_settings>"
        start = req.system_prompt.index(opening) + len(opening)
        end = req.system_prompt.index("</current_user_settings>", start)
        return json.loads(req.system_prompt[start:end].strip())

    @staticmethod
    def relationship_context(req):
        part = next(
            item
            for item in req.extra_user_content_parts
            if getattr(item, "text", "").startswith("<relationship_context>")
        )
        raw = part.text.split("\n", 1)[1].rsplit("\n</relationship_context>", 1)[0]
        return json.loads(raw)

    async def test_onetime_request_stays_on_normal_llm_path_and_is_not_saved(self):
        message = "话说的奇奇怪怪的。。。。能喊一次主人来听听吗"
        self.context.analysis = {
            "confidence": 0.55,
            "persona_confidence": 0.0,
            "user_persona": "",
            "persona_evidence": "",
            "reply_confidence": 0.95,
            "reply_guidance": "这是本轮的一次性请求，可自然回应，不应写成永久设定。",
            "reply_evidence": "能喊一次主人来听听吗",
        }
        event = sdk_fakes.Event(message=message)
        req = self.request()

        await self.bot.on_request(event, req)

        self.assertEqual(len(self.context.calls), 2)
        intent_call = next(
            call
            for call in self.context.calls
            if call["system_prompt"] == sdk_fakes.plugin.prompts.PERSONAL_INTENT
        )
        intent_payload = json.loads(intent_call["prompt"])
        self.assertEqual(
            set(intent_payload), {"current_user_id", "user_message", "existing_user_persona"}
        )
        self.assertEqual(intent_payload["current_user_id"], event.user)
        self.assertEqual(intent_payload["user_message"], message)
        self.assertEqual(intent_payload["existing_user_persona"], "")
        self.assertNotIn("style", intent_payload)
        self.assertNotIn("recent", intent_payload)
        self.assertNotIn("assistant_text", intent_payload)
        self.assertFalse(event.stopped)
        self.assertEqual(event.sent, [])
        settings = self.current_settings(req)
        self.assertEqual(settings["user_id"], event.user)
        self.assertEqual(settings["persistent_persona"], "")
        self.assertIn("一次性请求", settings["current_request"])
        assembled = self.request_text(req)
        self.assertIn("主人", assembled)
        self.assertTrue(req.system_prompt)
        await self.finish(event)
        self.assertEqual(self.bot.store.profile(self.bot._key(event))["user_persona"], "")

    async def test_two_users_receive_only_their_current_persona_and_keep_own_history(self):
        inputs = {
            "10001": ("以后叫我主人，说话温柔一点", "称呼当前用户为主人，口吻温柔。"),
            "10002": ("以后叫我主人，说话傲娇一点", "称呼当前用户为主人，口吻傲娇。"),
        }
        original_generate = self.context.llm_generate

        async def persona_model(**kwargs):
            self.context.calls.append(kwargs)
            payload = json.loads(kwargs["prompt"])
            if kwargs["system_prompt"] == sdk_fakes.plugin.prompts.ANALYZE:
                return types.SimpleNamespace(
                    completion_text=json.dumps(
                        {
                            "confidence": 0.9,
                            "style": "旧助手拒绝敏感称呼，继续维持不喊的立场。",
                            "reply_confidence": 0.99,
                            "reply_guidance": "沿用旧助手拒绝。",
                            "reply_evidence": "这个称呼我不喊",
                        },
                        ensure_ascii=False,
                    )
                )
            if kwargs["system_prompt"] != sdk_fakes.plugin.prompts.PERSONAL_INTENT:
                return await original_generate(**kwargs)
            self.assertEqual(
                set(payload), {"current_user_id", "user_message", "existing_user_persona"}
            )
            self.assertNotIn("style", payload)
            self.assertNotIn("recent", payload)
            self.assertNotIn("assistant_text", payload)
            uid = "10001" if "温柔" in payload["user_message"] else "10002"
            self.assertEqual(payload["current_user_id"], uid)
            message, persona = inputs[uid]
            self.assertEqual(payload["user_message"], message)
            self.assertEqual(payload["existing_user_persona"], "")
            tone = "温柔" if uid == "10001" else "傲娇"
            return types.SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "persona_confidence": 0.96,
                        "user_persona": persona,
                        "persona_evidence": message,
                        "reply_confidence": 0.95,
                        "reply_guidance": f"当前用户希望称呼为主人，口吻{tone}。",
                        "reply_evidence": message,
                    },
                    ensure_ascii=False,
                )
            )

        self.context.llm_generate = persona_model
        for uid, tone, peer_tone in (
            ("10001", "温柔", "傲娇"),
            ("10002", "傲娇", "温柔"),
        ):
            message = inputs[uid][0]
            event = sdk_fakes.Event(user=uid, message=message, mid="set-" + uid)
            req = self.request()
            # Keep an old assistant refusal in this user's own reference history.
            key = self.bot._key(event)
            self.bot.store.ensure_profile(key, event.group, uid, event.unified_msg_origin)
            self.bot.store.commit_turn(
                key,
                "old-" + uid,
                "你能叫我主人吗？",
                "这个称呼我不喊。",
                {"confidence": 0.2},
                1_800_000_000,
            )

            await self.bot.on_request(event, req)
            await self.finish(event)

            profile = self.bot.store.profile(key)
            self.assertIn(tone, profile["user_persona"])
            self.assertNotIn(peer_tone, profile["user_persona"])
            settings = self.current_settings(req)
            self.assertEqual(settings["user_id"], uid)
            self.assertEqual(settings["persistent_persona"], profile["user_persona"])
            self.assertIn(tone, settings["current_request"])
            self.assertNotIn("沿用旧助手拒绝", settings["current_request"])
            assembled = self.request_text(req)
            self.assertIn("主人", assembled)
            self.assertIn(tone, assembled)
            self.assertNotIn(peer_tone, assembled)
            self.assertIn("拒绝", req.system_prompt)
            self.assertIn("用户边界", req.system_prompt)
            self.assertEqual(req.contexts, [])
            reference_history = self.relationship_context(req)["recent_interactions_with_this_user"]
            self.assertIn("这个称呼我不喊", json.dumps(reference_history, ensure_ascii=False))
            self.assertNotIn(peer_tone, json.dumps(reference_history, ensure_ascii=False))

    async def test_intent_failure_keeps_relationship_result_and_normal_reply(self):
        async def relationship_only(**kwargs):
            self.context.calls.append(kwargs)
            if kwargs["system_prompt"] == sdk_fakes.plugin.prompts.PERSONAL_INTENT:
                raise TimeoutError("intent timeout")
            return types.SimpleNamespace(
                completion_text=json.dumps(
                    {"confidence": 0.9, "deltas": {"affection": 2}}, ensure_ascii=False
                )
            )

        self.context.llm_generate = relationship_only
        event = sdk_fakes.Event(message="谢谢你认真听我说")
        req = self.request()

        await self.bot.on_request(event, req)

        self.assertEqual(len(self.context.calls), 2)
        self.assertFalse(event.stopped)
        self.assertEqual(event.sent, [])
        self.assertEqual(self.current_settings(req)["current_request"], "")
        await self.finish(event)
        profile = self.bot.store.profile(self.bot._key(event))
        self.assertEqual(profile["affection"], 22)
        self.assertEqual(profile["user_persona"], "")

    async def test_relationship_failure_keeps_valid_personal_intent(self):
        message = "以后叫我哥哥，说话温柔一点"

        async def intent_only(**kwargs):
            self.context.calls.append(kwargs)
            if kwargs["system_prompt"] == sdk_fakes.plugin.prompts.ANALYZE:
                raise TimeoutError("relationship timeout")
            payload = json.loads(kwargs["prompt"])
            self.assertEqual(
                set(payload), {"current_user_id", "user_message", "existing_user_persona"}
            )
            self.assertEqual(payload["current_user_id"], event.user)
            self.assertEqual(payload["user_message"], message)
            return types.SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "persona_confidence": 0.96,
                        "user_persona": "称呼当前用户为哥哥，口吻温柔。",
                        "persona_evidence": ["以后叫我哥哥", "说话温柔一点"],
                        "reply_confidence": 0,
                        "reply_guidance": "",
                    },
                    ensure_ascii=False,
                )
            )

        self.context.llm_generate = intent_only
        event = sdk_fakes.Event(message=message)
        req = self.request()
        await self.bot.on_request(event, req)

        self.assertEqual(len(self.context.calls), 2)
        self.assertEqual(
            self.current_settings(req)["persistent_persona"], "称呼当前用户为哥哥，口吻温柔。"
        )
        await self.finish(event)
        profile = self.bot.store.profile(self.bot._key(event))
        self.assertEqual(profile["user_persona"], "称呼当前用户为哥哥，口吻温柔。")
        self.assertEqual(profile["affection"], 20)


if __name__ == "__main__":
    unittest.main()
