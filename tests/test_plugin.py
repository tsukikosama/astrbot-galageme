"""Adapter contract tests with a minimal AstrBot 4.29.1-compatible fake SDK.

These exercise the real plugin and SQLite together, not a live QQ installation.
"""

import asyncio
import enum
import importlib
import json
import logging
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __package__:
    from .bootstrap import ensure_package
else:
    from bootstrap import ensure_package

ensure_package()


def module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def last_call(context, system_prompt):
    return next(call for call in reversed(context.calls) if system_prompt in call["system_prompt"])


class CustomFilter:
    def __init__(self, *args, **kwargs):
        pass


class Filters:
    PermissionType = types.SimpleNamespace(ADMIN="admin")
    CustomFilter = CustomFilter

    def __getattr__(self, name):
        return lambda *args, **kwargs: lambda function: function


class Star:
    def __init__(self, context):
        self.context = context


class TextPart:
    def __init__(self, text):
        self.text = text
        self.temporary = False

    def mark_as_temp(self):
        self.temporary = True
        return self


class MessageChain:
    def __init__(self, chain):
        self.chain = chain


class At:
    def __init__(self, qq):
        self.qq = qq


class Plain:
    def __init__(self, text):
        self.text = text


module("astrbot")
module("astrbot.api", AstrBotConfig=dict, logger=logging.getLogger("test"))
module("astrbot.api.event", AstrMessageEvent=object, MessageChain=MessageChain, filter=Filters())
module("astrbot.api.star", Star=Star, Context=object)
module("astrbot.api.provider", LLMResponse=object, ProviderRequest=object)
module("astrbot.api.message_components", At=At, Plain=Plain)
module(
    "astrbot.api.web",
    request=types.SimpleNamespace(username=None),
    json_response=lambda data, **kwargs: {
        "payload": data,
        "status_code": kwargs.get("status_code", 200),
    },
    error_response=lambda message, **kwargs: {
        "payload": {"status": "error", "message": message},
        "status_code": kwargs.get("status_code", 400),
    },
)
module("astrbot.core")
module("astrbot.core.agent")
module("astrbot.core.agent.message", TextPart=TextPart)
module("astrbot.core.message")
ResultContentType = enum.Enum("ResultContentType", "LLM_RESULT STREAMING_RESULT GENERAL_RESULT")
module("astrbot.core.message.message_event_result", ResultContentType=ResultContentType)
module("astrbot.core.utils")
module("astrbot.core.utils.astrbot_path", get_astrbot_data_path=lambda: ".")
plugin = importlib.import_module("astrbot_plugin_relationship_memory.main")


class Event:
    def __init__(self, user="10001", message="谢谢你认真听我说", mid="1", group="20001"):
        self.message_str = message
        self.message_obj = types.SimpleNamespace(self_id="99999", message_id=mid)
        self.created_at = 100
        self.user, self.group = user, group
        self.unified_msg_origin = "platform:GroupMessage:" + group
        self.extras = {}
        self.result = None
        self.sent = []
        self.role = "member"
        self.stopped = False
        self.is_wake = True
        self._has_send_oper = False

    def get_group_id(self):
        return self.group

    def get_sender_id(self):
        return self.user

    def get_sender_name(self):
        return "测试用户" + self.user[-2:]

    def get_platform_id(self):
        return "platform"

    def get_platform_name(self):
        return "aiocqhttp"

    def get_extra(self, key):
        return self.extras.get(key)

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_result(self):
        return self.result

    def set_result(self, result):
        self.result = result

    def stop_event(self):
        self.stopped = True

    def plain_result(self, text):
        return text

    async def send(self, result):
        self.sent.append(result)


class Context:
    def __init__(self):
        self.registered_web_apis = []
        self.sent = []
        self.calls = []
        self.analysis = {
            "confidence": 0.9,
            "deltas": {"affection": 2},
            "reason": "认真交流",
            "style": "逐渐习惯认真交流",
            "memories": [{"content": "用户喜欢简洁回答", "kind": "preference", "public": True}],
        }
        self.reply = {"send": True, "text": "之前聊的那件事，有新进展了吗？"}
        self.error = None

    def register_web_api(self, route, handler, methods, description):
        self.registered_web_apis.append((route, handler, methods, description))

    async def get_current_chat_provider_id(self, umo):
        return "provider"

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        data = (
            self.analysis
            if kwargs["system_prompt"] in {plugin.prompts.ANALYZE, plugin.prompts.PERSONAL_INTENT}
            else self.reply
        )
        return types.SimpleNamespace(completion_text=json.dumps(data, ensure_ascii=False))

    async def send_message(self, umo, chain):
        self.sent.append((umo, chain))
        return True


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(plugin, "get_astrbot_data_path", return_value=self.tmp.name)
        self.worker_patch = patch.object(plugin.Main, "_start_worker")
        self.path_patch.start()
        self.worker_patch.start()
        self.context = Context()
        self.config = {"enabled_groups": ["20001"], "base_probability": 1, "initiative": 2}
        self.bot = plugin.Main(self.context, self.config)

    async def asyncTearDown(self):
        await self.bot.terminate()
        self.worker_patch.stop()
        self.path_patch.stop()
        self.tmp.cleanup()

    def request(self):
        return types.SimpleNamespace(
            system_prompt="你是嘴硬但善良的角色。",
            prompt="这是提供方原始提示。",
            contexts=[],
            extra_user_content_parts=[],
        )

    @staticmethod
    def relationship_context(req):
        part = next(
            item
            for item in req.extra_user_content_parts
            if getattr(item, "text", "").startswith("<relationship_context>")
        )
        raw = part.text.split("\n", 1)[1].rsplit("\n</relationship_context>", 1)[0]
        return json.loads(raw)

    async def finish(self, event, streaming=False):
        event._has_send_oper = True
        event.result = types.SimpleNamespace(
            is_llm_result=lambda: not streaming,
            result_content_type=types.SimpleNamespace(
                name="STREAMING_RESULT" if streaming else "LLM_RESULT"
            ),
        )
        response = types.SimpleNamespace(
            completion_text="我听着呢，你继续说。", role="assistant", tools_call_name=[]
        )
        if streaming:
            await self.bot.after_sent(event)
            await self.bot.on_done(event, None, response)
        else:
            await self.bot.on_done(event, None, response)
            await self.bot.after_sent(event)

    async def test_reply_keeps_persona_and_commits_only_after_delivery(self):
        event, req = Event(), self.request()
        await self.bot.on_request(event, req)
        self.assertTrue(req.system_prompt.startswith("你是嘴硬但善良的角色。"))
        self.assertTrue(req.extra_user_content_parts[0].temporary)
        self.assertEqual(self.bot.store.profile(self.bot._key(event))["affection"], 20)
        await self.finish(event)
        self.assertEqual(self.bot.store.profile(self.bot._key(event))["affection"], 22)
        await self.bot.after_sent(event)
        self.assertEqual(self.bot.store.profile(self.bot._key(event))["affection"], 22)

    async def test_request_capture_discards_later_shared_injection_and_restores_baseline(self):
        event = Event()
        req = self.request()
        original_attachment = TextPart("original attachment payload")
        req.extra_user_content_parts.append(original_attachment)
        await self.bot.capture_request(event, req)

        req.system_prompt += "\nSHARED SYSTEM MEMORY from another user"
        req.prompt += "\nSHARED PROMPT MEMORY from another user"
        req.contexts.append("SHARED CONTEXT MEMORY from another user")
        injected = TextPart("SHARED EXTRA MEMORY from another user")
        req.extra_user_content_parts.append(injected)

        await self.bot.on_request(event, req)

        self.assertNotIn("SHARED", req.system_prompt)
        self.assertNotIn("SHARED", req.prompt)
        self.assertNotIn("SHARED", json.dumps(req.contexts, ensure_ascii=False))
        self.assertNotIn(
            "SHARED", " ".join(getattr(part, "text", "") for part in req.extra_user_content_parts)
        )
        self.assertIn("原始提示", req.prompt)
        self.assertIn("嘴硬但善良", req.system_prompt)
        self.assertIn("<current_user_settings>", req.system_prompt)
        self.assertTrue(
            any(
                getattr(part, "text", "") == "original attachment payload"
                for part in req.extra_user_content_parts
            )
        )
        self.assertFalse(any(part is injected for part in req.extra_user_content_parts))

    async def test_users_have_separate_context_and_conflicting_personal_personas(self):
        responses = {
            "10001": {
                "confidence": 0.9,
                "persona_confidence": 0.95,
                "user_persona": "称呼我为阿岚，我住在海边。",
                "persona_evidence": "称呼我为阿岚，我住在海边",
            },
            "10002": {
                "confidence": 0.9,
                "persona_confidence": 0.95,
                "user_persona": "称呼我为小岚，我住在山里。",
                "persona_evidence": "称呼我为小岚，我住在山里",
            },
        }
        original = self.context.llm_generate

        async def per_user(**kwargs):
            payload = json.loads(kwargs["prompt"])
            if kwargs["system_prompt"] == plugin.prompts.ANALYZE:
                # Simulate old-history contamination; Main must ignore intent fields here.
                return types.SimpleNamespace(
                    completion_text=json.dumps(
                        {
                            "confidence": 0.9,
                            "reply_confidence": 0.99,
                            "reply_guidance": "沿用旧助手拒绝，继续不喊主人。",
                            "reply_evidence": "旧回复拒绝称呼",
                        },
                        ensure_ascii=False,
                    )
                )
            if kwargs["system_prompt"] == plugin.prompts.PERSONAL_INTENT:
                user_id = "10001" if "阿岚" in payload["user_message"] else "10002"
                return types.SimpleNamespace(
                    completion_text=json.dumps(responses[user_id], ensure_ascii=False)
                )
            return await original(**kwargs)

        self.context.llm_generate = per_user
        a = Event(user="10001", message="请记住：称呼我为阿岚，我住在海边。", mid="a")
        await self.bot.on_request(a, self.request())
        await self.finish(a)

        b = Event(user="10002", message="请记住：称呼我为小岚，我住在山里。", mid="b")
        b_req = self.request()
        await self.bot.on_request(b, b_req)
        await self.finish(b)

        a_key, b_key = self.bot._key(a), self.bot._key(b)
        self.assertEqual(
            self.bot.store.profile(a_key)["user_persona"], responses["10001"]["user_persona"]
        )
        self.assertEqual(
            self.bot.store.profile(b_key)["user_persona"], responses["10002"]["user_persona"]
        )
        context = self.relationship_context(b_req)
        self.assertEqual(context["user_id"], "10002")
        self.assertNotIn("海边", json.dumps(context, ensure_ascii=False))
        self.assertIn("小岚", json.dumps(context, ensure_ascii=False))

        self.bot.store.forget(a_key)
        self.assertEqual(self.bot.store.profile(a_key)["user_persona"], "")
        self.assertEqual(
            self.bot.store.profile(b_key)["user_persona"], responses["10002"]["user_persona"]
        )

    async def test_failed_delivery_does_not_save_user_persona(self):
        self.context.analysis.update(
            {
                "user_persona": "称呼我为阿岚。",
                "persona_evidence": "称呼我为阿岚",
            }
        )
        event = Event(message="请记住，称呼我为阿岚。")
        await self.bot.on_request(event, self.request())
        await self.bot.on_done(
            event,
            None,
            types.SimpleNamespace(completion_text="没有发出的回复", role="assistant"),
        )
        event.result = types.SimpleNamespace(
            is_llm_result=lambda: True,
            result_content_type=ResultContentType.LLM_RESULT,
        )
        await self.bot.after_sent(event)
        self.assertEqual(self.bot.store.profile(self.bot._key(event))["user_persona"], "")

    async def test_two_users_can_save_same_title_and_change_only_their_own_persona(self):
        for uid, tone in [("10001", "温柔"), ("10002", "傲娇")]:
            message = f"以后叫我主人，说话{tone}一点"
            self.context.analysis = {
                "confidence": 0.95,
                "user_persona": f"称呼当前用户为主人，口吻{tone}",
                "persona_evidence": message,
            }
            event = Event(user=uid, message=message, mid="setting-" + uid)
            await self.bot.on_request(event, self.request())
            await self.finish(event)
        keys = [self.bot._key(Event(user=uid)) for uid in ("10001", "10002")]
        self.context.analysis = {"confidence": 0.95}
        for uid, tone, other in [("10001", "温柔", "傲娇"), ("10002", "傲娇", "温柔")]:
            event = Event(user=uid, message="你好", mid="hello-" + uid)
            req = self.request()
            # Even a host session containing a peer's setting must not be replayed.
            req.contexts = [{"role": "user", "content": "PEER_EXCLUSIVE_MASTER"}]
            await self.bot.on_request(event, req)
            personal = self.relationship_context(req)["user_persona"]
            self.assertIn("主人", personal)
            self.assertIn(tone, personal)
            self.assertNotIn(other, personal)
            self.assertEqual(req.contexts, [])
            recent = self.relationship_context(req)["recent_interactions_with_this_user"]
            history = json.dumps(recent, ensure_ascii=False)
            self.assertIn(tone, history)
            self.assertNotIn(other, history)
            self.assertNotIn(
                "PEER_EXCLUSIVE_MASTER",
                req.system_prompt
                + req.prompt
                + json.dumps(req.contexts, ensure_ascii=False)
                + " ".join(getattr(part, "text", "") for part in req.extra_user_content_parts),
            )
        self.bot.store.reset_relationship(keys[0])
        self.assertEqual(self.bot.store.profile(keys[0])["user_persona"], "")
        self.assertIn("傲娇", self.bot.store.profile(keys[1])["user_persona"])

    async def test_core_group_context_is_removed_but_current_quote_and_media_remain(self):
        event, req = Event(), self.request()
        req.extra_user_content_parts = [
            TextPart(
                "<system_reminder>You are in a group chat. Belows are group chat context after your last reply:\n--- BEGIN CONTEXT---\n[peer]: PEER_PERSONA_ONLY\n--- END CONTEXT ---\n</system_reminder>"
            ),
            TextPart("<Quoted Message>这个菜怎么做？</Quoted Message>"),
            {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
        ]
        await self.bot.capture_request(event, req)
        await self.bot.on_request(event, req)
        payload = " ".join(getattr(p, "text", "") for p in req.extra_user_content_parts)
        self.assertNotIn("PEER_PERSONA_ONLY", payload)
        self.assertIn("这个菜怎么做", payload)
        self.assertIn(
            {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
            req.extra_user_content_parts,
        )

    async def test_forget_during_analysis_also_removes_history_from_reply_request(self):
        event = Event(message="新的聊天", mid="new")
        key = self.bot._key(event)
        self.bot.store.ensure_profile(key, event.group, event.user, event.unified_msg_origin)
        self.bot.store.commit_turn(
            key, "old", "PERSONAL_OLD_SETTING", "old reply", {"confidence": 0.9}
        )

        async def forget_while_analyzing(**kwargs):
            self.bot.store.forget(key)
            return types.SimpleNamespace(completion_text='{"confidence":0.9}')

        self.context.llm_generate = forget_while_analyzing
        req = self.request()
        await self.bot.on_request(event, req)
        self.assertEqual(req.contexts, [])
        self.assertNotIn(
            "PERSONAL_OLD_SETTING", json.dumps(self.relationship_context(req), ensure_ascii=False)
        )

    async def test_streaming_delivery_order(self):
        event = Event()
        await self.bot.on_request(event, self.request())
        await self.finish(event, streaming=True)
        self.assertEqual(self.bot.store.profile(self.bot._key(event))["affection"], 22)

    async def test_group_disabled_does_not_invoke_model(self):
        event = Event(group="other")
        await self.bot.on_request(event, self.request())
        self.assertEqual(self.context.calls, [])
        self.assertEqual(self.bot.store.profiles(), [])

    async def test_model_failure_still_allows_normal_reply(self):
        event = Event()
        self.context.error = TimeoutError()
        await self.bot.on_request(event, self.request())
        await self.finish(event)
        self.assertEqual(self.bot.store.profile(self.bot._key(event))["affection"], 20)

    async def test_incomplete_turn_does_not_update_relationship(self):
        event = Event()
        await self.bot.on_request(event, self.request())
        await self.bot.on_done(
            event, None, types.SimpleNamespace(completion_text="生成但没有发送", role="assistant")
        )
        self.assertEqual(self.bot.store.recent(self.bot._key(event)), [])

    async def test_failed_delivery_hook_does_not_commit(self):
        event = Event()
        await self.bot.on_request(event, self.request())
        await self.bot.on_done(
            event, None, types.SimpleNamespace(completion_text="未发送", role="assistant")
        )
        event.result = types.SimpleNamespace(
            is_llm_result=lambda: True, result_content_type=ResultContentType.LLM_RESULT
        )
        await self.bot.after_sent(event)
        self.assertEqual(self.bot.store.recent(self.bot._key(event)), [])

    async def test_preparation_disables_streaming_before_agent_stage(self):
        event = Event()
        listener_filter = plugin.AlreadyAwake()
        self.assertTrue(listener_filter.filter(event, {}))
        await self.bot.prepare_dialogue(event)
        self.assertIs(event.get_extra("enable_streaming"), False)
        event.is_wake = False
        self.assertFalse(listener_filter.filter(event, {}))
        other = Event(group="other")
        await self.bot.prepare_dialogue(other)
        self.assertIsNone(other.get_extra("enable_streaming"))

    async def test_forget_invalidates_inflight_turn(self):
        event = Event(message="这是要删除的信息")
        await self.bot.on_request(event, self.request())
        control = Event(message="清空我的记忆", mid="control")
        await self.bot.on_request(control, self.request())
        self.assertTrue(control.stopped)
        await self.finish(event)
        self.assertEqual(self.bot.store.recent(self.bot._key(event)), [])
        self.assertEqual(self.bot.store.memories(self.bot._key(event)), [])

    async def test_optout_intercepts_without_model_and_blocks_late_commit(self):
        event = Event()
        await self.bot.on_request(event, self.request())
        calls = len(self.context.calls)
        control = Event(message="以后别记我的聊天", mid="off")
        await self.bot.on_request(control, self.request())
        self.assertEqual(len(self.context.calls), calls)
        await self.finish(event)
        profile = self.bot.store.profile(self.bot._key(event))
        self.assertFalse(profile["memory_enabled"])
        self.assertFalse(profile["proactive_enabled"])
        self.assertEqual(self.bot.store.recent(self.bot._key(event)), [])

    async def test_affection_query_answers_from_store_without_model_or_score_change(self):
        first = Event()
        await self.bot.on_request(first, self.request())
        await self.finish(first)
        second = Event(user="10002", message="很高兴认识你")
        await self.bot.on_request(second, self.request())
        await self.finish(second)
        # Give the two users different scores so a crossed reply is detectable.
        self.bot.store.commit_turn(
            self.bot._key(first),
            "extra",
            "再聊一句",
            "好呀",
            plugin.validate_analysis({"confidence": 0.9, "deltas": {"affection": 2}}, 0),
        )
        normal = Event(mid="normal")
        normal_request = self.request()
        await self.bot.on_request(normal, normal_request)
        await self.finish(normal)
        self.assertIn("发送“查看好感度”", normal_request.system_prompt)
        calls = len(self.context.calls)
        replies = {}
        for event in (first, second):
            query = Event(user=event.user, message="查看好感度", mid="q-" + event.user)
            request = self.request()
            await self.bot.on_request(query, request)
            self.assertTrue(query.stopped)
            self.assertEqual(len(query.sent), 1)
            self.assertIsNone(query.get_extra("relationship_pending"))
            replies[event.user] = query.sent[0]
        # No model call, no relationship change, and each user only sees their own number.
        self.assertEqual(len(self.context.calls), calls)
        own = {u: self.bot.store.profile(self.bot._key(Event(user=u)))["affection"] for u in replies}
        self.assertNotEqual(own["10001"], own["10002"])
        for user, reply in replies.items():
            self.assertIn(f"{own[user]} / 100", reply)
        self.assertNotIn(f"{own['10001']} / 100", replies["10002"])
        self.assertEqual(self.bot.store.profile(self.bot._key(first))["affection"], own["10001"])
        # A query is not an interaction: it must not be stored or settled as a turn.
        self.assertNotIn(
            "查看好感度", [t["user_text"] for t in self.bot.store.recent(self.bot._key(first))]
        )

    async def test_affection_query_disabled_keeps_normal_model_reply(self):
        self.config["affection_query_enabled"] = False
        event = Event(message="查看好感度")
        request = self.request()
        await self.bot.on_request(event, request)
        self.assertFalse(event.stopped)
        self.assertEqual(event.sent, [])
        self.assertIsNotNone(event.get_extra("relationship_pending"))
        self.assertNotIn("由插件给出当前真实数值", request.system_prompt)

    async def test_daily_limits_follow_config_and_zero_is_preserved(self):
        self.assertEqual(self.bot._daily_limits()["affection_gain"], 8)
        self.config["daily_affection_gain"] = 2
        self.assertEqual(self.bot._daily_limits()["affection_gain"], 2)
        self.config["daily_affection_gain"] = 0
        limits = self.bot._daily_limits()
        self.assertEqual(limits["affection_gain"], 0)  # 0 is a real value: no limit
        self.assertEqual(limits["affection_loss"], 10)
        self.config["daily_affection_gain"] = "oops"
        self.assertEqual(self.bot._daily_limits()["affection_gain"], 8)

    async def test_analysis_does_not_use_other_users_history(self):
        first = Event(user="10001")
        await self.bot.on_request(first, self.request())
        await self.finish(first)
        second = Event(user="10002", message="你好")
        await self.bot.on_request(second, self.request())
        payload = json.loads(last_call(self.context, plugin.prompts.ANALYZE)["prompt"])
        self.assertEqual(payload["recent"], [])
        self.assertEqual(payload["existing_memories"], [])

    async def test_model_tags_persist_and_only_public_tags_are_injected(self):
        self.context.analysis["tags"] = [
            {
                "name": "喜欢科幻",
                "category": "interest",
                "confidence": 0.9,
                "evidence": "用户明确表达",
                "public": True,
            },
            {
                "name": "仅管理可见",
                "category": "preference",
                "confidence": 0.9,
                "evidence": "用户明确表达",
                "public": False,
            },
        ]
        event = Event()
        await self.bot.on_request(event, self.request())
        await self.finish(event)
        self.assertEqual(len(self.bot.store.tags(self.bot._key(event))), 2)
        next_request = self.request()
        await self.bot.on_request(Event(mid="2"), next_request)
        context = json.loads(
            next_request.extra_user_content_parts[0].text.split("\n", 1)[1].rsplit("\n", 1)[0]
        )
        self.assertEqual([tag["name"] for tag in context["user_tags"]], ["喜欢科幻"])

    async def test_admin_has_runtime_permission_check(self):
        event = Event()
        result = [part async for part in self.bot.admin(event, "重置关系", event.user)]
        self.assertIn("仅 AstrBot 管理员", result[0])

    async def test_proactive_mentions_once_and_uses_public_memories(self):
        event = Event()
        now = datetime(2026, 10, 27, 12, tzinfo=plugin.TZ).timestamp()
        key = self.bot._key(event)
        self.bot.store.ensure_profile(
            key, event.group, event.user, event.unified_msg_origin, "provider", "嘴硬但善良"
        )
        self.bot.store.commit_turn(
            key,
            "old",
            "明天面试",
            "好",
            {
                "confidence": 0.9,
                "followups": [
                    {"content": "面试结果", "due_at": "2026-10-27T10:00:00+08:00", "public": True}
                ],
                "memories": [{"content": "秘密地址", "kind": "fact", "public": False}],
            },
            now=now - 7200,
        )
        with (
            patch.object(self.bot, "_can_contact", return_value=True),
            patch.object(plugin.time, "time", return_value=now),
            patch.object(plugin.random, "random", return_value=0),
        ):
            await self.bot._proactive(now)
            await self.bot._proactive(now + 60)
        self.assertEqual(len(self.context.sent), 1)
        self.assertEqual(self.context.sent[0][1].chain[0].qq, event.user)
        payload = json.loads(last_call(self.context, plugin.prompts.PROACTIVE)["prompt"])
        self.assertNotIn("秘密地址", payload["public_memories"])

    async def test_quiet_hours_skip_proactive(self):
        await self.bot._proactive(datetime(2026, 10, 27, 1, tzinfo=plugin.TZ).timestamp())
        self.assertEqual(self.context.calls, [])

    async def test_daily_job_archives_previous_day_and_is_idempotent(self):
        event = Event()
        now = datetime(2026, 10, 27, 4, tzinfo=plugin.TZ).timestamp()
        key = self.bot._key(event)
        self.bot.store.ensure_profile(
            key, event.group, event.user, event.unified_msg_origin, persona="原本人设"
        )
        self.bot.store.commit_turn(key, "yesterday", "喜欢绿茶", "记住了", {}, now=now - 86400)
        self.context.reply = {
            "summary": "昨天讨论了饮品偏好",
            "style": "熟悉对方的偏好",
            "memories": [{"content": "喜欢绿茶", "kind": "preference", "public": True}],
        }
        await self.bot._summaries(now)
        self.assertEqual(self.bot.store.pending_summaries(now), [])
        self.assertEqual(self.bot.store.memories(key)[0]["content"], "喜欢绿茶")
        self.assertEqual(
            self.bot.store.db.execute("SELECT COUNT(*) FROM daily_history").fetchone()[0], 1
        )
        calls = len(self.context.calls)
        await self.bot._summaries(now)
        self.assertEqual(len(self.context.calls), calls)

    async def test_malformed_summary_keeps_work_pending(self):
        event = Event()
        now = datetime(2026, 10, 27, 4, tzinfo=plugin.TZ).timestamp()
        key = self.bot._key(event)
        self.bot.store.ensure_profile(key, event.group, event.user, event.unified_msg_origin)
        self.bot.store.commit_turn(key, "yesterday", "信息", "回复", {}, now=now - 86400)
        self.context.reply = {}
        await self.bot._summaries(now)
        self.assertEqual(len(self.bot.store.pending_summaries(now)), 1)

    async def test_deletion_during_analysis_does_not_restore_raw_turn(self):
        event = Event(message="需要遗忘的信息")
        started, resume = asyncio.Event(), asyncio.Event()
        original = self.context.llm_generate

        async def delayed(**kwargs):
            started.set()
            await resume.wait()
            return await original(**kwargs)

        self.context.llm_generate = delayed
        running = asyncio.create_task(self.bot.on_request(event, self.request()))
        await started.wait()
        await self.bot.on_request(Event(message="清空我的记忆", mid="forget"), self.request())
        resume.set()
        await running
        await self.finish(event)
        self.assertEqual(self.bot.store.recent(self.bot._key(event)), [])
        self.assertEqual(self.bot.store.memories(self.bot._key(event)), [])

    async def test_contact_checks_membership_and_bot_mute(self):
        event = Event()
        key = self.bot._key(event)
        self.bot.store.ensure_profile(key, event.group, event.user, event.unified_msg_origin)
        muted = False

        async def action(name, **kwargs):
            return {
                "user_id": kwargs["user_id"],
                "shut_up_timestamp": 10**12
                if muted and str(kwargs["user_id"]) == event.message_obj.self_id
                else 0,
            }

        platform = types.SimpleNamespace(
            meta=lambda: types.SimpleNamespace(name="aiocqhttp"),
            get_client=lambda: types.SimpleNamespace(call_action=action),
        )
        self.context.get_platform_inst = lambda platform_id: platform
        self.assertTrue(await self.bot._can_contact(self.bot.store.profile(key)))
        muted = True
        self.assertFalse(await self.bot._can_contact(self.bot.store.profile(key)))

    async def test_old_role_background_tasks_are_inactive(self):
        event = Event()
        key = self.bot._key(event)
        profile = self.bot.store.ensure_profile(
            key, event.group, event.user, event.unified_msg_origin
        )
        self.config["role_id"] = "another-character"
        self.assertFalse(self.bot._active_profile(profile))

    async def test_worker_is_cancelled_on_terminate(self):
        self.bot._task = asyncio.create_task(asyncio.sleep(1000))
        task = self.bot._task
        # tearDown will call terminate as well; don't double-close the SQLite connection.
        await self.bot.terminate()
        self.assertTrue(task.cancelled())
        self.bot.store.close = lambda: None
