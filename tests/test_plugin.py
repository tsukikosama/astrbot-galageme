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
        data = self.analysis if kwargs["system_prompt"] == plugin.prompts.ANALYZE else self.reply
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
            system_prompt="你是嘴硬但善良的角色。", extra_user_content_parts=[]
        )

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

    async def test_analysis_does_not_use_other_users_history(self):
        first = Event(user="10001")
        await self.bot.on_request(first, self.request())
        await self.finish(first)
        second = Event(user="10002", message="你好")
        await self.bot.on_request(second, self.request())
        payload = json.loads(self.context.calls[-1]["prompt"])
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
        payload = json.loads(self.context.calls[-1]["prompt"])
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
