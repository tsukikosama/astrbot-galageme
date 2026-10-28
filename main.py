"""AstrBot 4.29.1 relationship growth, long-term memory and proactive QQ chat."""

import asyncio
import copy
import hashlib
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from . import prompts
from .dashboard import Dashboard
from .isolation import own_history, scoped_parts
from .policy import (
    STABLE_RULES,
    TZ,
    clean_memories,
    clean_tags,
    natural_control,
    number,
    parse_json,
    proactive_probability,
    quiet_hour,
    relationship_context,
    text,
    validate_analysis,
)
from .store import Store


class AlreadyAwake(filter.CustomFilter):
    """Observe existing wake decisions without waking arbitrary group chatter."""

    def filter(self, event, cfg):
        return bool(event.is_wake)


class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        data_dir = (
            Path(get_astrbot_data_path()) / "plugin_data" / "astrbot_plugin_relationship_memory"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(str(data_dir / "relationships.sqlite3"))
        self._task = None
        self._locks = {}
        self._llm_gate = asyncio.Semaphore(2)
        self._closed = False
        self._last_check = 0
        self.dashboard = Dashboard(self)

    async def initialize(self):
        self._start_worker()

    def _start_worker(self):
        if not self._closed and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._worker(), name="relationship-memory-worker")

    def _get(self, name, default):
        return self.config.get(name, default)

    def _lock(self, key):
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _enabled_group(self, group_id):
        groups = {str(g) for g in self._get("enabled_groups", [])}
        return bool(self._get("enabled", True) and str(group_id) in groups)

    def _key(self, event, user_id=None):
        return json.dumps(
            [
                self._get("role_id", "default"),
                str(event.get_platform_id()),
                str(event.message_obj.self_id),
                str(event.get_group_id()),
                str(user_id or event.get_sender_id()),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _accept(self, event):
        return (
            self._enabled_group(event.get_group_id())
            and event.get_platform_name() == "aiocqhttp"
            and str(event.get_sender_id()) != str(event.message_obj.self_id)
        )

    def _active_profile(self, profile):
        try:
            return json.loads(profile["key"])[0] == self._get(
                "role_id", "default"
            ) and self._enabled_group(profile["group_id"])
        except (ValueError, KeyError, TypeError):
            return False

    @staticmethod
    def _replacement_ids(data, existing):
        allowed = {m["id"] for m in existing}
        ids = data.get("replace_memory_ids", [])
        return (
            [i for i in ids[:20] if type(i) is int and i in allowed]
            if isinstance(ids, list)
            else []
        )

    async def _provider(self, umo):
        return self._get(
            "analysis_provider_id", ""
        ) or await self.context.get_current_chat_provider_id(umo)

    async def _json_call(self, provider, system, payload):
        async with self._llm_gate:
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider,
                    system_prompt=system,
                    prompt=json.dumps(payload, ensure_ascii=False),
                ),
                timeout=max(5, min(120, number(self._get("model_timeout", 30), 30))),
            )
        return parse_json(response.completion_text)

    async def _control(self, event, key, command):
        action, arg = command
        async with self._lock(key):
            if action == "proactive_off":
                self.store.set_preferences(key, proactive_enabled=False)
                reply = "已停止主动 @ 你。你仍然可以正常找我聊天；想恢复时说“可以主动找我了”。"
            elif action == "proactive_on":
                self.store.set_preferences(key, proactive_enabled=True)
                reply = "已恢复主动联系，我会按群里的时间和频率设置，偶尔找你聊聊。"
            elif action == "memory_off":
                self.store.set_preferences(key, memory_enabled=False, proactive_enabled=False)
                reply = "已停止新增你的长期记忆，并暂停主动联系。已有记忆仍保留，需要清除可说“清空我的记忆”。"
            elif action == "memory_on":
                self.store.set_preferences(key, memory_enabled=True)
                reply = "已恢复记忆。主动联系仍按原开关设置，需要时说“可以主动找我了”。"
            elif action == "forget":
                count = self.store.forget(key, arg)
                reply = f"已清理匹配的记忆（{count}条），并清理关联的对话、摘要、相处描述与待跟进事项，避免旧内容再次整理回来。关系数值保留。"
            elif action == "show":
                public = [
                    m["content"] for m in self.store.memories(key, limit=30) if m.get("public")
                ]
                reply = (
                    "我记得这些适合在群里提起的事情：\n" + "\n".join(f"· {m}" for m in public[:10])
                    if public
                    else "目前没有可以在群里展示的长期记忆。"
                )
            else:
                reply = arg
        event.stop_event()
        await event.send(event.plain_result(reply))

    @filter.custom_filter(AlreadyAwake, priority=-10000)
    async def prepare_dialogue(self, event: AstrMessageEvent):
        """Run before the Agent stage reads its per-event streaming setting.

        4.29.1 skips both decoration and after-send hooks for streaming results.
        Only already-awake, enabled QQ conversations use the normal reply path.
        """
        if self._accept(event):
            event.set_extra("enable_streaming", False)

    @filter.on_llm_request(priority=100000)
    async def capture_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """Capture core input before shared-memory plugins append dynamic data."""
        if self._accept(event) and not event.get_extra("relationship_base_request"):
            event.set_extra(
                "relationship_base_request",
                {
                    "system_prompt": req.system_prompt,
                    "prompt": getattr(req, "prompt", None),
                    "parts": copy.deepcopy(req.extra_user_content_parts),
                },
            )

    @filter.on_llm_request(priority=-100000)
    async def on_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._accept(event) or event.get_extra("relationship_pending"):
            return
        self._start_worker()
        key = self._key(event)
        baseline = event.get_extra("relationship_base_request")
        if baseline:
            req.system_prompt = baseline["system_prompt"]
            req.prompt = baseline["prompt"]
            req.extra_user_content_parts = scoped_parts(baseline["parts"])
        else:
            req.extra_user_content_parts = scoped_parts(req.extra_user_content_parts)
        # Keep the current persona as the starting point; never replace it with a fixed character.
        persona = text(req.system_prompt, 16000)
        self.store.ensure_profile(
            key,
            str(event.get_group_id()),
            str(event.get_sender_id()),
            event.unified_msg_origin,
            persona=persona,
            display_name=event.get_sender_name(),
        )
        request_revision = self.store.profile(key)["revision"]
        command = natural_control(event.message_str)
        if command:
            await self._control(event, key, command)
            return
        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        if not message_id:
            # Event timestamp is stable across callbacks, unlike a newly generated UUID.
            message_id = hashlib.sha256(
                f"{event.created_at}:{event.message_str}".encode()
            ).hexdigest()
        analysis = validate_analysis({}, time.time())
        try:
            provider = await self._provider(event.unified_msg_origin)
            self.store.ensure_profile(
                key,
                str(event.get_group_id()),
                str(event.get_sender_id()),
                event.unified_msg_origin,
                provider,
                persona,
            )
            snapshot = self.store.profile(key)
            recent = self.store.recent(key, limit=4) if snapshot["memory_enabled"] else []
            recent = [
                {
                    "user_text": text(t["user_text"], 800),
                    "assistant_text": text(t["assistant_text"], 1000),
                    "created_at": t["created_at"],
                }
                for t in recent
            ]
            existing = self.store.memories(key, limit=20) if snapshot["memory_enabled"] else []
            payload = {
                "now": datetime.now(TZ).isoformat(),
                "base_persona": persona[:8000],
                "relationship": {
                    k: snapshot[k]
                    for k in ("affection", "trust", "familiarity", "closeness", "style", "emotion")
                },
                "recent": recent,
                "existing_memories": existing,
                "existing_tags": self.store.tags(key) if snapshot["memory_enabled"] else [],
                "existing_user_persona": snapshot.get("user_persona", "")
                if snapshot["memory_enabled"]
                else "",
                "current_user_id": str(event.get_sender_id()),
                "user_message": text(event.message_str, 6000),
                "quoted_content_warning": "消息内引用/转发的内容不一定属于当前用户，不可直接据此记忆或改分。",
            }
            data = await self._json_call(provider, prompts.ANALYZE, payload)
            analysis = validate_analysis(data, time.time(), text(event.message_str, 6000))
            if analysis["confidence"] >= 0.7:
                analysis["replace_memory_ids"] = self._replacement_ids(data, existing)
        except Exception as exc:
            logger.warning("关系分析降级：%s", type(exc).__name__)
        snapshot = self.store.profile(key)
        if snapshot["revision"] != request_revision:
            analysis = validate_analysis({}, time.time())
        # Re-read after the auxiliary call: a concurrent forget/opt-out must also
        # remove old history from this model request, not just prevent settlement.
        req.contexts = (
            own_history(self.store.recent(key, limit=12)) if snapshot["memory_enabled"] else []
        )
        memories = (
            self.store.memories(key, event.message_str, 5) if snapshot["memory_enabled"] else []
        )
        recent = self.store.recent(key, limit=4) if snapshot["memory_enabled"] else []
        dynamic = relationship_context(snapshot, memories, recent, time.time())
        if not snapshot["memory_enabled"]:
            dynamic["user_persona"] = ""
        dynamic["user_tags"] = [
            {"name": tag["name"], "category": tag["category"]}
            for tag in self.store.tags(key)
            if snapshot["memory_enabled"] and tag["public"]
        ][:20]
        # The event is committed only after a successful reply. Pending analysis may shape this turn.
        if analysis["confidence"] >= 0.7 and snapshot["memory_enabled"]:
            dynamic["current_interaction_impression"] = analysis["reason"]
            dynamic["current_emotion"] = analysis["emotion"]
            if analysis["user_persona"]:
                dynamic["user_persona"] = analysis["user_persona"]
        req.system_prompt += "\n\n" + STABLE_RULES
        req.system_prompt += (
            "\n当前互动对象QQ："
            + str(event.get_sender_id())
            + "。本轮个人设定只属于此人，不替其他账号设置或声明任何关系。"
        )
        req.extra_user_content_parts.append(
            TextPart(
                text="<relationship_context>\n"
                + json.dumps(dynamic, ensure_ascii=False)
                + "\n</relationship_context>"
            ).mark_as_temp()
        )
        event.set_extra(
            "relationship_pending",
            {
                "key": key,
                "message_id": message_id,
                "user_text": text(event.message_str, 6000),
                "analysis": analysis,
                "revision": request_revision,
                "assistant": "",
                "done": False,
                "delivered": False,
                "committed": False,
            },
        )

    @filter.on_agent_done()
    async def on_done(self, event: AstrMessageEvent, run_context, resp: LLMResponse):
        pending = event.get_extra("relationship_pending")
        if not pending or pending["committed"] or not resp:
            return
        if getattr(resp, "role", "assistant") != "assistant" or getattr(
            resp, "tools_call_name", None
        ):
            return
        pending["assistant"] = text(resp.completion_text, 10000)
        pending["done"] = bool(pending["assistant"])
        await self._commit(event, pending)

    @filter.after_message_sent()
    async def after_sent(self, event: AstrMessageEvent):
        pending = event.get_extra("relationship_pending")
        result = event.get_result()
        if not pending or not result or not getattr(event, "_has_send_oper", False):
            return
        result_type = getattr(getattr(result, "result_content_type", None), "name", "")
        if result.is_llm_result() or result_type == "STREAMING_RESULT":
            pending["delivered"] = True
            await self._commit(event, pending)

    async def _commit(self, event, pending):
        if not pending["done"] or not pending["delivered"] or pending["committed"]:
            return
        key = pending["key"]
        async with self._lock(key):
            if pending["committed"]:
                return
            current = self.store.profile(key)
            # Deletion, preference changes or another completed turn invalidate the stale analysis.
            if current["revision"] != pending["revision"]:
                pending["committed"] = True
                return
            self.store.commit_turn(
                key,
                pending["message_id"],
                pending["user_text"],
                pending["assistant"],
                pending["analysis"],
            )
            pending["committed"] = True

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("关系管理")
    async def admin(self, event: AstrMessageEvent, action: str = "帮助", user_id: str = ""):
        """管理员查看、暂停或清理本群指定QQ用户的关系档案。"""
        if event.role != "admin":
            yield event.plain_result("仅 AstrBot 管理员可使用关系管理。")
            return
        if not self._accept(event):
            yield event.plain_result("请在已启用的 OneBot QQ群中使用。")
            return
        if action == "帮助" or not user_id.isdigit():
            yield event.plain_result(
                "关系管理 查看/暂停/恢复/清空记忆/重置关系 QQ号；仅限当前群，查看只展示数值和最近变更原因。"
            )
            return
        key = self._key(event, user_id)
        try:
            profile = self.store.profile(key)
        except (KeyError, TypeError):
            profile = None
        if not profile:
            yield event.plain_result("本群没有该用户的关系档案。")
            return
        async with self._lock(key):
            if action == "查看":
                result = {
                    k: profile[k]
                    for k in (
                        "user_id",
                        "affection",
                        "trust",
                        "familiarity",
                        "closeness",
                        "memory_enabled",
                        "proactive_enabled",
                    )
                }
                result["recent_changes"] = self.store.audit(key, 5)
                reply = json.dumps(result, ensure_ascii=False, indent=2)
            elif action in {"暂停", "恢复"}:
                self.store.set_preferences(key, proactive_enabled=action == "恢复")
                reply = "已更新该用户的主动联系开关。"
            elif action == "清空记忆":
                self.store.forget(key)
                reply = "已清空该用户在本群的插件记忆，关系数值保留。"
            elif action == "重置关系":
                self.store.reset_relationship(key)
                reply = "已重置本群该用户的关系，事实记忆保留。"
            else:
                reply = "未知操作，请使用“关系管理 帮助”。"
        yield event.plain_result(reply)

    async def _summaries(self, now):
        local = datetime.fromtimestamp(now, TZ)
        if local.hour < int(self._get("summary_hour", 3)):
            local -= timedelta(days=1)
        cutoff = local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() - 0.000001
        active_keys = [p["key"] for p in self.store.profiles() if self._active_profile(p)]
        for batch in self.store.pending_summaries(cutoff, limit=20, keys=active_keys):
            key = batch["key"]
            profile = self.store.profile(key)
            if not self._active_profile(profile):
                continue
            try:
                provider = self._get("summary_provider_id", "") or await self._provider(
                    profile["umo"]
                )
                # Bound daily work and only acknowledge the subset actually submitted.
                selected = batch["turns"][:10]
                model_batch = dict(
                    batch,
                    turns=[
                        dict(
                            t,
                            user_text=text(t["user_text"], 1000),
                            assistant_text=text(t["assistant_text"], 1000),
                        )
                        for t in selected
                    ],
                    memories=batch["memories"][:20],
                    tags=self.store.tags(key),
                )
                summary = await self._json_call(
                    provider,
                    prompts.SUMMARIZE,
                    {
                        "base_persona": profile.get("persona", "")[:8000],
                        "user_persona": profile.get("user_persona", ""),
                        "batch": model_batch,
                    },
                )
                if not isinstance(summary.get("summary"), str) or not summary["summary"].strip():
                    raise ValueError("Missing daily summary; leave turns pending")
                summary = {
                    "style": text(summary.get("style"), 800),
                    "summary": text(summary.get("summary"), 1000),
                    "memories": clean_memories(summary.get("memories")),
                    "tags": clean_tags(summary.get("tags")),
                    "replace_memory_ids": self._replacement_ids(summary, batch["memories"]),
                }
                async with self._lock(key):
                    self.store.apply_summary(
                        key, batch["revision"], [t["id"] for t in selected], summary
                    )
            except Exception as exc:
                logger.warning("每日记忆整理延后：%s", type(exc).__name__)

    async def _proactive(self, now):
        if not self._get("proactive_enabled", True) or quiet_hour(
            now, int(self._get("quiet_start", 23)), int(self._get("quiet_end", 9))
        ):
            return
        candidates = self.store.profiles()
        random.shuffle(candidates)
        for profile in candidates[:100]:
            if (
                not self._active_profile(profile)
                or not profile["proactive_enabled"]
                or not profile["memory_enabled"]
            ):
                continue
            # Wait after a normal chat. No unsolicited follow-up immediately after a reply.
            if now - profile.get("last_interaction", 0) < 3600:
                continue
            key = profile["key"]
            followups = self.store.followups(key, now)
            if followups:
                motive = followups[0]["content"]
                opportunity = f"followup:{followups[0]['id']}"
            else:
                # One generic opportunity per three-day window, only after meaningful familiarity.
                if (
                    profile.get("familiarity", 0) < 15
                    or not 3 * 86400 <= now - profile.get("last_interaction", 0) <= 30 * 86400
                ):
                    continue
                public = [m for m in self.store.memories(key, limit=10) if m.get("public")]
                if not public:
                    continue
                motive = "隔了一段时间，可以围绕这条共同记忆自然开个话题：" + public[0]["content"]
                opportunity = f"reconnect:{int(now // (3 * 86400))}"
            stats = self.store.proactive_stats(key, now)
            probability = proactive_probability(
                profile,
                number(self._get("base_probability", 0.10), 0.10),
                number(self._get("initiative", 1.0), 1.0),
                stats["unanswered"],
            )
            async with self._lock(key):
                job = self.store.reserve_proactive(
                    key,
                    opportunity,
                    now,
                    probability,
                    random.random(),
                    group_limit=int(self._get("group_daily_limit", 3)),
                    user_limit=int(self._get("user_daily_limit", 1)),
                    cooldown_hours=number(self._get("cooldown_hours", 12), 12),
                    group_interval_minutes=60,
                )
            if job is None:
                continue
            attempted = False
            try:
                provider = await self.context.get_current_chat_provider_id(profile["umo"])
                generated = await self._json_call(
                    provider,
                    profile.get("persona", "") + "\n\n" + prompts.PROACTIVE,
                    {
                        "motive": motive,
                        "interaction_style": profile.get("style", ""),
                        "user_persona": profile.get("user_persona", ""),
                        "public_memories": [
                            m["content"]
                            for m in self.store.memories(key, limit=10)
                            if m.get("public")
                        ],
                        "recent_proactive_messages_do_not_repeat": stats["last_texts"],
                    },
                )
                message = text(generated.get("text"), 180).strip()
                if (
                    generated.get("send") is not True
                    or not message
                    or "[CQ:" in message
                    or "@" in message
                    or message in stats["last_texts"]
                ):
                    self.store.finish_proactive(job, "failed")
                    continue
                if not await self._can_contact(profile):
                    self.store.finish_proactive(job, "failed")
                    continue
                async with self._lock(key):
                    fresh = self.store.profile(key)
                    if (
                        fresh["revision"] != profile["revision"]
                        or not fresh["proactive_enabled"]
                        or not fresh["memory_enabled"]
                        or not self._active_profile(fresh)
                        or quiet_hour(
                            time.time(),
                            int(self._get("quiet_start", 23)),
                            int(self._get("quiet_end", 9)),
                        )
                    ):
                        self.store.finish_proactive(job, "failed")
                        continue
                    attempted = True
                    sent = await asyncio.wait_for(
                        self.context.send_message(
                            fresh["umo"],
                            MessageChain(chain=[At(qq=fresh["user_id"]), Plain(message)]),
                        ),
                        timeout=30,
                    )
                    self.store.finish_proactive(
                        job, "sent" if sent else "failed", message if sent else ""
                    )
                    if sent and followups:
                        self.store.finish_followup(followups[0]["id"])
            except asyncio.CancelledError:
                self.store.finish_proactive(job, "unknown" if attempted else "failed")
                raise
            except Exception as exc:
                self.store.finish_proactive(job, "unknown" if attempted else "failed")
                logger.warning("主动互动已跳过：%s", type(exc).__name__)

    async def _can_contact(self, profile):
        """Check actual OneBot membership and the bot's current mute status."""
        try:
            _, platform_id, self_id, group_id, user_id = json.loads(profile["key"])
            platform = self.context.get_platform_inst(platform_id)
            if platform is None or platform.meta().name != "aiocqhttp":
                return False
            client = platform.get_client()
            for target in (user_id, self_id):
                info = await asyncio.wait_for(
                    client.call_action(
                        "get_group_member_info",
                        group_id=int(group_id),
                        user_id=int(target),
                        no_cache=True,
                        self_id=self_id,
                    ),
                    timeout=10,
                )
                if not isinstance(info, dict) or str(info.get("user_id")) != target:
                    return False
                if target == self_id and number(info.get("shut_up_timestamp")) > time.time():
                    return False
            return True
        except Exception:
            return False

    async def _worker(self):
        summary_at = 0
        while True:
            try:
                now = time.time()
                if self._get("enabled", True):
                    if now >= summary_at:
                        await self._summaries(now)
                        self.store.prune(now - max(1, int(self._get("retention_days", 30))) * 86400)
                        summary_at = now + 3600
                    interval = max(300, int(self._get("check_minutes", 15)) * 60)
                    if now - self._last_check >= interval:
                        self._last_check = now
                        await self._proactive(now)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("关系插件维护任务将在下轮重试：%s", type(exc).__name__)
            await asyncio.sleep(60)

    async def terminate(self):
        self._closed = True
        self.dashboard.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.store.close()
