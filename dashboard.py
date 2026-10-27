"""Authenticated AstrBot Plugin View API for relationship and user profiles."""

import hashlib
import json
import re

from astrbot.api.web import error_response, json_response, request

PLUGIN_NAME = "astrbot_plugin_relationship_memory"


def public_id(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class Dashboard:
    def __init__(self, plugin):
        self.plugin = plugin
        for endpoint, handler, method in (
            ("users", self.users, "GET"),
            ("user", self.user, "GET"),
            ("tags/save", self.save_tag, "POST"),
            ("tags/delete", self.delete_tag, "POST"),
        ):
            plugin.context.register_web_api(
                f"/{PLUGIN_NAME}/{endpoint}", handler, [method], "关系与用户画像管理"
            )

    def close(self):
        apis = self.plugin.context.registered_web_apis
        apis[:] = [item for item in apis if getattr(item[1], "__self__", None) is not self]

    def _guard(self):
        # The host already authenticates plugin-scope requests. Require the bound
        # Dashboard identity as well, rather than exposing another public server.
        if not isinstance(request.username, str) or not request.username.strip():
            return error_response("请先登录 AstrBot 管理面板。", status_code=401)
        if self.plugin._closed:
            return error_response("插件已停用，请刷新页面。", status_code=503)
        return None

    def _keys(self):
        role_id = self.plugin._get("role_id", "default")
        keys = []
        for profile in self.plugin.store.profiles():
            try:
                identity = json.loads(profile["key"])
                if isinstance(identity, list) and len(identity) == 5 and identity[0] == role_id:
                    keys.append(profile["key"])
            except (TypeError, ValueError):
                continue
        return keys

    def _resolve(self, identity):
        if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
            return None
        return next((key for key in self._keys() if public_id(key) == identity), None)

    @staticmethod
    def _revision(payload):
        revision = payload.get("revision")
        if type(revision) is not int or revision < 0:
            raise ValueError("缺少有效的数据版本，请刷新后再操作。")
        return revision

    async def users(self):
        denied = self._guard()
        if denied is not None:
            return denied
        try:
            limit = max(1, min(50, int(request.query.get("limit", "20"))))
            offset = max(0, min(1000000, int(request.query.get("offset", "0"))))
        except (ValueError, TypeError):
            return error_response("分页参数无效。", status_code=400)
        query = str(request.query.get("q", ""))[:100].strip()
        group = str(request.query.get("group", ""))[:128].strip()
        result = self.plugin.store.dashboard_profiles(
            query=query, group_id=group, offset=offset, limit=limit, keys=self._keys()
        )
        for item in result["items"]:
            item["id"] = public_id(item.pop("key"))
            item["group_enabled"] = self.plugin._enabled_group(item["group_id"])
        result["role_id"] = self.plugin._get("role_id", "default")
        result["offset"] = offset
        result["limit"] = limit
        return json_response(result)

    async def user(self):
        denied = self._guard()
        if denied is not None:
            return denied
        key = self._resolve(request.query.get("id"))
        if key is None:
            return error_response("未找到当前角色下的用户档案。", status_code=404)
        detail = self.plugin.store.dashboard_detail(key)
        if not detail:
            return error_response("用户档案已不存在，请刷新。", status_code=404)
        profile = detail["profile"]
        profile.pop("key", None)
        profile.pop("persona", None)
        profile.pop("umo", None)
        profile.pop("provider_id", None)
        profile["id"] = public_id(key)
        profile["group_enabled"] = self.plugin._enabled_group(profile["group_id"])
        # All nested rows must omit internal relation keys and model settings.
        for field in ("tags", "memories", "daily_summaries", "changes", "proactive"):
            for row in detail.get(field, []):
                row.pop("key", None)
        return json_response(detail)

    async def save_tag(self):
        denied = self._guard()
        if denied is not None:
            return denied
        payload = await request.json(default=None)
        denied = self._guard()
        if denied is not None:
            return denied
        if not isinstance(payload, dict):
            return error_response("请求内容必须是JSON对象。", status_code=400)
        key = self._resolve(payload.get("id"))
        if key is None:
            return error_response("未找到当前角色下的用户档案。", status_code=404)
        try:
            revision = self._revision(payload)
            name, category = payload.get("name"), payload.get("category")
            visible = payload.get("public")
            if not isinstance(name, str) or not name.strip() or len(name.strip()) > 24:
                raise ValueError("标签名称须为1～24个字符。")
            if category not in {
                "interest",
                "preference",
                "communication",
                "boundary",
                "impression",
            }:
                raise ValueError("标签分类无效。")
            if type(visible) is not bool:
                raise ValueError("请明确选择标签能否用于群聊回复。")
            async with self.plugin._lock(key):
                if self.plugin._closed:
                    return error_response("插件已停用，请刷新页面。", status_code=503)
                saved = self.plugin.store.update_tag(
                    key, name.strip(), category, visible, expected_revision=revision
                )
            if not saved:
                return error_response("档案已更新，请刷新后再保存。", status_code=409)
        except ValueError as exc:
            return error_response(str(exc), status_code=400)
        return json_response({"saved": True})

    async def delete_tag(self):
        denied = self._guard()
        if denied is not None:
            return denied
        payload = await request.json(default=None)
        denied = self._guard()
        if denied is not None:
            return denied
        if not isinstance(payload, dict):
            return error_response("请求内容必须是JSON对象。", status_code=400)
        key = self._resolve(payload.get("id"))
        if key is None:
            return error_response("未找到当前角色下的用户档案。", status_code=404)
        try:
            revision = self._revision(payload)
            tag_id = payload.get("tag_id")
            if type(tag_id) is not int or tag_id < 1:
                raise ValueError("标签标识无效。")
            async with self.plugin._lock(key):
                if self.plugin._closed:
                    return error_response("插件已停用，请刷新页面。", status_code=503)
                deleted = self.plugin.store.delete_tag(key, tag_id, expected_revision=revision)
            if not deleted:
                return error_response("标签或档案已变化，请刷新后再操作。", status_code=409)
        except ValueError as exc:
            return error_response(str(exc), status_code=400)
        return json_response({"deleted": True})
