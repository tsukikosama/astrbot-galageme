"""Authenticated Web API contract tests against the real plugin store."""

import importlib
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from .test_plugin import Context, Event, plugin
else:
    from test_plugin import Context, Event, plugin

dashboard = importlib.import_module("astrbot_plugin_relationship_memory.dashboard")


class Request:
    username = "dashboard-admin"

    def __init__(self):
        self.query = {}
        self.payload = {}

    async def json(self, default=None):
        return self.payload


class DashboardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(plugin, "get_astrbot_data_path", return_value=self.tmp.name)
        self.worker_patch = patch.object(plugin.Main, "_start_worker")
        self.path_patch.start()
        self.worker_patch.start()
        self.context = Context()
        self.bot = plugin.Main(self.context, {"enabled_groups": ["20001"]})
        self.request = Request()
        self.request_patch = patch.object(dashboard, "request", self.request)
        self.request_patch.start()
        self.event = Event()
        self.key = self.bot._key(self.event)
        self.bot.store.ensure_profile(
            self.key,
            "20001",
            "10001",
            "platform:GroupMessage:20001",
            persona="不应该暴露的人设",
            display_name="小林",
        )
        self.id = dashboard.public_id(self.key)

    async def asyncTearDown(self):
        await self.bot.terminate()
        self.request_patch.stop()
        self.worker_patch.stop()
        self.path_patch.stop()
        self.tmp.cleanup()

    async def test_web_routes_are_registered_and_removed_on_unload(self):
        self.assertEqual(len(self.context.registered_web_apis), 4)
        self.bot.dashboard.close()
        self.assertEqual(self.context.registered_web_apis, [])

    async def test_anonymous_requests_are_denied(self):
        self.request.username = None
        for handler in (
            self.bot.dashboard.users,
            self.bot.dashboard.user,
            self.bot.dashboard.save_tag,
            self.bot.dashboard.delete_tag,
        ):
            self.assertEqual((await handler())["status_code"], 401)

    async def test_list_is_scoped_paginated_and_does_not_expose_persona(self):
        other_key = self.key.replace('"default"', '"other-role"')
        self.bot.store.ensure_profile(other_key, "20001", "10001", "other")
        self.request.query = {"q": "小林", "limit": "1000"}
        result = (await self.bot.dashboard.users())["payload"]
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["limit"], 50)
        self.assertEqual(result["items"][0]["id"], self.id)
        for private in ("key", "persona", "umo", "style", "provider_id"):
            self.assertNotIn(private, result["items"][0])

    async def test_invalid_ids_cannot_access_another_role(self):
        other_key = self.key.replace('"default"', '"other-role"')
        self.bot.store.ensure_profile(other_key, "20001", "10001", "other")
        self.request.query = {"id": dashboard.public_id(other_key)}
        self.assertEqual((await self.bot.dashboard.user())["status_code"], 404)
        self.request.query = {"id": "../../etc/passwd"}
        self.assertEqual((await self.bot.dashboard.user())["status_code"], 404)

    async def test_authenticated_detail_includes_private_memory_but_not_system_prompt(self):
        self.bot.store.commit_turn(
            self.key,
            "a",
            "秘密",
            "回复",
            {"confidence": 1, "memories": [{"content": "非公开内容", "public": False}]},
        )
        self.request.query = {"id": self.id}
        result = (await self.bot.dashboard.user())["payload"]
        self.assertEqual(result["profile"]["id"], self.id)
        self.assertNotIn("persona", result["profile"])
        self.assertNotIn("key", result["profile"])
        self.assertEqual(result["memories"][0]["content"], "非公开内容")
        self.assertFalse(result["memories"][0]["public"])

    async def test_list_and_detail_include_score_levels(self):
        with self.bot.store._transaction():
            self.bot.store.db.execute(
                "UPDATE profiles SET affection=42,trust=70,familiarity=25,closeness=0 WHERE key=?",
                (self.key,),
            )
        self.request.query = {"id": self.id}
        profile = (await self.bot.dashboard.user())["payload"]["profile"]
        levels = profile["levels"]
        self.assertEqual(levels["affection"]["name"], "熟悉")
        self.assertEqual(levels["affection"]["label"], "Lv.3 熟悉")
        self.assertEqual(levels["trust"]["name"], "亲密")
        self.assertEqual(levels["familiarity"]["name"], "认识")
        self.assertEqual(levels["closeness"]["name"], "陌生")
        self.assertEqual(levels["affection"]["level"], 3)
        self.assertEqual(levels["trust"]["level"], 5)
        self.assertEqual(levels["familiarity"]["level"], 2)
        self.assertEqual(levels["closeness"]["level"], 1)
        for dimension in ("affection", "trust", "familiarity", "closeness"):
            self.assertTrue(levels[dimension]["description"])
        self.request.query = {"limit": "20"}
        listed = (await self.bot.dashboard.users())["payload"]["items"][0]
        self.assertEqual(listed["levels"]["affection"]["name"], "熟悉")
        # Levels are display-only: they must not be persisted as new score fields.
        self.assertNotIn("levels", self.bot.store.profile(self.key))

    async def test_tag_save_delete_and_stale_revision(self):
        revision = self.bot.store.profile(self.key)["revision"]
        self.request.payload = {
            "id": self.id,
            "revision": revision,
            "name": "喜欢科幻",
            "category": "interest",
            "public": False,
        }
        self.assertEqual((await self.bot.dashboard.save_tag())["status_code"], 200)
        self.assertEqual((await self.bot.dashboard.save_tag())["status_code"], 409)
        tag = self.bot.store.tags(self.key)[0]
        self.request.payload = {
            "id": self.id,
            "revision": self.bot.store.profile(self.key)["revision"],
            "tag_id": tag["id"],
        }
        self.assertEqual((await self.bot.dashboard.delete_tag())["status_code"], 200)
        self.assertEqual(self.bot.store.tags(self.key), [])

    async def test_json_boolean_and_name_validation(self):
        base = {
            "id": self.id,
            "revision": 0,
            "name": "喜欢科幻",
            "category": "interest",
            "public": False,
        }
        for replacement in (
            {"public": "false"},
            {"name": "x" * 25},
            {"category": "invalid"},
            {"revision": True},
        ):
            self.request.payload = dict(base, **replacement)
            self.assertEqual((await self.bot.dashboard.save_tag())["status_code"], 400)
        self.request.payload = []
        self.assertEqual((await self.bot.dashboard.save_tag())["status_code"], 400)
        self.assertEqual(self.bot.store.tags(self.key), [])

    async def test_tag_id_cannot_delete_another_users_tag(self):
        other_event = Event(user="10002")
        other_key = self.bot._key(other_event)
        self.bot.store.ensure_profile(other_key, "20001", "10002", "other")
        self.bot.store.update_tag(other_key, "另一个人的标签")
        tag = self.bot.store.tags(other_key)[0]
        self.request.payload = {
            "id": self.id,
            "revision": self.bot.store.profile(self.key)["revision"],
            "tag_id": tag["id"],
        }
        self.assertEqual((await self.bot.dashboard.delete_tag())["status_code"], 409)
        self.assertEqual(len(self.bot.store.tags(other_key)), 1)
