import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy import (
    TZ,
    clean_tags,
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
