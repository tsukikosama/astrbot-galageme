"""Build model history only from this plugin's sender-scoped records."""

import copy
import re

_GROUP_CONTEXT = re.compile(
    r"<system_reminder>\s*You are in a group chat\.[\s\S]*?</system_reminder>",
    re.IGNORECASE,
)
_SHARED_MARKERS = (
    "--- BEGIN CONTEXT---",
    "--- BEGIN CONTEXT ---",
    "## 相关记忆",
    "<relationship_context>",
    "<iris_memory",
)


def scoped_parts(parts):
    """Retain original current-message media/reminders, excluding shared context."""
    result = []
    for original in parts:
        part = copy.deepcopy(original)
        value = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if isinstance(value, str):
            value = _GROUP_CONTEXT.sub("", value).strip()
            if not value or any(marker in value for marker in _SHARED_MARKERS):
                continue
            if isinstance(part, dict):
                part["text"] = value
            else:
                part.text = value
        result.append(part)
    return result


def _clip(value, limit):
    value = value or ""
    return value if len(value) <= limit else value[:limit]


def own_history(turns, clip=2000):
    """Rebuild the dialogue from this sender's own records.

    The host conversation history is shared across a group and can carry other
    people's lines, so the model gets these sender-scoped turns instead.
    """
    result = []
    for turn in turns:
        # Image-only turns have no stored image here; do not invent an empty user turn.
        if not turn["user_text"]:
            continue
        result.append({"role": "user", "content": _clip(turn["user_text"], clip)})
        if turn["assistant_text"]:
            result.append({"role": "assistant", "content": _clip(turn["assistant_text"], clip)})
    return result
