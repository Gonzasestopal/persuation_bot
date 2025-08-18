import re
from typing import Tuple

_ALLOWED = {"pro", "con"}

_TOPIC_RE = re.compile(r"(?i)\btopic\s*:\s*(?P<topic>.+?)(?=,?\s*\bside\b|$)")
_SIDE_RE = re.compile(r"(?i)\bside\s*:\s*(?P<side>\w+)")
_MARKERS_RE = re.compile(r"(?i)\b(topic|side)\s*:")


def parse_topic_side(text: str) -> Tuple[str, str]:
    if not text or not text.strip():
        raise ValueError("message must not be empty")

    m_topic = _TOPIC_RE.search(text)
    m_side = _SIDE_RE.search(text)

    if not m_topic and not m_side:
        raise ValueError("message must contain Topic: and Side: fields")
    if not m_topic:
        raise ValueError("topic is missing")
    if not m_side:
        raise ValueError("side is missing")

    topic = m_topic.group("topic").strip(" .,\n\t")
    side = m_side.group("side").strip().lower()

    if side not in _ALLOWED:
        raise ValueError("side must be 'pro' or 'con'")
    if not topic:
        raise ValueError("topic must not be empty")

    return topic, side


def assert_no_topic_or_side_markers(text: str) -> None:
    if not text or not text.strip():
        raise ValueError("message must not be empty.")
    if _MARKERS_RE.search(text):
        raise ValueError("topic/side must not be provided when continuing a conversation")
