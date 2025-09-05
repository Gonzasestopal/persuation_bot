import re
from typing import Tuple

from app.domain.enums import Stance
from app.domain.errors import InvalidContinuationMessage, InvalidStartMessage

TOPIC_MAX_LENGTH = 100


_TOPIC_RE = re.compile(r'(?i)\btopic\s*:\s*(?P<topic>.+?)(?=,?\s*\bside\b|$)')
_SIDE_RE = re.compile(r'(?i)\bside\s*:\s*(?P<side>\w+)')
_MARKERS_RE = re.compile(r'(?i)\b(topic|side)\s*:')


def _to_stance(s: str) -> Stance:
    s_norm = s.strip().lower()
    return Stance(s_norm)  # raises ValueError if invalid


def parse_topic_side(text: str) -> Tuple[str, Stance]:
    if not text or not text.strip():
        raise InvalidStartMessage('message must not be empty')

    m_topic = _TOPIC_RE.search(text)
    m_side = _SIDE_RE.search(text)

    if not m_topic and not m_side:
        raise InvalidStartMessage('message must contain Topic: and Side: fields')
    if not m_topic:
        raise InvalidStartMessage('topic is missing')
    if not m_side:
        raise InvalidStartMessage('side is missing')

    topic = m_topic.group('topic').strip(' .,\n\t')
    stance = m_side.group('side').strip().lower()

    try:
        stance = _to_stance(stance)
    except ValueError:
        raise InvalidStartMessage("side must be 'pro' or 'con'")

    if not topic:
        raise InvalidStartMessage('topic must not be empty')

    if len(topic) >= TOPIC_MAX_LENGTH:
        raise InvalidStartMessage('topic must be less than 100 characters')

    return topic, stance


def assert_no_topic_or_side_markers(text: str) -> None:
    if not text or not text.strip():
        raise InvalidContinuationMessage('message must not be empty.')
    if _MARKERS_RE.search(text):
        raise InvalidContinuationMessage(
            'topic/side must not be provided when continuing a conversation'
        )
