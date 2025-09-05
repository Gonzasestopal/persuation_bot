# app/shared/text_utils.py
import re
from typing import Dict

# regex compilados una vez, multi-idioma
WORD_RX = re.compile(r'[^\W\d_]+', flags=re.UNICODE)
SENT_SPLIT_RX = re.compile(r'(?<=[.!?¿\?¡!])\s+')
IS_QUESTION_RX = re.compile(r'[¿\?]\s*$')
END_MARKERS_RX = re.compile(
    r'(match concluded\.?|debate concluded|debate is over)',
    flags=re.IGNORECASE,
)


def trunc(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[:n] + '…'


def round3(d: Dict[str, float]) -> Dict[str, float]:
    return {
        k: round(float(d.get(k, 0.0)), 3)
        for k in ('entailment', 'neutral', 'contradiction')
    }


def normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def sanitize_end_markers(text: str) -> str:
    return normalize_spaces(END_MARKERS_RX.sub('', text))


def drop_questions(text: str) -> str:
    sents = [s.strip() for s in re.split(SENT_SPLIT_RX, text) if s.strip()]
    sents = [s for s in sents if not IS_QUESTION_RX.search(s)]
    out = ' '.join(sents) if sents else text
    return re.sub(r'\.\.+$', '.', out).strip()


def word_count(text: str) -> int:
    return len(WORD_RX.findall(text))
