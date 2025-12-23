"""Utilities for transcript line handling and timing."""

from __future__ import annotations

from typing import Dict, List


SUFFIX_WORDS = {
    "거예요",
    "거에요",
    "거죠",
    "거야",
    "거랍니다",
    "거랍니다",
    "거였죠",
    "거였어",
    "입니다",
    "합니다",
    "했어요",
    "했죠",
    "했어",
    "됐죠",
    "됐어요",
    "되죠",
    "되요",
    "됩니다",
    "되네요",
    "그렇죠",
    "그래요",
    "그래서요",
    "라고요",
    "라고",
    "네요",
    "군요",
    "죠",
    "요",
    "예요",
    "에요",
}

SUFFIX_PREFIXES = (
    "거",
    "죠",
    "요",
    "예",
    "에",
    "네요",
    "군요",
    "랍니다",
    "랍니다",
    "듯",
    "듯이",
    "라고",
    "라고요",
)


def wrap_text(text: str, max_chars: int, max_lines: int = 0) -> List[str]:
    """Wrap text into lines constrained by max_chars."""
    words = text.replace("\n", " ").split()
    if not words:
        return []
    lines: List[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
            continue
        if len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if max_lines > 0:
        lines = lines[:max_lines]
    return fix_korean_line_breaks(lines, max_chars)


def fix_korean_line_breaks(lines: List[str], max_chars: int) -> List[str]:
    """Avoid awkward Korean splits by adjusting word boundaries."""
    def normalize_token(token: str) -> str:
        return token.strip().strip(".,!?\"'()[]{}")

    fixed = list(lines)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(fixed)):
            prev = fixed[i - 1].strip()
            curr = fixed[i].strip()
            if not prev or not curr:
                continue

            curr_words = curr.split()
            prev_words = prev.split()
            if not curr_words or not prev_words:
                continue

            first_norm = normalize_token(curr_words[0])
            curr_is_suffixy = False
            if len(curr_words) == 1:
                curr_is_suffixy = (
                    first_norm in SUFFIX_WORDS or first_norm.startswith(SUFFIX_PREFIXES)
                )
            else:
                curr_is_suffixy = first_norm in SUFFIX_WORDS

            if not curr_is_suffixy:
                continue

            merged = f"{prev} {curr}".strip()
            if len(merged) <= max_chars:
                fixed[i - 1] = merged
                fixed.pop(i)
                changed = True
                break

            if len(prev_words) >= 2:
                candidate_word = prev_words[-1]
                new_prev = " ".join(prev_words[:-1]).strip()
                new_curr = f"{candidate_word} {curr}".strip()
                if new_prev and len(new_prev) <= max_chars and len(new_curr) <= max_chars:
                    fixed[i - 1] = new_prev
                    fixed[i] = new_curr
                    changed = True
                    break

    return [line for line in fixed if line.strip()]


def split_sentence(
    sentence: Dict[str, float],
    max_chars: int,
    gap_sec: float = 0.0,
) -> List[Dict[str, float]]:
    """Split a sentence by max chars and allocate timing proportionally."""
    text = str(sentence.get("text", "")).strip()
    if len(text) <= max_chars:
        return [sentence]

    chunks = wrap_text(text, max_chars, 0)
    if not chunks:
        return [sentence]

    t0 = float(sentence.get("t0", 0.0))
    t1 = float(sentence.get("t1", 0.0))
    duration = t1 - t0
    if duration <= 0:
        return [sentence]

    weights = [max(1, len(chunk)) for chunk in chunks]
    total = float(sum(weights))
    gap_total = gap_sec * (len(chunks) - 1)
    if gap_total >= duration:
        gap_sec = 0.0
        gap_total = 0.0
    available = max(0.0, duration - gap_total)

    curr_t0 = t0
    created: List[Dict[str, float]] = []
    for idx, chunk in enumerate(chunks):
        if idx == len(chunks) - 1:
            curr_t1 = t1
        else:
            ratio = float(weights[idx]) / total
            curr_t1 = curr_t0 + (available * ratio)
        created.append(
            {
                "segment_id": sentence.get("segment_id"),
                "t0": float(curr_t0),
                "t1": float(curr_t1),
                "text": chunk,
                "confidence": sentence.get("confidence"),
                "metadata": {**(sentence.get("metadata") or {}), "split_from": sentence.get("id")},
            }
        )
        curr_t0 = curr_t1 + gap_sec

    return created
