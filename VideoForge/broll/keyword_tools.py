"""Keyword normalization and lightweight Korean -> English mappings."""
from __future__ import annotations

import re
from typing import Iterable, List

KOREAN_PARTICLES = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "에서",
    "의",
    "와",
    "과",
    "도",
    "만",
    "으로",
    "로",
    "까지",
    "부터",
    "보다",
}

KOREAN_SUFFIXES = (
    "입니다",
    "이다",
    "이라",
    "인데",
    "지만",
    "합니다",
    "했습니다",
    "했어요",
    "했죠",
    "습니다",
    "어요",
    "아요",
    "했다",
    "한다",
    "하는",
)

KOREAN_KEYWORD_MAP: dict[str, List[str]] = {
    "돈": ["money", "cash"],
    "현금": ["cash"],
    "재산": ["savings", "assets"],
    "저축": ["savings"],
    "통장": ["bank account", "passbook"],
    "은행": ["bank"],
    "만원": ["cash", "money"],
    "원": ["money"],
    "지폐": ["banknote", "cash"],
    "동전": ["coin"],
    "계산기": ["calculator"],
    "고지서": ["bill", "notice"],
    "청구서": ["bill", "invoice"],
    "과태료": ["fine", "ticket"],
    "벌금": ["fine", "penalty"],
    "서류": ["paperwork", "document"],
    "문서": ["document", "paperwork"],
    "계약서": ["contract", "document"],
    "퇴사": ["resignation", "office"],
    "사표": ["resignation"],
    "실직": ["unemployment"],
    "회사": ["office", "workplace"],
    "직장": ["office", "workplace"],
    "월세": ["rent"],
    "관리비": ["utility bill"],
    "전기세": ["electric bill", "utility bill"],
    "수도세": ["utility bill"],
    "구독": ["subscription"],
    "해지": ["cancel", "unsubscribe"],
    "취소": ["cancel"],
    "스마트폰": ["smartphone", "phone"],
    "휴대폰": ["mobile phone", "phone"],
    "핸드폰": ["phone"],
    "앱": ["app"],
    "신호등": ["traffic light"],
    "횡단보도": ["crosswalk"],
    "교차로": ["intersection"],
    "빨간불": ["red light", "traffic light"],
    "어린이": ["child", "children"],
    "학교": ["school"],
    "보호구역": ["school zone"],
    "도시": ["city"],
    "거리": ["street"],
    "창문": ["window"],
    "실루엣": ["silhouette"],
    "외로움": ["lonely", "solitude"],
    "우울": ["sad", "melancholy"],
}


def normalize_korean_token(token: str) -> str:
    cleaned = re.sub(r"[^A-Za-z가-힣]", "", str(token))
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    for suffix in KOREAN_SUFFIXES:
        if lowered.endswith(suffix) and len(lowered) > len(suffix) + 1:
            lowered = lowered[: -len(suffix)]
            break
    for suffix in sorted(KOREAN_PARTICLES, key=len, reverse=True):
        if lowered.endswith(suffix) and len(lowered) > len(suffix) + 1:
            lowered = lowered[: -len(suffix)]
            break
    return lowered


def translate_keywords(tokens: Iterable[str]) -> List[str]:
    translated: List[str] = []
    for token in tokens:
        normalized = normalize_korean_token(token)
        if not normalized:
            continue
        if re.search(r"[A-Za-z]", normalized):
            translated.append(normalized)
            continue
        mapped = KOREAN_KEYWORD_MAP.get(normalized)
        if mapped:
            translated.extend(mapped)
    return _dedupe_preserve_order(translated)


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
