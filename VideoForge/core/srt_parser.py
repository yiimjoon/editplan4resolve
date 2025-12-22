import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class SrtEntry:
    index: int
    start: float
    end: float
    text: str


TIME_RE = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})"
)


def _parse_timestamp(raw: str) -> float:
    match = TIME_RE.search(raw)
    if not match:
        return 0.0
    h = int(match.group("h"))
    m = int(match.group("m"))
    s = int(match.group("s"))
    ms = int(match.group("ms"))
    return h * 3600.0 + m * 60.0 + s + (ms / 1000.0)


def parse_srt(path: str | Path) -> List[SrtEntry]:
    content = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\r?\n\r?\n", content.strip())
    entries: List[SrtEntry] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        try:
            index = int(lines[0])
        except ValueError:
            continue
        times = lines[1]
        if "-->" not in times:
            continue
        start_raw, end_raw = [part.strip() for part in times.split("-->", 1)]
        start = _parse_timestamp(start_raw)
        end = _parse_timestamp(end_raw)
        text = " ".join(lines[2:]).strip()
        entries.append(SrtEntry(index=index, start=start, end=end, text=text))
    return entries
