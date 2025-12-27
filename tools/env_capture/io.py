"""File IO helpers for environment screen capture."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def write_frame(lines: Iterable[str], *, out_file: Path, echo: bool) -> None:
    payload = "\n".join(lines) + "\n"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("a", encoding="utf-8") as handle:
        handle.write(payload)
    if echo:
        print(payload, end="")
