"""
nupca3/step_pipeline.py

Loader for the split step pipeline implementation.
"""

from __future__ import annotations

from pathlib import Path


def _load_parts() -> None:
    base = Path(__file__).resolve().parent / "step_pipeline_parts"
    parts = [base / f"part{idx}.part" for idx in range(1, 5)]
    code = "".join(part.read_text(encoding="utf-8") for part in parts)
    exec(compile(code, str(base / "step_pipeline_concat"), "exec"), globals())


_load_parts()
