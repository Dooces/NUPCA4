#!/usr/bin/env python3
"""
tools/audit_types_contract.py

VERSION: v1.5b-perf.10 (2025-12-20)

One-pass audit of the public symbol contract for `nupca3.types`.

This tool is deliberately robust to two common failure modes:
  (1) Running from `tools/` where repo root is not on sys.path
  (2) `nupca3/types.py` currently being partially broken

It reports the full set of symbols referenced from `nupca3.types` across the repo,
so you do NOT get "one missing import name per rerun" whack-a-mole.

What counts as a required symbol:
  - `from nupca3.types import X`
  - `from .types import X` (inside the package)
  - `import nupca3.types as T` followed by `T.X` usage
  - `from nupca3 import types as T` followed by `T.X` usage

Exit codes:
  0: no missing symbols detected
  1: missing symbols detected
  2: repo layout not found / unrecoverable errors
"""

from __future__ import annotations

import ast
import sys
import traceback
import runpy
from pathlib import Path
from typing import Dict, Set, Tuple


def repo_root() -> Path:
    # tools/audit_types_contract.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _safe_parse(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return None


def collect_required_symbols(nupca3_dir: Path) -> Set[str]:
    required: Set[str] = set()

    for p in nupca3_dir.rglob("*.py"):
        tree = _safe_parse(p)
        if tree is None:
            continue

        # Track aliases bound to the types module within this file.
        # alias_name -> kind ("abs" or "rel") just for completeness
        types_aliases: Set[str] = set()

        for node in ast.walk(tree):
            # from nupca3.types import X
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "nupca3.types":
                    for a in node.names:
                        if a.name != "*":
                            required.add(a.name)
                # relative import inside package: from .types import X
                if mod == "types" and node.level == 1:
                    for a in node.names:
                        if a.name != "*":
                            required.add(a.name)

                # from nupca3 import types as T
                if mod == "nupca3" and node.level == 0:
                    for a in node.names:
                        if a.name == "types":
                            types_aliases.add(a.asname or a.name)

                # from . import types as T  (inside package)
                if mod == "" and node.level == 1:
                    for a in node.names:
                        if a.name == "types":
                            types_aliases.add(a.asname or a.name)

            # import nupca3.types as T
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "nupca3.types":
                        types_aliases.add(a.asname or "nupca3_types")

            # collect attribute usage: T.X for any alias T bound above
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in types_aliases:
                    required.add(node.attr)

    return required


def provided_symbols(types_path: Path) -> Tuple[Set[str], str]:
    # Best fidelity: execute the file and look at its globals.
    try:
        g = runpy.run_path(str(types_path))
        return set(g.keys()), "EXEC"
    except Exception:
        # Fallback: names defined in the file (won’t catch re-exports)
        try:
            tree = ast.parse(types_path.read_text(encoding="utf-8"), filename=str(types_path))
            provided: Set[str] = set()
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    provided.add(node.name)
                elif isinstance(node, ast.FunctionDef):
                    provided.add(node.name)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    provided.add(node.target.id)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            provided.add(t.id)
            return provided, "AST_FALLBACK"
        except Exception:
            return set(), "FAILED"


def main() -> int:
    root = repo_root()

    # Ensure repo root is on sys.path (so local imports resolve in EXEC mode).
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    nupca3_dir = root / "nupca3"
    types_path = nupca3_dir / "types.py"

    if not nupca3_dir.exists():
        print(f"ERROR: expected directory not found: {nupca3_dir}")
        return 2
    if not types_path.exists():
        print(f"ERROR: expected file not found: {types_path}")
        return 2

    required = collect_required_symbols(nupca3_dir)
    provided, mode = provided_symbols(types_path)

    if mode != "EXEC":
        print("WARN: could not execute nupca3/types.py; using", mode)
        print("      (AST fallback may miss names re-exported from other modules.)")

    missing = sorted([x for x in required if x not in provided])

    print(f"repo_root: {root}")
    print(f"types_path: {types_path}")
    print(f"mode: {mode}")
    print(f"required_from_types: {len(required)}")
    print(f"provided_symbols:     {len(provided)}")
    print(f"missing:             {len(missing)}")
    for x in missing:
        print(f"  - {x}")

    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
