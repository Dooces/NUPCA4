#!/usr/bin/env python3
"""
drive_dir_audit.py

Parse Windows `dir` text dumps (the classic "Directory of X:\\...") and generate
cleanup + organization reports.

Inputs supported:
  - a .zip containing one or more .txt dumps
  - one or more .txt files
  - a folder containing .txt dumps

Outputs (written to --out):
  - audit.sqlite (portable database you can query later)
  - summary.txt (human-readable)
  - drive_totals.csv
  - top_root_folders.csv
  - extensions.csv
  - largest_files.csv
  - old_large_files.csv
  - dup_name_size_groups.csv + dup_name_size_paths.csv
  - dup_relpath_size_groups.csv + dup_relpath_size_paths.csv
  - delete_candidates.csv (REVIEW LIST; does not delete anything)

Typical usage:
  python drive_dir_audit.py --input drives.zip --out out

Notes:
  - Duplicate detection is heuristic because the dump has no hashes. It groups by:
      (file name + size) and (relative path + size).
  - Deletion candidates are conservative (system paths excluded by default).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import re
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

DIR_HEADER_RE = re.compile(r"^\s*Directory of\s+(?P<path>.+?)\s*$", re.IGNORECASE)

# Example lines:
# 10/19/2022  06:52 PM           112,104 appverifUI.dll
# 07/06/2025  04:20 PM    <DIR>          ESD
ENTRY_RE = re.compile(
    r"^\s*(?P<date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<time>\d{1,2}:\d{2})\s+"
    r"(?P<ampm>AM|PM)\s+"
    r"(?P<size_or_tag><[^>]+>|[\d,]+)\s+"
    r"(?P<name>.+?)\s*$"
)

SKIP_PREFIXES = (
    "Volume in drive",
    "Volume Serial Number",
    "Total Files Listed",
)

SKIP_LINE_RE = re.compile(
    r"^\s*(?:\d+\s+File\(s\)|\d+\s+Dir\(s\)|File\(s\)|Dir\(s\))\b|^\s*$",
    re.IGNORECASE
)

DEFAULT_SYSTEM_EXCLUDES = [
    r"^C:\\Windows\\",
    r"^C:\\Program Files\\",
    r"^C:\\Program Files \(x86\)\\",
    r"^C:\\ProgramData\\",
    r"^C:\\Users\\[^\\]+\\AppData\\",
    r"^C:\\$",
]

TRASHY_PATH_HINTS = [
    r"\\downloads\\",
    r"\\download\\",
    r"\\temp\\",
    r"\\tmp\\",
    r"\\cache\\",
    r"\\recycle\.bin\\",
    r"\\\$recycle\.bin\\",
    r"\\crashdumps\\",
    r"\\logs\\",
]

TRASHY_EXTS = {"tmp", "log", "dmp", "bak", "old", "etl", "chk"}
ARCHIVE_EXTS = {"zip", "7z", "rar", "iso", "img", "gz", "bz2", "xz", "tar", "tgz", "zst"}
INSTALLER_EXTS = {"exe", "msi", "msp", "cab", "appx", "msix", "dmg", "pkg"}


def _safe_join_dir(dir_path: str, name: str) -> str:
    return f"{dir_path}{name}" if dir_path.endswith("\\") else f"{dir_path}\\{name}"


def _drive_letter(path: str) -> str:
    if len(path) >= 2 and path[1] == ":":
        return path[0].upper()
    return "?"


def _relpath_from_drive(full_path: str) -> str:
    # "E:\\backup\\foo" -> "backup\\foo"
    if len(full_path) >= 3 and full_path[1:3] == ":\\":
        return full_path[3:]
    return full_path


def _root1(relpath: str) -> str:
    if not relpath:
        return ""
    parts = relpath.split("\\")
    return parts[0] if parts else ""


def _ext_of(name: str) -> str:
    i = name.rfind(".")
    if i <= 0 or i == len(name) - 1:
        return ""
    return name[i + 1 :].lower()


def iter_text_sources(input_path: Path) -> Iterator[Tuple[str, io.TextIOBase]]:
    """
    Yield (source_name, text_stream) pairs.
    Supports zip, directory, or single file.
    """
    if input_path.is_dir():
        for p in sorted(input_path.glob("*.txt")):
            yield p.name, p.open("r", encoding="utf-8", errors="replace", newline="")
        return

    if input_path.is_file():
        yield input_path.name, input_path.open("r", encoding="utf-8", errors="replace", newline="")
        return

    raise FileNotFoundError(str(input_path))


def parse_dir_dump(
    source_name: str,
    stream: io.TextIOBase,
    *,
    emit_dirs: bool = False,
) -> Iterator[Tuple[bool, str, str, str, int, int, str, str, str]]:
    """
    Yields tuples:
      (is_dir, drive, dir_path, name, size_bytes, mtime_ts, full_path, relpath, root1)

    For directories: size_bytes = 0
    """
    current_dir: Optional[str] = None

    for line in stream:
        line = line.rstrip("\r\n")

        if any(line.startswith(p) for p in SKIP_PREFIXES):
            continue
        if SKIP_LINE_RE.match(line):
            continue

        m = DIR_HEADER_RE.match(line)
        if m:
            current_dir = m.group("path").strip()
            continue

        if current_dir is None:
            continue

        m = ENTRY_RE.match(line)
        if not m:
            continue

        name = m.group("name").strip()
        if name in (".", ".."):
            continue

        tag = m.group("size_or_tag").strip()

        try:
            mtime = dt.datetime.strptime(
                f"{m.group('date')} {m.group('time')} {m.group('ampm')}",
                "%m/%d/%Y %I:%M %p",
            )
            mtime_ts = int(mtime.timestamp())
        except Exception:
            mtime_ts = 0

        is_dir = tag.startswith("<") and "DIR" in tag.upper()
        if is_dir:
            if not emit_dirs:
                continue
            size = 0
        else:
            try:
                size = int(tag.replace(",", ""))
            except Exception:
                # Unknown tag (e.g. <JUNCTION>) — ignore unless emitting dirs.
                if emit_dirs:
                    is_dir = True
                    size = 0
                else:
                    continue

        full_path = _safe_join_dir(current_dir, name)
        drive = _drive_letter(current_dir)
        relpath = _relpath_from_drive(full_path)
        root = _root1(relpath)

        yield (is_dir, drive, current_dir, name, size, mtime_ts, full_path, relpath, root)


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Speed pragmas; acceptable because this DB is rebuildable.
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
          full_path   TEXT PRIMARY KEY,
          drive       TEXT NOT NULL,
          dir_path    TEXT NOT NULL,
          name        TEXT NOT NULL,
          name_lc     TEXT NOT NULL,
          relpath     TEXT NOT NULL,
          relpath_lc  TEXT NOT NULL,
          root1       TEXT NOT NULL,
          ext         TEXT NOT NULL,
          size_bytes  INTEGER NOT NULL,
          mtime_ts    INTEGER NOT NULL
        );
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_name_size ON files(name_lc, size_bytes);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_rel_size  ON files(relpath_lc, size_bytes);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_drive      ON files(drive);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_root1      ON files(drive, root1);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_ext        ON files(ext);")
    conn.commit()
    return conn


def insert_files(conn: sqlite3.Connection, rows: Sequence[Tuple]) -> None:
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO files
        (full_path, drive, dir_path, name, name_lc, relpath, relpath_lc, root1, ext, size_bytes, mtime_ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def query_to_csv(conn: sqlite3.Connection, sql: str, params: Sequence, out_path: Path, header: Sequence[str]) -> None:
    cur = conn.cursor()
    cur.execute(sql, params)
    write_csv(out_path, header, cur.fetchall())


def generate_reports(
    conn: sqlite3.Connection,
    out_dir: Path,
    *,
    top_n: int,
    old_days: int,
    old_min_mb: int,
    dup_min_mb: int,
    max_dup_groups: int,
    max_paths_per_group: int,
    max_delete_candidates: int,
    min_candidate_mb: int,
    system_excludes: List[re.Pattern],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()

    # Totals per drive
    query_to_csv(
        conn,
        """
        SELECT drive,
               COUNT(*) AS file_count,
               SUM(size_bytes) AS total_bytes
        FROM files
        GROUP BY drive
        ORDER BY total_bytes DESC;
        """,
        (),
        out_dir / "drive_totals.csv",
        ["drive", "file_count", "total_bytes"],
    )

    # Top root folders by size per drive
    query_to_csv(
        conn,
        """
        SELECT drive,
               root1,
               COUNT(*) AS file_count,
               SUM(size_bytes) AS total_bytes
        FROM files
        GROUP BY drive, root1
        ORDER BY total_bytes DESC
        LIMIT ?;
        """,
        (max(1000, top_n * 10),),
        out_dir / "top_root_folders.csv",
        ["drive", "root1", "file_count", "total_bytes"],
    )

    # Extensions by size
    query_to_csv(
        conn,
        """
        SELECT ext,
               COUNT(*) AS file_count,
               SUM(size_bytes) AS total_bytes
        FROM files
        GROUP BY ext
        ORDER BY total_bytes DESC
        LIMIT ?;
        """,
        (max(200, top_n * 2),),
        out_dir / "extensions.csv",
        ["ext", "file_count", "total_bytes"],
    )

    # Largest files
    query_to_csv(
        conn,
        """
        SELECT size_bytes, mtime_ts, drive, full_path
        FROM files
        ORDER BY size_bytes DESC
        LIMIT ?;
        """,
        (top_n,),
        out_dir / "largest_files.csv",
        ["size_bytes", "mtime_ts", "drive", "full_path"],
    )

    # Old large files
    threshold_ts = int(time.time()) - int(old_days) * 86400
    min_old_bytes = int(old_min_mb) * 1024 * 1024
    query_to_csv(
        conn,
        """
        SELECT size_bytes, mtime_ts, drive, full_path
        FROM files
        WHERE mtime_ts > 0 AND mtime_ts < ? AND size_bytes >= ?
        ORDER BY size_bytes DESC
        LIMIT ?;
        """,
        (threshold_ts, min_old_bytes, top_n),
        out_dir / "old_large_files.csv",
        ["size_bytes", "mtime_ts", "drive", "full_path"],
    )

    # Duplicate groups by (name_lc, size)
    min_dup_bytes = int(dup_min_mb) * 1024 * 1024
    query_to_csv(
        conn,
        """
        SELECT name_lc,
               size_bytes,
               COUNT(*) AS copies,
               SUM(size_bytes) AS total_bytes
        FROM files
        WHERE size_bytes >= ?
        GROUP BY name_lc, size_bytes
        HAVING COUNT(*) > 1
        ORDER BY total_bytes DESC, copies DESC
        LIMIT ?;
        """,
        (min_dup_bytes, max_dup_groups),
        out_dir / "dup_name_size_groups.csv",
        ["name_lc", "size_bytes", "copies", "total_bytes"],
    )

    # Expand duplicate paths (name+size)
    cur.execute(
        """
        SELECT name_lc, size_bytes
        FROM (
            SELECT name_lc, size_bytes, COUNT(*) AS copies, SUM(size_bytes) AS total_bytes
            FROM files
            WHERE size_bytes >= ?
            GROUP BY name_lc, size_bytes
            HAVING COUNT(*) > 1
            ORDER BY total_bytes DESC, copies DESC
            LIMIT ?
        );
        """,
        (min_dup_bytes, max_dup_groups),
    )
    keys = cur.fetchall()

    def iter_dup_paths_by_name_size() -> Iterator[Tuple]:
        for name_lc, size_bytes in keys:
            cur.execute(
                """
                SELECT drive, mtime_ts, full_path
                FROM files
                WHERE name_lc = ? AND size_bytes = ?
                ORDER BY mtime_ts DESC, drive, full_path
                LIMIT ?;
                """,
                (name_lc, size_bytes, max_paths_per_group),
            )
            for drive, mtime_ts, full_path in cur.fetchall():
                yield (name_lc, size_bytes, drive, mtime_ts, full_path)

    write_csv(
        out_dir / "dup_name_size_paths.csv",
        ["name_lc", "size_bytes", "drive", "mtime_ts", "full_path"],
        iter_dup_paths_by_name_size(),
    )

    # Duplicate groups by (relpath_lc, size)
    query_to_csv(
        conn,
        """
        SELECT relpath_lc,
               size_bytes,
               COUNT(*) AS copies,
               SUM(size_bytes) AS total_bytes
        FROM files
        WHERE size_bytes >= ?
        GROUP BY relpath_lc, size_bytes
        HAVING COUNT(*) > 1
        ORDER BY total_bytes DESC, copies DESC
        LIMIT ?;
        """,
        (min_dup_bytes, max_dup_groups),
        out_dir / "dup_relpath_size_groups.csv",
        ["relpath_lc", "size_bytes", "copies", "total_bytes"],
    )

    # Expand duplicate paths (relpath+size)
    cur.execute(
        """
        SELECT relpath_lc, size_bytes
        FROM (
            SELECT relpath_lc, size_bytes, COUNT(*) AS copies, SUM(size_bytes) AS total_bytes
            FROM files
            WHERE size_bytes >= ?
            GROUP BY relpath_lc, size_bytes
            HAVING COUNT(*) > 1
            ORDER BY total_bytes DESC, copies DESC
            LIMIT ?
        );
        """,
        (min_dup_bytes, max_dup_groups),
    )
    rel_keys = cur.fetchall()

    def iter_dup_paths_by_relpath_size() -> Iterator[Tuple]:
        for relpath_lc, size_bytes in rel_keys:
            cur.execute(
                """
                SELECT drive, mtime_ts, full_path
                FROM files
                WHERE relpath_lc = ? AND size_bytes = ?
                ORDER BY mtime_ts DESC, drive, full_path
                LIMIT ?;
                """,
                (relpath_lc, size_bytes, max_paths_per_group),
            )
            for drive, mtime_ts, full_path in cur.fetchall():
                yield (relpath_lc, size_bytes, drive, mtime_ts, full_path)

    write_csv(
        out_dir / "dup_relpath_size_paths.csv",
        ["relpath_lc", "size_bytes", "drive", "mtime_ts", "full_path"],
        iter_dup_paths_by_relpath_size(),
    )

    # Build delete candidate list (review-only) in a streaming way.
    trash_path_re = re.compile("|".join(TRASHY_PATH_HINTS), re.IGNORECASE) if TRASHY_PATH_HINTS else None
    min_candidate_bytes = int(min_candidate_mb) * 1024 * 1024

    def is_system_excluded(p: str) -> bool:
        return any(rx.search(p) for rx in system_excludes)

    def consider(best: Dict[str, Tuple[int, str, int, int]], full_path: str, prio: int, reason: str, size: int, mtime_ts: int):
        # Keep the highest-priority reason per path; tie-break by size.
        prev = best.get(full_path)
        if prev is None:
            best[full_path] = (prio, reason, size, mtime_ts)
        else:
            prev_prio, prev_reason, prev_size, prev_mtime = prev
            if prio > prev_prio or (prio == prev_prio and size > prev_size):
                best[full_path] = (prio, reason, size, mtime_ts)

    def reason_and_prio(path: str, ext: str, size_bytes: int, mtime_ts: int) -> Optional[Tuple[str, int]]:
        if is_system_excluded(path):
            return None

        e = (ext or "").lower()
        p = path

        # Strong path hints.
        if trash_path_re and trash_path_re.search(p):
            if size_bytes < min_candidate_bytes:
                return None
            if e in ARCHIVE_EXTS or e in INSTALLER_EXTS:
                return ("in downloads/temp/cache + archive/installer", 2)
            if e in TRASHY_EXTS:
                return ("in downloads/temp/cache + log/tmp/dump", 2)
            if size_bytes == 0:
                return ("zero-byte file in downloads/temp/cache", 2)
            return ("in downloads/temp/cache (review)", 1)

        # File-type hints.
        if e in TRASHY_EXTS and size_bytes >= min_candidate_bytes:
            return ("log/tmp/dump extension (review)", 1)
        if e in ARCHIVE_EXTS and size_bytes >= max(min_candidate_bytes, 50 * 1024 * 1024):
            return ("large archive (review)", 1)
        if e in INSTALLER_EXTS and size_bytes >= max(min_candidate_bytes, 50 * 1024 * 1024):
            return ("large installer (review)", 1)

        return None

    best: Dict[str, Tuple[int, str, int, int]] = {}

    # Content-based candidates (streaming)
    cur.execute("SELECT full_path, ext, size_bytes, mtime_ts FROM files;")
    while True:
        chunk = cur.fetchmany(50000)
        if not chunk:
            break
        for full_path, ext, size_bytes, mtime_ts in chunk:
            rp = reason_and_prio(full_path, ext, int(size_bytes), int(mtime_ts))
            if rp is None:
                continue
            reason, prio = rp
            consider(best, full_path, prio, reason, int(size_bytes), int(mtime_ts))

    # Duplicate-based candidates: mark all but newest as "extras" (capped per group)
    cur2 = conn.cursor()
    cur2.execute(
        """
        SELECT name_lc, size_bytes
        FROM files
        WHERE size_bytes >= ?
        GROUP BY name_lc, size_bytes
        HAVING COUNT(*) > 1;
        """,
        (min_dup_bytes,),
    )
    dup_groups = cur2.fetchall()
    for name_lc, size_bytes in dup_groups:
        cur2.execute(
            """
            SELECT full_path, mtime_ts
            FROM files
            WHERE name_lc = ? AND size_bytes = ?
            ORDER BY mtime_ts DESC, full_path
            LIMIT ?;
            """,
            (name_lc, size_bytes, max_paths_per_group),
        )
        paths = cur2.fetchall()
        if len(paths) <= 1:
            continue
        keep_path = paths[0][0]
        for full_path, mtime_ts in paths[1:]:
            if is_system_excluded(full_path):
                continue
            consider(
                best,
                full_path,
                3,
                f"duplicate name+size (keep: {keep_path})",
                int(size_bytes),
                int(mtime_ts),
            )

    # Write delete candidates, largest first (cap to max_delete_candidates)
    rows = [
        (reason, size, mtime_ts, full_path)
        for full_path, (prio, reason, size, mtime_ts) in best.items()
    ]
    rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
    if max_delete_candidates > 0:
        rows = rows[:max_delete_candidates]

    write_csv(
        out_dir / "delete_candidates.csv",
        ["reason", "size_bytes", "mtime_ts", "full_path"],
        rows,
    )

    # Write summary.txt
    summary_lines: List[str] = []
    summary_lines.append("Drive audit summary")
    summary_lines.append("===================")
    summary_lines.append("")

    cur3 = conn.cursor()
    cur3.execute("SELECT drive, COUNT(*), SUM(size_bytes) FROM files GROUP BY drive ORDER BY SUM(size_bytes) DESC;")
    summary_lines.append("Totals by drive:")
    for drive, cnt, total in cur3.fetchall():
        summary_lines.append(f"  {drive}: {cnt:,} files, {total or 0:,} bytes")
    summary_lines.append("")

    summary_lines.append(f"Largest files: see largest_files.csv (top {top_n}).")
    summary_lines.append(f"Old large files: see old_large_files.csv (>{old_min_mb} MB and older than {old_days} days).")
    summary_lines.append(f"Duplicate candidates: see dup_name_size_*.csv and dup_relpath_size_*.csv (min {dup_min_mb} MB).")
    summary_lines.append(f"Deletion review list: delete_candidates.csv (max {max_delete_candidates:,} rows, min {min_candidate_mb} MB; system paths excluded).")
    summary_lines.append("")

    # Root folder overlap across drives (organization signal)
    cur3.execute(
        """
        SELECT root1, COUNT(DISTINCT drive) AS drives, SUM(size_bytes) AS total_bytes, COUNT(*) AS files
        FROM files
        WHERE root1 != ''
        GROUP BY root1
        HAVING drives > 1
        ORDER BY total_bytes DESC
        LIMIT 50;
        """
    )
    overlap = cur3.fetchall()
    if overlap:
        summary_lines.append("Top root folders present on multiple drives (consolidation candidates):")
        for root1, drives, total_bytes, files in overlap:
            summary_lines.append(f"  {root1}: on {drives} drives, {files:,} files, {total_bytes:,} bytes")
        summary_lines.append("")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit Windows dir.txt dumps for cleanup and organization.")
    ap.add_argument("--input", required=True, help="Path to drives.zip OR a dir dump .txt OR a folder of .txt files")
    ap.add_argument("--out", default="audit_out", help="Output folder for reports")
    ap.add_argument("--db", default="", help="Path to sqlite db (default: <out>/audit.sqlite)")
    ap.add_argument("--top", type=int, default=200, help="Rows for top-N lists (largest/old-large)")
    ap.add_argument("--old-days", type=int, default=730, help="Old threshold in days (for old_large_files.csv)")
    ap.add_argument("--old-min-mb", type=int, default=200, help="Minimum size (MB) for old_large_files.csv")
    ap.add_argument("--dup-min-mb", type=int, default=50, help="Minimum size (MB) for duplicates reports")
    ap.add_argument("--max-dup-groups", type=int, default=5000, help="Max duplicate groups to output")
    ap.add_argument("--max-paths-per-group", type=int, default=50, help="Max paths per duplicate group expansion")
    ap.add_argument("--max-delete-candidates", type=int, default=20000, help="Max rows for delete_candidates.csv (0 = no cap)")
    ap.add_argument("--min-candidate-mb", type=int, default=10, help="Minimum size (MB) for non-duplicate delete candidates")
    ap.add_argument("--no-system-excludes", action="store_true", help="Include system paths on C: in delete candidates")
    args = ap.parse_args(argv)

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db).expanduser() if args.db else (out_dir / "audit.sqlite")

    system_excludes: List[re.Pattern] = []
    if not args.no_system_excludes:
        system_excludes = [re.compile(p, re.IGNORECASE) for p in DEFAULT_SYSTEM_EXCLUDES]

    conn = init_db(db_path)

    zf: Optional[zipfile.ZipFile] = None
    try:
        # If input is zip, keep it open while iterating its members.
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            zf = zipfile.ZipFile(input_path, "r")

            def sources_from_zip(z: zipfile.ZipFile):
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    if not info.filename.lower().endswith(".txt"):
                        continue
                    raw = z.open(info, "r")
                    stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
                    yield info.filename, stream

            sources = sources_from_zip(zf)
        elif input_path.is_dir():
            sources = iter_text_sources(input_path)
        else:
            sources = iter_text_sources(input_path)

        batch: List[Tuple] = []
        batch_size = 5000

        for src_name, stream in sources:
            with stream:
                for is_dir, drive, dir_path, name, size_bytes, mtime_ts, full_path, relpath, root1 in parse_dir_dump(
                    src_name, stream, emit_dirs=False
                ):
                    row = (
                        full_path,
                        drive,
                        dir_path,
                        name,
                        name.lower(),
                        relpath,
                        relpath.lower(),
                        root1,
                        _ext_of(name),
                        int(size_bytes),
                        int(mtime_ts),
                    )
                    batch.append(row)
                    if len(batch) >= batch_size:
                        insert_files(conn, batch)
                        conn.commit()
                        batch.clear()

        if batch:
            insert_files(conn, batch)
            conn.commit()

        generate_reports(
            conn,
            out_dir,
            top_n=args.top,
            old_days=args.old_days,
            old_min_mb=args.old_min_mb,
            dup_min_mb=args.dup_min_mb,
            max_dup_groups=args.max_dup_groups,
            max_paths_per_group=args.max_paths_per_group,
            max_delete_candidates=args.max_delete_candidates,
            min_candidate_mb=args.min_candidate_mb,
            system_excludes=system_excludes,
        )

    finally:
        try:
            conn.close()
        except Exception:
            pass
        if zf is not None:
            try:
                zf.close()
            except Exception:
                pass

    print(f"Wrote reports to: {out_dir}")
    print(f"SQLite database: {db_path}")
    print("Start with: summary.txt, delete_candidates.csv, largest_files.csv, dup_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
