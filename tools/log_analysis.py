"""
log_analysis.py

Utility for extracting transport-aware learning diagnostics from NUPCA3 log files,
then emitting aligned summaries and plots. Use this to prove that transported priors
keep residuals alive before the learner even runs, that periphery diagnostics stay
stable, and that non-zero candidate counts follow each transport-driven shift.

Usage:

    python tools/log_analysis.py path/to/run.log

The script creates `analysis_data.csv`, `transport_mae.png`, `unobs_mae.png`, and
`candidates.png` under the `analysis_outputs/` directory by default, and prints a
step-aligned table and aggregated stats to stdout.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterator, List

import matplotlib.pyplot as plt


def _maybe_float(raw: str) -> float:
    if raw.lower() == "nan":
        return math.nan
    return float(raw)


def _extract_step(line: str) -> int | None:
    m = re.search(r"\bstep=(\d+)", line)
    if m:
        return int(m.group(1))
    m = re.search(r"\bt\s*=\s*(-?\d+)", line)
    if m:
        return int(m.group(1))
    return None


def _extract_float(line: str, key: str) -> float | None:
    match = re.search(fr"{key}=([0-9.+-eE]+|nan)", line)
    if match:
        return _maybe_float(match.group(1))
    return None


def _extract_bool(line: str, key: str) -> bool | None:
    match = re.search(fr"{key}=(True|False)", line)
    if match:
        return match.group(1) == "True"
    return None


def _extract_token(line: str, key: str) -> str | None:
    match = re.search(fr"{key}=([A-Za-z0-9_]+)", line)
    if match:
        return match.group(1)
    return None


def _next_non_empty_line(lines_iter: Iterator[str]) -> str | None:
    for raw_line in lines_iter:
        line = raw_line.rstrip("\n")
        if line.strip():
            return line
    return None


def _collect_grid_rows(
    first_line: str,
    label: str,
    side: int,
    lines_iter: Iterator[str],
) -> List[List[str]]:
    rows: List[List[str]] = []
    stripped = first_line.strip()
    if not stripped:
        return rows
    tokens = stripped.split()
    row_tokens = tokens[1:] if tokens and tokens[0] == label else tokens
    row = row_tokens[:side]
    if len(row) < side:
        row.extend([""] * (side - len(row)))
    rows.append(row)
    for _ in range(side - 1):
        next_line = _next_non_empty_line(lines_iter)
        if next_line is None:
            break
        tokens = next_line.strip().split()
        row = tokens[:side]
        if len(row) < side:
            row.extend([""] * (side - len(row)))
        rows.append(row)
    return rows


def _handle_visual_block(
    step: int,
    lines_iter: Iterator[str],
    data: Dict[int, Dict[str, float]],
) -> None:
    env_line = _next_non_empty_line(lines_iter)
    if env_line is None:
        return
    env_tokens = env_line.strip().split()
    if not env_tokens or env_tokens[0] != "ENV":
        return
    side = len(env_tokens) - 1
    if side <= 0:
        return
    env_rows = _collect_grid_rows(env_line, "ENV", side, lines_iter)
    if len(env_rows) != side:
        return
    obs_line = _next_non_empty_line(lines_iter)
    if obs_line is None:
        return
    _collect_grid_rows(obs_line, "OBS", side, lines_iter)
    prev_line = _next_non_empty_line(lines_iter)
    if prev_line is None:
        return
    _collect_grid_rows(prev_line, "PREV", side, lines_iter)
    pred_line = _next_non_empty_line(lines_iter)
    if pred_line is None:
        return
    pred_rows = _collect_grid_rows(pred_line, "PRED", side, lines_iter)
    if len(pred_rows) != side:
        return
    diff_line = _next_non_empty_line(lines_iter)
    if diff_line is None:
        return
    diff_rows = _collect_grid_rows(diff_line, "DIFF", side, lines_iter)
    if len(diff_rows) != side:
        return

    env_flat = [token for row in env_rows for token in row]
    pred_flat = [token for row in pred_rows for token in row]
    diff_tokens = [token for row in diff_rows for token in row]
    ascii_env_pred_equal = env_flat == pred_flat
    ascii_diff_clear = bool(diff_tokens and all(token == ".." for token in diff_tokens))
    row = data.setdefault(step, {})
    row["ascii_visual_inspected"] = True
    row["ascii_env_pred_equal"] = ascii_env_pred_equal
    row["ascii_diff_clear"] = ascii_diff_clear


def _parse_log(path: str) -> Dict[int, Dict[str, float]]:
    data: Dict[int, Dict[str, float]] = defaultdict(dict)
    last_step: int | None = None

    transport_check_re = re.compile(
        r"\[transport check\].*?step=(\d+).*?true_delta=\((-?\d+),\s*(-?\d+)\).*?"
        r"transport_delta=\((-?\d+),\s*(-?\d+)\).*?match=(True|False)"
    )
    transport_diag_periph_re = re.compile(
        r"\[transport diag\] step=(\d+).*?periph_dims_missing_count=(\d+)"
    )
    learning_info_re = re.compile(r"learning_info .*?candidates=(\d+).*?clamped=(\d+)")
    permit_summary_re = re.compile(r"permit_param_summary .*?cand=(\d+)")
    periph_missing_count_re = re.compile(r"periph_dims_missing_count=(-?\d+)")
    candidate_count_re = re.compile(r"candidate_count=(-?\d+)")
    clamped_count_re = re.compile(r"clamped_count=(-?\d+)")
    diff_count_re = re.compile(r"\[diff check\].*?step=(\d+).*?diff_count=(-?\d+)")
    env_true_delta_re = re.compile(
        r"\[env\b.*?step=(\d+).*?true_delta=\((-?\d+),\s*(-?\d+)\)"
    )
    visual_re = re.compile(r"\[VISUAL step=(\d+)")

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines_iter = iter(fh)
        for raw_line in lines_iter:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            step_hint = _extract_step(line)
            if step_hint is not None:
                last_step = step_hint

            visual_match = visual_re.search(stripped)
            if visual_match:
                step = int(visual_match.group(1))
                _handle_visual_block(step, lines_iter, data)
                continue

            match = transport_check_re.search(line)
            if match:
                step = int(match.group(1))
                data[step]["true_delta_dx"] = float(match.group(2))
                data[step]["true_delta_dy"] = float(match.group(3))
                data[step]["transport_delta_dx"] = float(match.group(4))
                data[step]["transport_delta_dy"] = float(match.group(5))
                data[step]["transport_match"] = float(match.group(6) == "True")
                continue

            if "A13 transport_diag" in line:
                step = _extract_step(line)
                if step is None:
                    continue
                delta_match = re.search(r"delta=\((-?\d+),\s*(-?\d+)\)", line)
                if delta_match:
                    data[step]["transport_delta_dx"] = float(delta_match.group(1))
                    data[step]["transport_delta_dy"] = float(delta_match.group(2))
                for key in (
                    "transport_mae_pre",
                    "transport_mae_post",
                    "mae_pos_prior",
                    "mae_pos_prior_unobs",
                    "mae_pos_unobs_pre",
                    "mae_pos_unobs_post",
                ):
                    value = _extract_float(line, key)
                    if value is not None:
                        data[step][key] = value
                transport_norm = _extract_float(line, "trans_norm")
                if transport_norm is not None:
                    data[step]["transport_norm"] = transport_norm
                source = _extract_token(line, "transport_source")
                if source:
                    data[step]["transport_source"] = source
                periph_sel = _extract_bool(line, "periph_selected")
                if periph_sel is not None:
                    data[step]["periph_selected"] = float(periph_sel)
                periph_missing_match = periph_missing_count_re.search(line)
                if periph_missing_match:
                    data[step]["periph_dims_missing"] = int(periph_missing_match.group(1))
                continue

            env_match = env_true_delta_re.search(line)
            if env_match:
                step = int(env_match.group(1))
                data[step]["true_delta_dx"] = float(env_match.group(2))
                data[step]["true_delta_dy"] = float(env_match.group(3))
                continue

            periph_match = transport_diag_periph_re.search(line)
            if periph_match:
                step = int(periph_match.group(1))
                data[step]["periph_dims_missing"] = int(periph_match.group(2))
                continue

            learn_match = learning_info_re.search(line)
            if learn_match:
                step = step_hint if step_hint is not None else last_step
                if step is None:
                    continue
                data[step]["candidate_count"] = int(learn_match.group(1))
                data[step]["clamped_count"] = int(learn_match.group(2))
                continue

            permit_match = permit_summary_re.search(line)
            if permit_match:
                step = step_hint if step_hint is not None else last_step
                if step is None:
                    continue
                data[step].setdefault("candidate_count", int(permit_match.group(1)))
            diff_match = diff_count_re.search(line)
            if diff_match:
                step = int(diff_match.group(1))
                data[step]["diff_count"] = int(diff_match.group(2))
            step = step_hint if step_hint is not None else last_step
            if step is not None:
                periph_missing_count_match = periph_missing_count_re.search(line)
                if periph_missing_count_match:
                    data[step]["periph_dims_missing"] = int(periph_missing_count_match.group(1))
                candidate_match = candidate_count_re.search(line)
                if candidate_match:
                    data[step]["candidate_count"] = int(candidate_match.group(1))
                clamped_match = clamped_count_re.search(line)
                if clamped_match:
                    data[step]["clamped_count"] = int(clamped_match.group(1))
    return data


def _write_csv(rows: List[Dict[str, float]], out_dir: str) -> str:
    fieldnames = [
        "step",
        "transport_delta_dx",
        "transport_delta_dy",
        "true_delta_dx",
        "true_delta_dy",
        "transport_delta_nonzero",
        "true_delta_nonzero",
        "delta_match",
        "transport_mae_pre",
        "transport_mae_post",
        "transport_norm",
        "mae_pos_prior",
        "mae_pos_prior_unobs",
        "mae_pos_unobs_pre",
        "mae_pos_unobs_post",
        "periph_dims_missing",
        "candidate_count",
        "clamped_count",
        "transport_source",
        "periph_selected",
        "diff_count",
    ]
    path = os.path.join(out_dir, "analysis_data.csv")
    def _normalize(value: float | str | int | None) -> str | float | int:
        if isinstance(value, float) and math.isnan(value):
            return ""
        return value if value is not None else ""

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _normalize(row.get(key, "")) for key in fieldnames})
    return path


def _plot_metric(rows: List[Dict[str, float]], out_dir: str, y_keys: List[str], title: str, filename: str) -> str:
    steps = [row["step"] for row in rows]
    plt.figure(figsize=(10, 4.5))
    for key in y_keys:
        values = [row.get(key, math.nan) for row in rows]
        plt.plot(steps, values, label=key)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _print_summary(rows: List[Dict[str, float]]) -> None:
    print("\nParsed steps: {}".format(len(rows)))
    valid_drop = [
        row["transport_mae_pre"] - row["transport_mae_post"]
        for row in rows
        if "transport_mae_pre" in row and "transport_mae_post" in row
        and not math.isnan(row["transport_mae_pre"])
        and not math.isnan(row["transport_mae_post"])
    ]
    if valid_drop:
        avg_drop = sum(valid_drop) / len(valid_drop)
        best_step, best_drop = max(
            ((row["step"], row["transport_mae_pre"] - row["transport_mae_post"]) for row in rows
             if "transport_mae_pre" in row and "transport_mae_post" in row),
            key=lambda pair: pair[1],
        )
        print(f"Transport MAE drop avg={avg_drop:.4f}, peak drop={best_drop:.4f} at step {best_step}")
    unobs_drop = [
        row["mae_pos_unobs_pre"] - row["mae_pos_unobs_post"]
        for row in rows
        if "mae_pos_unobs_pre" in row and "mae_pos_unobs_post" in row
        and not math.isnan(row["mae_pos_unobs_pre"])
        and not math.isnan(row["mae_pos_unobs_post"])
    ]
    if unobs_drop:
        avg_unobs_drop = sum(unobs_drop) / len(unobs_drop)
        print(f"Positive unobserved MAE drop avg={avg_unobs_drop:.4f}")
    candidates = [row["candidate_count"] for row in rows if "candidate_count" in row]
    if candidates:
        print(
            "Candidate counts: min={min:d} max={max:d} mean={mean:.2f}".format(
                min=min(candidates),
                max=max(candidates),
                mean=sum(candidates) / len(candidates),
            )
        )
    missing = [row.get("periph_dims_missing") for row in rows if "periph_dims_missing" in row]
    if missing:
        print("Periphery missing dims: majority={} steps".format(sum(1 for v in missing if v > 0)))
    sources = [row["transport_source"] for row in rows if "transport_source" in row]
    if sources:
        counts = Counter(sources)
        stats = ", ".join(f"{src}={cnt}" for src, cnt in counts.items())
        print(f"Transport source: {stats}")
    periph_flags = [row.get("periph_selected") for row in rows if "periph_selected" in row]
    if periph_flags:
        sel_steps = sum(1 for v in periph_flags if v)
        print(f"Periph selected: {sel_steps}/{len(periph_flags)} steps")
    diff_counts = [row["diff_count"] for row in rows if "diff_count" in row]
    if diff_counts:
        perfect = sum(1 for v in diff_counts if v == 0)
        print(
            f"Prediction match rate: {perfect}/{len(diff_counts)} "
            f"({perfect/len(diff_counts)*100:.1f}%)"
        )
    nonzero_true = sum(1 for row in rows if row.get("true_delta_nonzero") == 1.0)
    match_nonzero = sum(
        1
        for row in rows
        if row.get("true_delta_nonzero") == 1.0 and row.get("delta_match") == 1.0
    )
    diff_zero_rows = [row for row in rows if row.get("diff_count") == 0]
    delta_matches_with_diff = sum(
        1
        for row in diff_zero_rows
        if row.get("delta_match") == 1.0
    )
    trans_nonzero = sum(1 for row in rows if row.get("transport_delta_nonzero") == 1.0)
    if diff_zero_rows:
        print(
            f"Delta match rate (diff==0 steps): {delta_matches_with_diff}/{len(diff_zero_rows)} "
            f"({delta_matches_with_diff/len(diff_zero_rows)*100:.1f}%)"
        )
    ascii_rows = [row for row in rows if row.get("ascii_visual_inspected")]
    if ascii_rows:
        ascii_success = sum(
            1
            for row in ascii_rows
            if row.get("ascii_env_pred_equal")
            and row.get("ascii_diff_clear")
            and row.get("diff_count") == 0
        )
        print(
            f"ASCII env/pred match rate: {ascii_success}/{len(ascii_rows)} "
            f"({ascii_success/len(ascii_rows)*100:.1f}%)"
        )
    if nonzero_true:
        print(
            f"Match rate when true motion ≠ 0: {match_nonzero}/{nonzero_true} "
            f"({match_nonzero/nonzero_true*100:.1f}%)"
        )
    if trans_nonzero:
        print(f"Transport nonzero steps: {trans_nonzero} / {len(rows)}")
    print("\nSample row alignment (first 10):")
    header = [
        "step",
        "transport_delta_dx",
        "transport_mae_pre",
        "transport_mae_post",
        "mae_pos_prior",
        "mae_pos_prior_unobs",
        "mae_pos_unobs_pre",
        "mae_pos_unobs_post",
        "transport_source",
        "periph_selected",
        "candidate_count",
        "periph_dims_missing",
    ]
    print("\t".join(header))
    for row in rows[:10]:
        print("\t".join(str(row.get(key, "")) for key in header))


def main() -> None:
    parser = argparse.ArgumentParser(description="Make charts from NUPCA3 transport diagnostics.")
    parser.add_argument("logfile", help="Path to the agent output log file.")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Directory to dump CSV/PNG outputs (created if missing).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    parsed = _parse_log(args.logfile)
    if not parsed:
        raise SystemExit("No transport diagnostics found in log!")

    rows = []
    for step in sorted(parsed.keys()):
        row = {"step": step, **parsed[step]}
        dx = row.get("transport_delta_dx")
        dy = row.get("transport_delta_dy")
        true_dx = row.get("true_delta_dx")
        true_dy = row.get("true_delta_dy")
        if dx is not None and dy is not None:
            row["transport_delta_nonzero"] = float(dx != 0 or dy != 0)
        if true_dx is not None and true_dy is not None:
            row["true_delta_nonzero"] = float(true_dx != 0 or true_dy != 0)
        if (
            dx is not None
            and dy is not None
            and true_dx is not None
            and true_dy is not None
        ):
            row["delta_match"] = float(dx == true_dx and dy == true_dy)
        rows.append(row)

    csv_path = _write_csv(rows, args.output_dir)
    path1 = _plot_metric(
        rows,
        args.output_dir,
        ["transport_mae_pre", "transport_mae_post"],
        "Transport MAE before/after clamp",
        "transport_mae.png",
    )
    path2 = _plot_metric(
        rows,
        args.output_dir,
        ["mae_pos_unobs_pre", "mae_pos_unobs_post"],
        "Positive unobserved MAE before/after transport",
        "unobs_mae.png",
    )
    path3 = _plot_metric(
        rows,
        args.output_dir,
        ["candidate_count"],
        "Learning candidate count per step",
        "candidates.png",
    )

    _print_summary(rows)
    print("\nGenerated outputs:")
    print(f"- data table: {csv_path}")
    print(f"- transport MAE chart: {path1}")
    print(f"- unobserved MAE chart: {path2}")
    print(f"- candidate chart: {path3}")


if __name__ == "__main__":
    main()
