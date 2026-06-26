"""Summarize geometry-study amplitudes by patch count.

Reads ``geometry_performance_rows.csv`` and creates one wide table per patch
count. Each output row is one distinct geometry with tip and RMS amplitude
columns for modes 1, 2, and 3.

Example
-------
python Modeling/tasks/Fish/summarize_geometry_amplitudes.py
python Modeling/tasks/Fish/summarize_geometry_amplitudes.py --patch-count 2 3
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT = Path(__file__).resolve().parent / "geometry_performance_results" / "geometry_performance_rows.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "geometry_performance_results"
DEFAULT_MODES = (1, 2, 3)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def available_patch_counts(rows: Iterable[dict[str, str]]) -> list[int]:
    counts = {int(row["n_patches"]) for row in rows if row.get("n_patches")}
    return sorted(counts)


def wide_amplitude_rows(
    rows: Iterable[dict[str, str]],
    *,
    patch_count: int,
    modes: Iterable[int] = DEFAULT_MODES,
) -> list[dict[str, str]]:
    """Return one wide amplitude row per geometry for one patch count."""
    modes = tuple(int(mode) for mode in modes)
    by_geometry: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: {"tip": {}, "rms": {}}
    )

    for row in rows:
        if int(row["n_patches"]) != int(patch_count):
            continue
        mode = int(row["mode_number"])
        if mode not in modes:
            continue
        geometry = row["geometry"]
        tip = float(row["tip"])
        rms = float(row["rms"])

        # If duplicate rows exist for the same geometry/mode, keep the largest
        # value for each metric independently.
        previous_tip = by_geometry[geometry]["tip"].get(mode)
        previous_rms = by_geometry[geometry]["rms"].get(mode)
        if previous_tip is None or tip > previous_tip:
            by_geometry[geometry]["tip"][mode] = tip
        if previous_rms is None or rms > previous_rms:
            by_geometry[geometry]["rms"][mode] = rms

    output_rows = []
    for geometry in sorted(by_geometry):
        values = by_geometry[geometry]
        output_rows.append(
            {
                "name": geometry,
                "mode_1_tip_amplitude": _format_float(values["tip"].get(1)),
                "mode_2_tip_amplitude": _format_float(values["tip"].get(2)),
                "mode_3_tip_amplitude": _format_float(values["tip"].get(3)),
                "mode_1_rms_amplitude": _format_float(values["rms"].get(1)),
                "mode_2_rms_amplitude": _format_float(values["rms"].get(2)),
                "mode_3_rms_amplitude": _format_float(values["rms"].get(3)),
            }
        )
    return output_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "mode_1_tip_amplitude",
        "mode_2_tip_amplitude",
        "mode_3_tip_amplitude",
        "mode_1_rms_amplitude",
        "mode_2_rms_amplitude",
        "mode_3_rms_amplitude",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, str]], *, title: str, limit: int | None = None) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("No rows")
        return

    display_rows = rows if limit is None else rows[:limit]
    columns = [
        "name",
        "mode_1_tip_amplitude",
        "mode_2_tip_amplitude",
        "mode_3_tip_amplitude",
        "mode_1_rms_amplitude",
        "mode_2_rms_amplitude",
        "mode_3_rms_amplitude",
    ]
    widths = {
        col: max(len(col), *(len(str(row[col])) for row in display_rows))
        for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for row in display_rows:
        print("  ".join(str(row[col]).ljust(widths[col]) for col in columns))
    if limit is not None and len(rows) > limit:
        print(f"... {len(rows) - limit} more rows")


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to geometry_performance_rows.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated CSV tables")
    parser.add_argument("--patch-count", type=int, nargs="*", default=None, help="Patch counts to summarize. Defaults to all counts in the input.")
    parser.add_argument("--print-limit", type=int, default=20, help="Max rows printed per patch-count table. Use 0 for no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    patch_counts = args.patch_count or available_patch_counts(rows)
    print_limit = None if args.print_limit == 0 else int(args.print_limit)

    for patch_count in patch_counts:
        table = wide_amplitude_rows(rows, patch_count=patch_count)
        output_path = args.output_dir / f"geometry_amplitudes_Np{patch_count}.csv"
        write_csv(output_path, table)
        print_table(table, title=f"Np = {patch_count} geometry amplitudes", limit=print_limit)
        print(f"Wrote {len(table)} rows to {output_path}")


if __name__ == "__main__":
    main()
