"""Collect unique optimizer sweep geometries into one JSON file.

The script scans sweep result folders for ``data/layout.json`` files, de-
duplicates geometries by their patch intervals, and writes a catalog with the
first case name plus all duplicate source cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def rounded_tuple(values, decimals: int) -> tuple[float, ...]:
    return tuple(round(float(v), decimals) for v in values)


def geometry_key(layout: dict[str, Any], case: dict[str, Any] | None, decimals: int) -> tuple:
    """Return a stable key for duplicate detection.

    Patch intervals are the geometry we care about for the optimizer. Include
    beam length when it is available so identical patches on different beams do
    not collapse together.
    """
    x_l = rounded_tuple(layout.get("xL", ()), decimals)
    x_r = rounded_tuple(layout.get("xR", ()), decimals)

    beam_length = None
    if case is not None and case.get("L") is not None:
        beam_length = float(case["L"])
    elif layout.get("x_starts") and layout.get("tip_substrate") is not None:
        beam_length = float(layout["x_starts"][-1]) + float(layout["tip_substrate"])

    rounded_l = None if beam_length is None else round(beam_length, decimals)
    return (rounded_l, x_l, x_r)


def source_name(root: Path, layout_path: Path) -> str:
    case_dir = layout_path.parent.parent
    try:
        rel = case_dir.relative_to(root)
        return "__".join(rel.parts)
    except ValueError:
        return case_dir.name


def read_optional_json(path: Path) -> Any | None:
    return read_json(path) if path.exists() else None


def compact_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    keys = (
        "Np",
        "objective",
        "mode_numbers",
        "mode_label",
        "phase_mode",
        "output",
        "score",
        "freq_hz",
        "best_z_mm",
        "xL_mm",
        "xR_mm",
    )
    return {key: summary[key] for key in keys if key in summary}


def make_source_record(root: Path, layout_path: Path) -> dict[str, Any]:
    data_dir = layout_path.parent
    case_dir = data_dir.parent
    case = read_optional_json(data_dir / "case.json")
    summary = read_optional_json(data_dir / "summary.json")
    return {
        "name": source_name(root, layout_path),
        "case_dir": str(case_dir.relative_to(root)),
        "layout_json": str(layout_path.relative_to(root)),
        "case": case,
        "summary": compact_summary(summary),
    }


def collect_unique_geometries(root: Path, decimals: int) -> dict[str, Any]:
    layout_paths = sorted(root.rglob("layout.json"))
    geometries: dict[tuple, dict[str, Any]] = {}

    for layout_path in layout_paths:
        data_dir = layout_path.parent
        layout = read_json(layout_path)
        case = read_optional_json(data_dir / "case.json")
        key = geometry_key(layout, case, decimals)
        source = make_source_record(root, layout_path)

        if key not in geometries:
            geometries[key] = {
                "name": source["name"],
                "geometry": layout,
                "sources": [source],
            }
        else:
            geometries[key]["sources"].append(source)

    unique = []
    for item in geometries.values():
        duplicate_names = [source["name"] for source in item["sources"][1:]]
        unique.append(
            {
                "name": item["name"],
                "duplicate_count": len(item["sources"]) - 1,
                "duplicate_names": duplicate_names,
                "geometry": item["geometry"],
                "sources": item["sources"],
            }
        )

    return {
        "root": str(root),
        "dedupe_decimals": decimals,
        "layout_file_count": len(layout_paths),
        "unique_geometry_count": len(unique),
        "duplicate_geometry_count": len(layout_paths) - len(unique),
        "geometries": unique,
    }


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent / "optimizer_sweep_results"
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "optimizer_sweep_results",
        help="Root folder containing sweep result directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <root>/unique_geometries.json.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=12,
        help="Decimal places used when comparing floating-point geometry values.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve() if args.output is not None else root / "unique_geometries.json"
    catalog = collect_unique_geometries(root, args.decimals)
    write_json(output, catalog)
    print(
        f"Wrote {catalog['unique_geometry_count']} unique geometries "
        f"from {catalog['layout_file_count']} layout files to {output}"
    )


if __name__ == "__main__":
    main()
