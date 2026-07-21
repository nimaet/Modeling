#!/usr/bin/env python
"""Apply a Dropbox offline/online-only policy by file type.

Default policy:
  - Keep .py and .ipynb files available offline.
  - Make all other files online-only.

By default this applies the Dropbox attribute changes.
Pass --dry-run to preview actions without changing attributes.
When no root is provided, the script applies to the directory containing
this script, so you can drop/copy it into any Dropbox folder and run it there.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


OFFLINE_EXTENSIONS = {".py", ".ipynb", ".png"}
SKIP_DIR_NAMES = {
	".git",
	".hg",
	".svn",
	".mypy_cache",
	".pytest_cache",
	".ruff_cache",
	"__pycache__",
	"node_modules",
}


def is_hidden_or_system(path: Path) -> bool:
	"""Return True for hidden/system files on Windows."""
	if os.name != "nt":
		return path.name.startswith(".")

	try:
		attrs = path.stat().st_file_attributes
	except OSError:
		return False

	hidden = getattr(os, "FILE_ATTRIBUTE_HIDDEN", 0x2)
	system = getattr(os, "FILE_ATTRIBUTE_SYSTEM", 0x4)
	return bool(attrs & (hidden | system))


def normalize_extensions(values: list[str]) -> set[str]:
	exts: set[str] = set()
	for value in values:
		value = value.strip().casefold()
		if not value:
			continue
		if not value.startswith("."):
			value = "." + value
		exts.add(value)
	return exts


def has_offline_extension(path: Path, offline_exts: set[str]) -> bool:
	"""Return True when the file's final or compound suffix is configured offline."""
	suffixes = [suffix.casefold() for suffix in path.suffixes]
	if not suffixes:
		return False

	if suffixes[-1] in offline_exts:
		return True

	for start in range(len(suffixes) - 1):
		compound_suffix = "".join(suffixes[start:])
		if compound_suffix in offline_exts:
			return True

	return False


def iter_files(root: Path, recursive: bool, include_hidden: bool):
	if recursive:
		for current_root, dir_names, file_names in os.walk(root):
			current = Path(current_root)
			kept_dirs = []
			for dir_name in dir_names:
				child = current / dir_name
				if dir_name in SKIP_DIR_NAMES:
					continue
				if not include_hidden and is_hidden_or_system(child):
					continue
				kept_dirs.append(dir_name)
			dir_names[:] = kept_dirs

			for file_name in file_names:
				path = current / file_name
				if include_hidden or not is_hidden_or_system(path):
					yield path
	else:
		for path in root.iterdir():
			if path.is_file() and (include_hidden or not is_hidden_or_system(path)):
				yield path


def run_attrib(path: Path, make_offline: bool) -> subprocess.CompletedProcess[str]:
	if make_offline:
		command = ["attrib", "+P", "-U", str(path)]
	else:
		command = ["attrib", "-P", "+U", str(path)]

	return subprocess.run(command, text=True, capture_output=True, check=False)


def hydrate_file(path: Path) -> None:
	"""Read a file to encourage Dropbox to download its contents."""
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			if not chunk:
				break


def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Set a Dropbox folder policy: selected file extensions available "
			"offline, all other files online-only."
		)
	)
	parser.add_argument(
		"root",
		nargs="?",
		type=Path,
		default=Path(__file__).resolve().parent,
		help="Folder to process. Defaults to the folder containing this script.",
	)
	parser.add_argument(
		"--offline-ext",
		nargs="+",
		default=sorted(OFFLINE_EXTENSIONS),
		help="Extensions to keep available offline. Default: .ipynb .py",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Preview actions without changing attributes.",
	)
	parser.add_argument(
		"--no-recursive",
		action="store_true",
		help="Only process files directly inside root.",
	)
	parser.add_argument(
		"--include-hidden",
		action="store_true",
		help="Include hidden/system files and folders.",
	)
	return parser.parse_args(argv)


def main(argv: list[str]) -> int:
	args = parse_args(argv)
	root = args.root.resolve()
	offline_exts = normalize_extensions(args.offline_ext)

	if os.name != "nt":
		print("ERROR: This script uses Windows attrib flags and must run on Windows.", file=sys.stderr)
		return 2

	if not root.exists() or not root.is_dir():
		print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
		return 2

	print(f"Root: {root}")
	print(f"Offline extensions: {', '.join(sorted(offline_exts))}")
	print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
	print()

	matched_offline = 0
	matched_online_only = 0
	changed = 0
	hydrated = 0
	failed = 0
	skipped = 0

	for path in iter_files(root, recursive=not args.no_recursive, include_hidden=args.include_hidden):
		if path.resolve() == Path(__file__).resolve():
			skipped += 1
			continue

		make_offline = has_offline_extension(path, offline_exts)
		if make_offline:
			matched_offline += 1
			action = "OFFLINE"
		else:
			matched_online_only += 1
			action = "ONLINE-ONLY"

		print(f"{action:11} {path}")

		if args.dry_run:
			continue

		result = run_attrib(path, make_offline=make_offline)
		if result.returncode == 0:
			changed += 1
			if make_offline:
				try:
					hydrate_file(path)
					hydrated += 1
				except OSError as error:
					failed += 1
					print(f"  FAILED: {error}", file=sys.stderr)
		else:
			failed += 1
			error = (result.stderr or result.stdout).strip()
			print(f"  FAILED: {error}", file=sys.stderr)

	print()
	print("Summary")
	print(f"  .py/.ipynb or configured offline files: {matched_offline}")
	print(f"  online-only files: {matched_online_only}")
	print(f"  skipped files: {skipped}")
	print(f"  hydrated offline files: {hydrated}")
	if args.dry_run:
		print(f"  changed successfully: {changed}")
		print(f"  failed: {failed}")
	else:
		print("  attributes changed and offline files hydrated automatically; pass --dry-run to preview only")

	return 1 if failed else 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
