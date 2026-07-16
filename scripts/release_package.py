#!/usr/bin/env python3
"""Test, version, tag, and push a mkdocs-piper-tts release."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_DIR / "pyproject.toml"
VOICE_ASSET_SCRIPT = PROJECT_DIR / "scripts" / "example_voice_asset.py"
VERSION_PATTERN = re.compile(r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Package version without the v prefix, for example 0.2.3")
    return parser.parse_args()


def main() -> None:
    version = parse_args().version.removeprefix("v")
    if not VERSION_PATTERN.fullmatch(version):
        raise SystemExit("Version must be semantic MAJOR.MINOR.PATCH, for example 0.2.3")
    require_clean_worktree()
    subprocess.run([sys.executable, str(VOICE_ASSET_SCRIPT), "restore"], cwd=PROJECT_DIR, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not cuda"], cwd=PROJECT_DIR, check=True)
    set_version(version)
    run_git("add", "pyproject.toml")
    run_git("commit", "-m", f"Release v{version}")
    run_git("tag", f"v{version}")
    run_git("push", "origin", "main")
    run_git("push", "origin", f"v{version}")
    print(f"Pushed mkdocs-piper-tts v{version}; GitHub Actions will publish it to PyPI.")


def require_clean_worktree() -> None:
    status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_DIR, check=True, capture_output=True, text=True).stdout
    if status:
        raise SystemExit("Commit or stash all mkdocs-piper-tts changes before creating a release:\n" + status)


def set_version(version: str) -> None:
    contents = PYPROJECT_PATH.read_text(encoding="utf-8")
    updated, count = re.subn(r'(?m)^version = "[^"]+"$', f'version = "{version}"', contents, count=1)
    if count != 1:
        raise SystemExit(f"Could not update the project version in {PYPROJECT_PATH}")
    PYPROJECT_PATH.write_text(updated, encoding="utf-8")


def run_git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=PROJECT_DIR, check=True)


if __name__ == "__main__":
    main()
