#!/usr/bin/env python3
"""Restore, test, build, and deploy the mkdocs-piper-tts example Pages site."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
VOICE_ASSET_SCRIPT = PROJECT_DIR / "scripts" / "example_voice_asset.py"
EXAMPLE_CONFIG = PROJECT_DIR / "examples" / "simple-site" / "mkdocs.yml"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    subprocess.run([sys.executable, str(VOICE_ASSET_SCRIPT), "restore"], cwd=PROJECT_DIR, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_example_e2e.py", "-q", "-m", "not cuda"], cwd=PROJECT_DIR, check=True)
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "gh-deploy", "--config-file", str(EXAMPLE_CONFIG), "--strict", "--force"],
        cwd=PROJECT_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
