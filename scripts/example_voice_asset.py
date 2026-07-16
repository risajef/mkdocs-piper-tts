#!/usr/bin/env python3
"""Publish and restore the Piper example voice as a GitHub Release asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "examples" / "simple-site" / "models"
VOICE_NAME = "en_US-lessac-medium.onnx"
ASSET_TAG = "example-voice-v1"
ARCHIVE_NAME = f"{ASSET_TAG}.tar.gz"
CHECKSUM_NAME = f"{ARCHIVE_NAME}.sha256"
MANIFEST_NAME = "example-voice-assets.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("publish", "Package the local example voice and upload it to GitHub Releases"),
        ("restore", "Download, verify, and restore the example voice from GitHub Releases"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--tag", default=ASSET_TAG)
        command.add_argument("--repo", help="GitHub repository, such as owner/repository")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "publish":
        publish(args.tag, args.repo)
    else:
        restore(args.tag, args.repo)


def publish(tag: str, repo: str | None) -> None:
    voice_files = required_voice_files(MODEL_DIR)
    with tempfile.TemporaryDirectory(prefix="piper-example-voice-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        archive_path = temp_dir / f"{tag}.tar.gz"
        checksum_path = temp_dir / f"{tag}.tar.gz.sha256"
        manifest = {"version": 1, "files": {path.name: sha256(path) for path in voice_files}}
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in voice_files:
                archive.add(path, arcname=path.name)
            manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(manifest_bytes)
            archive.addfile(info, BytesReader(manifest_bytes))
        checksum_path.write_text(f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8")

        if not release_exists(tag, repo):
            run_gh("release", "create", tag, "--title", tag, "--notes", "Piper example voice artifact.", repo=repo)
        run_gh("release", "upload", tag, str(archive_path), str(checksum_path), "--clobber", repo=repo)
    print(f"Published Piper example voice asset to GitHub Release {tag}")


def restore(tag: str, repo: str | None) -> None:
    with tempfile.TemporaryDirectory(prefix="piper-example-voice-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        run_gh("release", "download", tag, "--pattern", f"{tag}.tar.gz*", "--dir", str(temp_dir), repo=repo)
        archive_path = temp_dir / f"{tag}.tar.gz"
        checksum_path = temp_dir / f"{tag}.tar.gz.sha256"
        verify_checksum(archive_path, checksum_path)
        extract_voice(archive_path, MODEL_DIR)
    print(f"Restored Piper example voice asset from GitHub Release {tag}")


def required_voice_files(directory: Path) -> tuple[Path, Path]:
    model_path = directory / VOICE_NAME
    config_path = model_path.with_suffix(".onnx.json")
    missing = [path for path in (model_path, config_path) if not path.is_file()]
    if missing:
        locations = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Piper example voice files are missing:\n{locations}")
    return model_path, config_path


def extract_voice(archive_path: Path, destination: Path) -> None:
    if not archive_path.is_file():
        raise SystemExit(f"Piper example voice archive is missing: {archive_path}")
    with tempfile.TemporaryDirectory(prefix="piper-example-voice-extract-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if any(not is_safe_member(member.name) for member in members):
                raise SystemExit("Piper example voice archive contains an unsafe path")
            archive.extractall(temp_dir, members, filter="data")
        manifest_path = temp_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise SystemExit("Piper example voice archive is missing its manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_files = {VOICE_NAME, f"{VOICE_NAME}.json"}
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, dict) or manifest.get("version") != 1 or set(files) != expected_files:
            raise SystemExit("Piper example voice archive has an invalid manifest")
        for filename, expected_hash in files.items():
            source = temp_dir / filename
            if not isinstance(expected_hash, str) or not source.is_file() or sha256(source) != expected_hash:
                raise SystemExit(f"Piper example voice checksum failed: {filename}")
        destination.mkdir(parents=True, exist_ok=True)
        for filename in expected_files:
            shutil.copy2(temp_dir / filename, destination / filename)


def verify_checksum(archive_path: Path, checksum_path: Path) -> None:
    if not archive_path.is_file():
        raise SystemExit(f"Piper example voice archive is missing: {archive_path}")
    if not checksum_path.is_file():
        raise SystemExit(f"Piper example voice checksum is missing: {checksum_path}")
    expected_hash, _, filename = checksum_path.read_text(encoding="utf-8").strip().partition("  ")
    if filename != archive_path.name or len(expected_hash) != 64 or sha256(archive_path) != expected_hash:
        raise SystemExit(f"Piper example voice checksum failed: {archive_path}")


def release_exists(tag: str, repo: str | None) -> bool:
    return subprocess.run(
        gh_command("release", "view", tag, repo=repo),
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0


def run_gh(*args: str, repo: str | None) -> None:
    subprocess.run(gh_command(*args, repo=repo), cwd=PROJECT_DIR, check=True)


def gh_command(*args: str, repo: str | None) -> list[str]:
    command = ["gh", *args]
    if repo:
        command.extend(("--repo", repo))
    return command


def is_safe_member(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BytesReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.data) - self.position
        chunk = self.data[self.position : self.position + size]
        self.position += len(chunk)
        return chunk


if __name__ == "__main__":
    main()
