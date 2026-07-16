from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPOSITORY_ROOT / "examples" / "simple-site"
VOICE_NAME = "en_US-lessac-medium.onnx"
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"


def _model_dir() -> Path:
    configured_path = os.environ.get("PIPER_TTS_TEST_MODEL_DIR")
    if configured_path:
        return Path(configured_path).resolve()
    return Path.home() / ".cache" / "mkdocs-piper-tts" / "test-voices" / "en_US-lessac-medium"


@pytest.fixture(scope="session")
def test_model_dir() -> Path:
    model_dir = _model_dir()
    model_path = model_dir / VOICE_NAME
    config_path = model_path.with_suffix(".onnx.json")
    if model_path.is_file() and config_path.is_file():
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in (VOICE_NAME, f"{VOICE_NAME}.json"):
        try:
            urllib.request.urlretrieve(VOICE_URL + filename, model_dir / filename)
        except OSError as error:
            pytest.skip(f"could not download the Piper test voice: {error}")
    return model_dir


def _build_example(tmp_path: Path, model_dir: Path, *, use_cuda: bool) -> subprocess.CompletedProcess[str]:
    project_dir = tmp_path / ("cuda" if use_cuda else "cpu")
    shutil.copytree(EXAMPLE_DIR, project_dir)
    model_path = next(model_dir.glob("*.onnx"), None)
    if model_path is None or not model_path.with_suffix(".onnx.json").is_file():
        pytest.skip(f"no Piper model and matching .onnx.json found in {model_dir}")

    config_path = project_dir / "mkdocs.yml"
    config = config_path.read_text(encoding="utf-8")
    config = config.replace("model_dir: models", f"model_dir: {model_dir.as_posix()}")
    config = config.replace("model: en_US-lessac-medium.onnx", f"model: {model_path.name}")
    config = config.replace("use_cuda: false", f"use_cuda: {str(use_cuda).lower()}")
    config_path.write_text(config, encoding="utf-8")

    return subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict"],
        cwd=project_dir,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_build(result: subprocess.CompletedProcess[str], project_dir: Path, provider: str) -> None:
    assert result.returncode == 0, result.stdout + result.stderr
    assert provider in result.stdout + result.stderr
    assert list((project_dir / "site" / "assets" / "piper-tts" / "audio").glob("*.mp3"))


def test_example_builds_on_cpu(tmp_path: Path, test_model_dir: Path) -> None:
    project_dir = tmp_path / "cpu"
    result = _build_example(tmp_path, test_model_dir, use_cuda=False)
    _assert_build(result, project_dir, "CPUExecutionProvider")


@pytest.mark.cuda
def test_example_builds_on_cuda(tmp_path: Path, test_model_dir: Path) -> None:
    try:
        import onnxruntime

        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            pytest.skip("ONNX Runtime was built without CUDAExecutionProvider")
    except ImportError:
        pytest.skip("onnxruntime is not installed")

    project_dir = tmp_path / "cuda"
    result = _build_example(tmp_path, test_model_dir, use_cuda=True)
    if result.returncode != 0 and "CUDAExecutionProvider" not in result.stdout + result.stderr:
        pytest.skip("CUDA execution provider could not initialize on this host")
    _assert_build(result, project_dir, "CUDAExecutionProvider")