import pytest
from mkdocs.exceptions import PluginError
from pathlib import Path

from mkdocs_piper_tts.plugin import PiperTTSPlugin


def test_extract_text_adds_pauses_for_paragraphs_and_line_breaks() -> None:
    html = "<p>First paragraph</p><p>Second line<br>continues here</p><p>Done.</p>"

    assert PiperTTSPlugin._extract_text(None, html) == "First paragraph. Second line, continues here. Done."


def test_missing_voice_error_lists_expected_paths_and_download_urls(tmp_path: Path) -> None:
    plugin = PiperTTSPlugin()
    plugin._model_dir = tmp_path / "models"

    with pytest.raises(PluginError) as error:
        plugin._voice_files(
            {
                "model": "voice.onnx",
                "config": "voice.onnx.json",
                "download_url": "https://example.test/voice.onnx",
            },
            "en",
        )

    message = str(error.value)
    assert "there is no ONNX voice model and/or JSON configuration" in message
    assert f"Model directory: {plugin._model_dir}" in message
    assert str(plugin._model_dir / "voice.onnx") in message
    assert str(plugin._model_dir / "voice.onnx.json") in message
    assert "https://example.test/voice.onnx" in message
    assert "https://example.test/voice.onnx.json" in message
