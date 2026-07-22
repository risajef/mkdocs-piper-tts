from __future__ import annotations

import html as html_lib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mkdocs.exceptions import PluginError

from mkdocs_piper_tts.plugin import (
    PiperTTSPlugin,
    _PageTextExtractor,
    _audio_from_batch,
)


class _Config(dict):
    """Dictionary-shaped MkDocs config with the attribute MkDocs supplies."""


def _configured_plugin(
    tmp_path: Path, *, generate_audio: bool = True
) -> tuple[PiperTTSPlugin, _Config]:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    config = _Config(docs_dir=str(docs_dir))
    config.config_file_path = str(tmp_path / "mkdocs.yml")

    plugin = PiperTTSPlugin()
    plugin.config = {
        "asset_dir": "assets/piper-tts",
        "audio_dir": "audio",
        "model_dir": "models",
        "languages": {},
        "button_class": "piper-tts-button",
        "ffmpeg_path": Path("ffmpeg"),
        "generate_audio": generate_audio,
        "use_cuda": False,
        "batch_size": 2,
        "insert_reading_time_after_heading": False,
        "reading_time_min_seconds": 0.0,
    }
    plugin.on_config(config)
    return plugin, config


def _voice_files(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    model_path.write_bytes(b"model")
    config_path.write_text("{}", encoding="utf-8")
    return model_path, config_path


def test_page_text_extractor_ignores_non_spoken_tags() -> None:
    parser = _PageTextExtractor()
    parser.feed(
        "<h1>Heading</h1><script>skip()</script><p>Body &amp; text<br>next</p><style>nope</style>"
    )

    assert "".join(parser.parts) == "\nHeading.\n\nBody & text, next\n"


def test_on_config_honours_environment_and_indexes_valid_cached_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIPER_TTS_GENERATE_AUDIO", "off")
    cache_dir = tmp_path / "docs" / "assets" / "piper-tts" / "audio"
    cache_dir.mkdir(parents=True)
    (cache_dir / "valid.mp3").write_bytes(b"mp3")
    (cache_dir / "valid.mp3.json").write_text(
        json.dumps({"source_hash": "source"}), encoding="utf-8"
    )
    (cache_dir / "empty.mp3").write_bytes(b"")
    (cache_dir / "empty.mp3.json").write_text("{}", encoding="utf-8")
    (cache_dir / "bad.mp3.json").write_text("not json", encoding="utf-8")

    plugin, _ = _configured_plugin(tmp_path)

    assert plugin._generate_audio is False
    assert plugin._cache_index == {
        plugin._metadata_key({"source_hash": "source"}): (
            cache_dir / "valid.mp3",
            cache_dir / "valid.mp3.json",
        )
    }
    assert plugin._model_dir == tmp_path / "models"


def test_on_config_rejects_invalid_generate_audio_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = PiperTTSPlugin()
    plugin.config = {"generate_audio": True}
    monkeypatch.setenv("PIPER_TTS_GENERATE_AUDIO", "sometimes")

    with pytest.raises(PluginError, match="must be a boolean"):
        plugin.on_config({})


def test_configured_languages_adds_default_config_and_validates_custom_language() -> (
    None
):
    plugin = PiperTTSPlugin()
    plugin.config = {"languages": {"EN-us": {"model": "custom.onnx", "label": "Hear"}}}

    languages = plugin._configured_languages()

    assert languages["en"] == {
        "model": "custom.onnx",
        "label": "Hear",
        "config": "custom.onnx.json",
    }
    assert languages["de"]["config"] == "de_DE-thorsten-medium.onnx.json"

    plugin.config = {"languages": {"fr": {}}}
    with pytest.raises(PluginError, match="must define a model"):
        plugin._configured_languages()


def test_voice_files_and_speakers_support_names_numbers_and_errors(
    tmp_path: Path,
) -> None:
    plugin = PiperTTSPlugin()
    plugin._model_dir = tmp_path
    model_path, config_path = _voice_files(tmp_path)
    config_path.write_text(
        json.dumps({"speaker_id_map": {"alice": 3, "bob": 7}}), encoding="utf-8"
    )

    assert plugin._voice_files(
        {"model": model_path.name, "config": config_path.name}, "en"
    ) == (model_path, config_path)
    assert plugin._resolve_speaker_id({"speaker": "alice"}, config_path, "en") == 3
    assert plugin._resolve_speaker_id({"speaker": 7}, config_path, "en") == 7
    assert plugin._resolve_speaker_id({}, config_path, "en") is None

    with pytest.raises(PluginError, match="Unknown Piper speaker"):
        plugin._resolve_speaker_id({"speaker": "nobody"}, config_path, "en")

    config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PluginError, match="Could not read Piper speaker map"):
        plugin._resolve_speaker_id({"speaker": "alice"}, config_path, "en")


def test_cache_paths_status_and_hashing(tmp_path: Path) -> None:
    plugin, _ = _configured_plugin(tmp_path)
    source = tmp_path / "docs" / "guide.md"
    source.write_text("content", encoding="utf-8")
    first_hash = plugin._hash_file(source)
    source.write_text("changed", encoding="utf-8")

    assert plugin._hash_file(source) != first_hash
    audio_path, metadata_path = plugin._cache_paths("guide/Über view.md", "intro")
    assert (
        audio_path.relative_to(plugin._audio_cache_dir).as_posix()
        == "guide/%C3%9Cber%20view/intro.mp3"
    )
    unicode_page_path, _ = plugin._cache_paths("你好.md", "intro")
    japanese_page_path, _ = plugin._cache_paths("こんにちは.md", "intro")
    assert unicode_page_path != japanese_page_path
    assert metadata_path == audio_path.with_suffix(".mp3.json")
    assert plugin._cache_status(
        audio_path, metadata_path, {"source_hash": first_hash}
    ) == (False, "audio missing or empty")

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"mp3")
    metadata_path.write_text("not json", encoding="utf-8")
    assert plugin._cache_status(
        audio_path, metadata_path, {"source_hash": first_hash}
    ) == (
        False,
        "metadata missing or invalid",
    )
    metadata_path.write_text("[]", encoding="utf-8")
    assert plugin._cache_status(
        audio_path, metadata_path, {"source_hash": first_hash}
    ) == (False, "metadata is not an object")
    metadata_path.write_text(
        json.dumps({"source_hash": "other", "plugin_hash": "new"}), encoding="utf-8"
    )
    assert plugin._cache_status(
        audio_path, metadata_path, {"source_hash": first_hash}
    ) == (
        False,
        "metadata mismatch (plugin_hash, source_hash)",
    )
    metadata_path.write_text(
        json.dumps({"source_hash": first_hash, "extra": True}), encoding="utf-8"
    )
    assert plugin._cache_status(
        audio_path, metadata_path, {"source_hash": first_hash}
    ) == (True, "valid")


def test_page_content_queues_then_reuses_cached_audio(tmp_path: Path) -> None:
    plugin, config = _configured_plugin(tmp_path)
    model_dir = tmp_path / "models"
    model_path, config_path = _voice_files(model_dir)
    plugin._languages = {"en": {"model": model_path.name, "config": config_path.name}}
    source = Path(config["docs_dir"]) / "page.md"
    source.write_text("# Page", encoding="utf-8")
    page_file = SimpleNamespace(abs_src_path=str(source), src_path="page.md")
    page = SimpleNamespace(meta={"lang": "en-GB"}, file=page_file)

    html = "<p>Hello world</p>"
    assert plugin.on_page_content(html, page=page, config=config, files=[]) == html
    audio_path, task = next(iter(plugin._pending_audio.items()))
    assert task["text"] == "Hello world"
    assert task["speaker_id"] is None
    assert task["source_path"] == "page.md"
    assert task["section_title"] == "Page"
    expected_metadata = task["expected_metadata"]
    assert plugin._cache_misses == 1
    assert plugin._playlist_by_page["page.md"][0]["title"] == "Page"
    assert plugin._playlist_by_page["page.md"][0]["duration_seconds"] > 0

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"mp3")
    audio_path.with_suffix(".mp3.json").write_text(
        json.dumps(expected_metadata), encoding="utf-8"
    )
    plugin._pending_audio.clear()
    plugin.on_page_content(html, page=page, config=config, files=[])

    assert plugin._cache_hits == 1
    assert plugin._pending_audio == {}
    assert plugin._audio_by_page["page.md"] == audio_path


def test_page_content_invalidates_cache_when_voice_changes(tmp_path: Path) -> None:
    plugin, config = _configured_plugin(tmp_path)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_a = model_dir / "voice-a.onnx"
    config_a = model_dir / "voice-a.onnx.json"
    model_b = model_dir / "voice-b.onnx"
    config_b = model_dir / "voice-b.onnx.json"
    model_a.write_bytes(b"voice-a")
    config_a.write_text("{}", encoding="utf-8")
    model_b.write_bytes(b"voice-b")
    config_b.write_text("{}", encoding="utf-8")

    plugin._languages = {"en": {"model": model_a.name, "config": config_a.name}}
    source = Path(config["docs_dir"]) / "page.md"
    source.write_text("# Page", encoding="utf-8")
    page_file = SimpleNamespace(abs_src_path=str(source), src_path="page.md")
    page = SimpleNamespace(meta={"lang": "en"}, file=page_file)

    html = "<p>Hello world</p>"
    plugin.on_page_content(html, page=page, config=config, files=[])
    audio_path, task = next(iter(plugin._pending_audio.items()))
    expected_metadata = task["expected_metadata"]

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"mp3")
    audio_path.with_suffix(".mp3.json").write_text(
        json.dumps(expected_metadata), encoding="utf-8"
    )
    plugin._pending_audio.clear()
    plugin._cache_hits = 0
    plugin._cache_misses = 0

    plugin._languages = {"en": {"model": model_b.name, "config": config_b.name}}
    plugin.on_page_content(html, page=page, config=config, files=[])

    assert plugin._cache_hits == 0
    assert plugin._cache_misses == 1
    assert audio_path in plugin._pending_audio


def test_page_content_skips_unknown_language_and_rejects_missing_source(
    tmp_path: Path,
) -> None:
    plugin, config = _configured_plugin(tmp_path)
    no_voice_page = SimpleNamespace(
        meta={"lang": "fr"}, file=SimpleNamespace(src_path="page.md")
    )
    assert (
        plugin.on_page_content(
            "<p>Ignored</p>", page=no_voice_page, config=config, files=[]
        )
        == "<p>Ignored</p>"
    )

    model_dir = tmp_path / "models"
    model_path, config_path = _voice_files(model_dir)
    plugin._languages = {"en": {"model": model_path.name, "config": config_path.name}}
    missing_source_page = SimpleNamespace(
        meta={"lang": "en"}, file=SimpleNamespace(abs_src_path="", src_path="gone.md")
    )
    with pytest.raises(PluginError, match="source Markdown is missing"):
        plugin.on_page_content(
            "<p>Missing</p>", page=missing_source_page, config=config, files=[]
        )


def test_render_button_urls_escapes_labels_and_registers_template_helper(
    tmp_path: Path,
) -> None:
    plugin, _ = _configured_plugin(tmp_path)
    plugin._playlist_by_page = {
        "guide/page.md": [
            {
                "title": "Intro",
                "audio_path": plugin._audio_cache_dir / "guide" / "page" / "voice.mp3",
                "duration_seconds": 10.0,
            }
        ]
    }
    plugin._audio_by_page = {
        "guide/page.md": plugin._audio_cache_dir / "guide" / "page" / "voice.mp3"
    }
    plugin._languages = {"en": {"label": "Listen & learn"}}
    page = SimpleNamespace(
        file=SimpleNamespace(src_path="guide/page.md"),
        meta={"lang": "en"},
        url="guide/page/",
    )

    rendered = str(plugin.render_button(page))
    assert 'src="../../assets/piper-tts/audio/guide/page/voice.mp3"' in rendered
    assert 'aria-label="Listen &amp; learn"' in rendered
    assert "piper-tts-button-playlist" in rendered
    assert "data-playlist" in rendered
    playlist_match = re.search(r'data-playlist="([^"]+)"', rendered)
    assert playlist_match is not None
    playlist = json.loads(html_lib.unescape(playlist_match.group(1)))
    assert playlist == [
        {"title": "Intro", "url": "../../assets/piper-tts/audio/guide/page/voice.mp3"}
    ]
    assert str(plugin.render_button()) == ""
    env = SimpleNamespace(globals={})
    assert plugin.on_env(env, {}, []) is env
    assert env.globals["piper_tts_button"] == plugin.render_button
    assert env.globals["piper_tts_playlist"] == plugin.render_button
    assert env.globals["piper_tts_reading_time"] == plugin.render_reading_time
    assert env.globals["piper_tts_controls"] == plugin.render_controls


def test_render_button_compact_mode_uses_stable_classes_and_language_labels(
    tmp_path: Path,
) -> None:
    plugin, _ = _configured_plugin(tmp_path)
    plugin._playlist_by_page = {
        "guide/page.md": [
            {
                "title": "Intro",
                "audio_path": plugin._audio_cache_dir / "guide" / "page" / "voice.mp3",
                "duration_seconds": 10.0,
            },
            {
                "title": "Body",
                "audio_path": plugin._audio_cache_dir / "guide" / "page" / "body.mp3",
                "duration_seconds": 10.0,
            },
        ]
    }
    plugin._languages = {
        "de": {
            "label": "Vorlesen",
            "prev_label": "Vorheriger Titel",
            "next_label": "Nächster Titel",
            "play_label": "Abspielen",
            "pause_label": "Pause",
        }
    }
    page = SimpleNamespace(
        file=SimpleNamespace(src_path="guide/page.md"),
        meta={"lang": "de"},
        url="guide/page/",
    )

    rendered = str(plugin.render_button(page, mode="compact"))

    assert "piper-tts-button-wrapper piper-tts-mode-compact" in rendered
    assert "piper-tts-compact-bar" in rendered
    assert 'class="piper-tts-prev" aria-label="Vorheriger Titel"' in rendered
    assert 'class="piper-tts-next" aria-label="Nächster Titel"' in rendered
    assert 'class="piper-tts-play-pause" aria-label="Abspielen"' in rendered
    assert "piper-tts-now-playing" in rendered
    assert "piper-tts-button-playlist" in rendered
    assert "data-track-index" in rendered
    assert "controls" not in rendered.split("<script>", 1)[0]

    with pytest.raises(PluginError, match="Piper TTS player mode"):
        plugin.render_button(page, mode="bogus")


@pytest.mark.parametrize(
    ("seconds", "formatted"),
    [(-1, "0s"), (59.9, "59s"), (61, "1m01s"), (3_661, "1h01m")],
)
def test_duration_and_batch_size_validation(seconds: float, formatted: str) -> None:
    plugin = PiperTTSPlugin()
    plugin.config = {"batch_size": 1}

    assert plugin._format_duration(seconds) == formatted
    assert plugin._batch_size() == 1
    plugin.config["batch_size"] = 0
    with pytest.raises(PluginError, match="one or greater"):
        plugin._batch_size()


def test_post_build_copies_cached_audio_and_cache_only_mode_reports_pending(
    tmp_path: Path,
) -> None:
    plugin, config = _configured_plugin(tmp_path, generate_audio=False)
    config.site_dir = str(tmp_path / "site")
    cached_audio = plugin._audio_cache_dir / "cached.mp3"
    cached_audio.parent.mkdir(parents=True, exist_ok=True)
    cached_audio.write_bytes(b"mp3")
    plugin._playlist_by_page = {
        "cached.md": [
            {"title": "Cached", "audio_path": cached_audio, "duration_seconds": 1.0}
        ]
    }
    stale_site_audio = (
        Path(config.site_dir) / "assets" / "piper-tts" / "audio" / "stale.mp3"
    )
    stale_site_audio.parent.mkdir(parents=True, exist_ok=True)
    stale_site_audio.write_bytes(b"stale")

    plugin.on_post_build(config)
    assert (
        Path(config.site_dir) / "assets" / "piper-tts" / "audio" / "cached.mp3"
    ).read_bytes() == b"mp3"
    assert not stale_site_audio.exists()

    plugin._pending_audio[plugin._audio_cache_dir / "new.mp3"] = {
        "model_path": Path("model"),
        "config_path": Path("config"),
        "text": "text",
        "speaker_id": None,
        "expected_metadata": {},
        "source_path": "new.md",
        "section_title": "New",
    }
    with pytest.raises(
        PluginError, match="cache-only build found missing or stale audio for: new.md"
    ):
        plugin.on_post_build(config)


def test_audio_from_batch_pads_inputs_normalizes_waveforms_and_tracks_timing() -> None:
    class Session:

        def __init__(self):
            self.inputs = None

        def run(self, _outputs, inputs):
            self.inputs = inputs
            return [
                np.array(
                    [[[[0.0, 0.2, -0.4, 0.0]]], [[[0.1, 0.0, 0.0, 0.0]]]],
                    dtype=np.float32,
                )
            ]

    session = Session()
    voice = SimpleNamespace(
        config=SimpleNamespace(
            length_scale=1.0,
            noise_scale=0.5,
            noise_w=0.8,
            num_speakers=2,
            sample_rate=20,
        ),
        session=session,
    )
    timing = {"session": 0.0, "postprocess": 0.0}

    waveforms = _audio_from_batch(voice, [[1, 2], [3]], speaker_id=4, timing=timing)

    assert session.inputs["input"].tolist() == [[1, 2], [3, 0]]
    assert session.inputs["input_lengths"].tolist() == [2, 1]
    assert session.inputs["sid"].tolist() == [4, 4]
    assert np.allclose(waveforms[0], [0.0, 0.5, -1.0])
    assert np.allclose(waveforms[1], [1.0])
    assert timing["session"] >= 0
    assert timing["postprocess"] >= 0


def test_generate_pending_audio_creates_real_mp3_from_piper_voice(
    tmp_path: Path,
) -> None:
    model_dir = (
        Path(__file__).resolve().parents[1] / "examples" / "simple-site" / "models"
    )
    model_path = model_dir / "en_US-lessac-medium.onnx"
    config_path = model_path.with_suffix(".onnx.json")
    if not model_path.is_file() or not config_path.is_file():
        pytest.fail(f"The checked-in Piper test voice is missing from {model_dir}")

    plugin, _ = _configured_plugin(tmp_path)
    audio_path = plugin._audio_cache_dir / "real-synthesis.mp3"
    expected_metadata = {"plugin_hash": "test", "source_hash": "real-synthesis"}
    plugin._pending_audio = {
        audio_path: {
            "model_path": model_path,
            "config_path": config_path,
            "text": "This is a real Piper synthesis test.",
            "speaker_id": None,
            "expected_metadata": expected_metadata,
            "source_path": "test.md",
            "section_title": "test",
        },
    }

    plugin._generate_pending_audio()

    assert audio_path.is_file()
    assert audio_path.stat().st_size > 0
    assert audio_path.read_bytes()[:3] in {
        b"ID3",
        b"\xff\xfb",
        b"\xff\xf3",
        b"\xff\xf2",
    }
    metadata = json.loads(
        audio_path.with_suffix(".mp3.json").read_text(encoding="utf-8")
    )
    assert metadata["plugin_hash"] == expected_metadata["plugin_hash"]
    assert metadata["source_hash"] == expected_metadata["source_hash"]
    assert metadata["duration_seconds"] > 0


def test_extract_sections_for_single_h1_with_intro_and_h2_playlist() -> None:
    plugin = PiperTTSPlugin()
    html = "<h1>Main Topic</h1><p>Welcome.</p><h2>Setup</h2><p>Install this.</p><h2>Run</h2><p>Execute.</p>"

    sections = plugin._extract_sections(html, fallback_title="Fallback")

    assert [section["title"] for section in sections] == [
        "(Intro) Main Topic",
        "Setup",
        "Run",
    ]
    assert "Main Topic" in sections[0]["text"]
    assert "Setup" in sections[1]["text"]


def test_extract_sections_for_single_h1_without_h2_is_total() -> None:
    plugin = PiperTTSPlugin()
    html = "<h1>Only Title</h1><p>All content.</p>"

    sections = plugin._extract_sections(html, fallback_title="Fallback")

    assert len(sections) == 1
    assert sections[0]["title"] == "Only Title"
    assert "All content" in sections[0]["text"]


def test_render_reading_time_sums_playlist_duration(tmp_path: Path) -> None:
    plugin, _ = _configured_plugin(tmp_path)
    plugin._playlist_by_page = {
        "guide/page.md": [
            {
                "title": "Intro",
                "audio_path": plugin._audio_cache_dir / "a.mp3",
                "duration_seconds": 61.1,
            },
            {
                "title": "Body",
                "audio_path": plugin._audio_cache_dir / "b.mp3",
                "duration_seconds": 59.0,
            },
        ]
    }
    page = SimpleNamespace(file=SimpleNamespace(src_path="guide/page.md"))

    rendered = str(plugin.render_reading_time(page))

    assert "approximate reading time:" in rendered
    assert "2m00s" in rendered


def test_render_reading_time_uses_language_specific_label_by_default(
    tmp_path: Path,
) -> None:
    plugin, _ = _configured_plugin(tmp_path)
    plugin._playlist_by_page = {
        "guide/page.md": [
            {
                "title": "Intro",
                "audio_path": plugin._audio_cache_dir / "a.mp3",
                "duration_seconds": 61.0,
            },
        ]
    }
    plugin._languages = {"de": {"reading_time_label": "Ungefähre Lesezeit:"}}
    page = SimpleNamespace(
        file=SimpleNamespace(src_path="guide/page.md"), meta={"lang": "de"}
    )

    rendered = str(plugin.render_reading_time(page))

    assert "Ungefähre Lesezeit:" in rendered

    explicit = str(plugin.render_reading_time(page, "Custom label:"))
    assert "Custom label:" in explicit
    assert "Ungefähre Lesezeit:" not in explicit


def test_render_reading_time_hides_short_durations_below_min_seconds(
    tmp_path: Path,
) -> None:
    plugin, _ = _configured_plugin(tmp_path)
    plugin._playlist_by_page = {
        "guide/page.md": [
            {
                "title": "Intro",
                "audio_path": plugin._audio_cache_dir / "a.mp3",
                "duration_seconds": 45.0,
            },
        ]
    }
    page = SimpleNamespace(file=SimpleNamespace(src_path="guide/page.md"))

    assert str(plugin.render_reading_time(page, min_seconds=60)) == ""

    rendered = str(plugin.render_reading_time(page, min_seconds=30))
    assert "45s" in rendered


def test_render_reading_time_uses_config_default_min_seconds(tmp_path: Path) -> None:
    plugin, _ = _configured_plugin(tmp_path)
    plugin.config["reading_time_min_seconds"] = 60.0
    plugin._playlist_by_page = {
        "guide/page.md": [
            {
                "title": "Intro",
                "audio_path": plugin._audio_cache_dir / "a.mp3",
                "duration_seconds": 45.0,
            },
        ]
    }
    page = SimpleNamespace(file=SimpleNamespace(src_path="guide/page.md"))

    assert str(plugin.render_reading_time(page)) == ""
    # An explicit call-site override still wins over the configured default.
    assert "45s" in str(plugin.render_reading_time(page, min_seconds=0))


def test_render_controls_combines_reading_time_and_player(tmp_path: Path) -> None:
    plugin, _ = _configured_plugin(tmp_path)
    plugin._playlist_by_page = {
        "guide/page.md": [
            {
                "title": "Intro",
                "audio_path": plugin._audio_cache_dir / "guide" / "page" / "voice.mp3",
                "duration_seconds": 61.0,
            }
        ]
    }
    plugin._languages = {
        "en": {"label": "Listen", "reading_time_label": "Approximate reading time:"}
    }
    page = SimpleNamespace(
        file=SimpleNamespace(src_path="guide/page.md"),
        meta={"lang": "en"},
        url="guide/page/",
    )

    rendered = str(plugin.render_controls(page))

    assert '<div class="piper-tts-controls">' in rendered
    assert "piper-tts-reading-time" in rendered
    assert "piper-tts-button-wrapper" in rendered
    assert rendered.index("piper-tts-reading-time") < rendered.index(
        "piper-tts-button-wrapper"
    )
    assert str(plugin.render_controls()) == ""


def test_on_page_content_auto_inserts_reading_time_after_first_heading(
    tmp_path: Path,
) -> None:
    plugin, config = _configured_plugin(tmp_path)
    plugin.config["insert_reading_time_after_heading"] = True
    model_dir = tmp_path / "models"
    model_path, config_path = _voice_files(model_dir)
    plugin._languages = {"en": {"model": model_path.name, "config": config_path.name}}
    source = Path(config["docs_dir"]) / "page.md"
    source.write_text("# Page", encoding="utf-8")
    page_file = SimpleNamespace(abs_src_path=str(source), src_path="page.md")
    page = SimpleNamespace(meta={"lang": "en"}, file=page_file)

    html_content = "<h1>Page</h1><p>Hello world</p>"
    result = plugin.on_page_content(html_content, page=page, config=config, files=[])

    assert (
        result.index("</h1>")
        < result.index("piper-tts-reading-time")
        < result.index("<p>Hello world</p>")
    )
