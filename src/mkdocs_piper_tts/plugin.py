from __future__ import annotations

import hashlib
import html
import json
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import time
import wave
import numpy as np
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

from markupsafe import Markup
from mkdocs.config import config_options
from mkdocs.exceptions import PluginError
from mkdocs.plugins import BasePlugin, get_plugin_logger
from mkdocs.structure.pages import Page
from piper import PiperVoice


log = get_plugin_logger(__name__)

DEFAULT_LANGUAGES = {
    "de": {
        "model": "de_DE-thorsten-medium.onnx",
        "label": "Vorlesen",
        "download_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
    },
    "en": {
        "model": "en_US-lessac-medium.onnx",
        "label": "Listen",
        "download_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    },
}


def _log_timing(label: str, started: float, *, plugin_started: float | None = None, level: str = "info"):
    """Log wall-clock and monotonic timing so build phases can be correlated."""
    elapsed = time.perf_counter() - started
    since_plugin = ""
    if plugin_started is not None:
        since_plugin = f", since_plugin={time.perf_counter() - plugin_started:.3f}s"
    message = (
        f"Piper TTS timing: {label} "
        f"at={datetime.now().isoformat(timespec='milliseconds')} "
        f"duration={elapsed:.3f}s{since_plugin}"
    )
    getattr(log, level)(message)


def _log_duration(label: str, duration: float, *, plugin_started: float | None = None):
    """Log an already accumulated duration with the same timing format."""
    since_plugin = ""
    if plugin_started is not None:
        since_plugin = f", since_plugin={time.perf_counter() - plugin_started:.3f}s"
    log.info(
        "Piper TTS timing: %s at=%s duration=%.3fs%s",
        label,
        datetime.now().isoformat(timespec="milliseconds"),
        duration,
        since_plugin,
    )


def _load_voice(
    model_path: Path,
    config_path: Path,
    *,
    use_cuda: bool,
    plugin_started: float | None = None,
):
    """Load Piper with either the CUDA or CPU execution provider."""
    started = time.perf_counter()
    import onnxruntime  # pylint: disable=import-outside-toplevel
    from piper.config import PiperConfig  # pylint: disable=import-outside-toplevel

    _log_timing(
        "voice import",
        started,
        plugin_started=plugin_started,
    )

    started = time.perf_counter()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "1")
    onnxruntime.preload_dlls(cuda=use_cuda, cudnn=use_cuda, msvc=False)
    _log_timing(
        "CUDA DLL preload",
        started,
        plugin_started=plugin_started,
    )
    started = time.perf_counter()
    config = PiperConfig.from_dict(json.loads(Path(config_path).read_text(encoding="utf-8")))
    session_options = onnxruntime.SessionOptions()
    _log_timing(
        "Piper config/session options",
        started,
        plugin_started=plugin_started,
    )
    if use_cuda:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    started = time.perf_counter()
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )
    _log_timing(
        "ONNX Runtime session creation",
        started,
        plugin_started=plugin_started,
    )

    started = time.perf_counter()
    voice = PiperVoice(session=session, config=config)
    if use_cuda and "CUDAExecutionProvider" not in voice.session.get_providers():
        raise RuntimeError(f"Piper TTS could not initialize CUDAExecutionProvider; providers={voice.session.get_providers()}")
    _log_timing(
        "Piper voice initialization",
        started,
        plugin_started=plugin_started,
    )
    return voice


def _audio_from_batch(
    voice: PiperVoice,
    phoneme_ids: list[list[int]],
    speaker_id: int | None,
    timing: dict[str, float] | None = None,
):
    """Run one padded batch and return one trimmed float waveform per item."""

    max_length = max(len(ids) for ids in phoneme_ids)
    input_ids = np.zeros((len(phoneme_ids), max_length), dtype=np.int64)
    input_lengths = np.empty(len(phoneme_ids), dtype=np.int64)
    for index, ids in enumerate(phoneme_ids):
        input_ids[index, : len(ids)] = ids
        input_lengths[index] = len(ids)

    # Piper 1.2 exposes the model defaults directly on PiperConfig. Newer
    # releases renamed ``noise_w`` to ``noise_w_scale`` and added
    # SynthesisConfig, but this plugin does not override those values.
    length_scale = voice.config.length_scale
    noise_scale = voice.config.noise_scale
    noise_w_scale = getattr(
        voice.config,
        "noise_w_scale",
        getattr(voice.config, "noise_w", 0.8),
    )
    inputs = {
        "input": input_ids,
        "input_lengths": input_lengths,
        "scales": np.asarray([noise_scale, length_scale, noise_w_scale], dtype=np.float32),
    }
    if voice.config.num_speakers > 1:
        inputs["sid"] = np.full(len(phoneme_ids), 0 if speaker_id is None else speaker_id, dtype=np.int64)

    session_started = time.perf_counter()
    output = voice.session.run(None, inputs)[0][:, 0, 0, :]
    if timing is not None:
        timing["session"] += time.perf_counter() - session_started

    postprocess_started = time.perf_counter()
    waveforms = []
    for audio in output:
        # Batched Piper output is padded to the longest item. Remove the
        # generated tail noise before writing the individual WAV files.
        window = max(1, voice.config.sample_rate // 20)
        if len(audio) < window:
            energy = np.sqrt(np.mean(audio * audio))[None]
        else:
            # Moving RMS via cumulative sum. O(n), unlike np.convolve here.
            squared = audio * audio
            cumulative = np.cumsum(squared, dtype=np.float64)
            window_energy = cumulative[window - 1 :].copy()
            window_energy[1:] -= cumulative[:-window]
            energy = np.sqrt(window_energy / window)
        active = np.flatnonzero(energy > 0.005)
        end = min(len(audio), (int(active[-1]) + window) if len(active) else 0)
        audio = audio[:end]
        if len(audio):
            maximum = np.max(np.abs(audio)) if len(audio) else 0
            if maximum < 1e-8:
                audio = np.zeros_like(audio)
            else:
                audio = audio / maximum
        waveforms.append(np.clip(audio, -1.0, 1.0).astype(np.float32))
    if timing is not None:
        timing["postprocess"] += time.perf_counter() - postprocess_started
    return waveforms


def _generate_audio_batch(
    tasks: list[dict],
    voice: PiperVoice,
    batch_size: int,
    ffmpeg_path: Path,
    progress_callback: callable | None = None,
    plugin_started: float | None = None,
    inference_progress_callback: callable | None = None,
):
    """Generate and encode a group of pages sharing one Piper model/voice."""

    started = time.perf_counter()
    durations = {}
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_dir = Path(temporary_dir)
        wav_paths = [temporary_dir / f"{index}.wav" for index in range(len(tasks))]
        wav_files = []
        segments = []
        sample_counts = [0 for _ in tasks]
        phonemize_started = time.perf_counter()
        for index, task in enumerate(tasks):
            text = task["text"]
            speaker_id = task["speaker_id"]
            wav_file = wave.open(str(wav_paths[index]), "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)
            wav_files.append(wav_file)
            for phonemes in voice.phonemize(text):
                if phonemes:
                    ids = voice.phonemes_to_ids(phonemes)
                    segments.append((index, ids, speaker_id))
        _log_timing(
            f"phonemization ({len(tasks)} files, {len(segments)} segments)",
            phonemize_started,
            plugin_started=plugin_started,
        )

        inference_started = time.perf_counter()
        inference_batches = 0
        total_batches = (len(segments) + batch_size - 1) // batch_size
        inference_timing = {"session": 0.0, "postprocess": 0.0}
        if inference_progress_callback is not None:
            inference_progress_callback(0, total_batches, 0, 0.0)
        try:
            for offset in range(0, len(segments), batch_size):
                inference_batches += 1
                current = segments[offset : offset + batch_size]
                waveforms = _audio_from_batch(
                    voice,
                    [segment[1] for segment in current],
                    current[0][2],
                    timing=inference_timing,
                )
                for segment, waveform in zip(current, waveforms):
                    sample_counts[segment[0]] += len(waveform)
                    wav_files[segment[0]].writeframes(np.clip(waveform * 32767, -32767, 32767).astype(np.int16).tobytes())
                if inference_progress_callback is not None:
                    inference_progress_callback(
                        inference_batches,
                        total_batches,
                        min(offset + len(current), len(segments)),
                        time.perf_counter() - inference_started,
                    )
        finally:
            for wav_file in wav_files:
                wav_file.close()
        _log_timing(
            f"inference loop ({inference_batches} batches)",
            inference_started,
            plugin_started=plugin_started,
        )
        _log_duration(
            "CUDA session.run",
            inference_timing["session"],
            plugin_started=plugin_started,
        )
        _log_duration(
            "CPU waveform postprocessing",
            inference_timing["postprocess"],
            plugin_started=plugin_started,
        )

        encoding_started = time.perf_counter()
        for index, task in enumerate(tasks):
            audio_path = task["audio_path"]
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            mp3_path = temporary_dir / f"{index}.mp3"
            result = subprocess.run(
                [
                    ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(wav_paths[index]),
                    "-codec:a",
                    "libmp3lame",
                    "-q:a",
                    "3",
                    "-y",
                    str(mp3_path),
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            if result.returncode != 0:
                error = result.stderr.strip() or f"ffmpeg exited with code {result.returncode}"
                raise RuntimeError(f"Piper TTS MP3 conversion failed: {error}")
            # The temporary directory may be on a different filesystem
            # (e.g. /tmp versus the project volume), so Path.replace() can
            # fail with EXDEV. shutil.move() falls back to copy-and-unlink.
            shutil.move(str(mp3_path), str(audio_path))
            durations[str(audio_path)] = sample_counts[index] / voice.config.sample_rate
            if progress_callback is not None:
                progress_callback(audio_path, task["source_path"])
        _log_timing(
            f"FFmpeg encoding and MP3 copies ({len(tasks)} files)",
            encoding_started,
            plugin_started=plugin_started,
        )

    _log_timing(
        f"audio batch ({len(tasks)} files)",
        started,
        plugin_started=plugin_started,
    )
    return time.perf_counter() - started, durations


class _PageTextExtractor(HTMLParser):
    """Extract text from HTML while ignoring script/style/noscript content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.ignored_depth += 1
        elif not self.ignored_depth and tag == "br":
            self.parts.append(", ")
        elif not self.ignored_depth and tag in {"div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif not self.ignored_depth and tag in {"h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            self.parts.append(".\n")
        elif not self.ignored_depth and tag in {"div", "li", "p", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


class _PageSectionExtractor(HTMLParser):
    """Capture heading structure and surrounding text from rendered HTML."""

    _HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    _BLOCK_TAGS = {"div", "p", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.events = []
        self._text_parts = []
        self._ignored_depth = 0
        self._heading_tag = None
        self._heading_parts = []

    def _flush_text(self) -> None:
        if not self._text_parts:
            return
        raw_text = "".join(self._text_parts)
        if raw_text.strip():
            self.events.append(
                {
                    "type": "text",
                    "raw": raw_text,
                }
            )
        self._text_parts.clear()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in self._HEADING_TAGS:
            self._flush_text()
            self._heading_tag = tag
            self._heading_parts = []
            return
        if self._heading_tag:
            return
        if tag == "br":
            self._text_parts.append(", ")
        elif tag in self._BLOCK_TAGS or tag == "li":
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if self._heading_tag == tag:
            heading_text = " ".join("".join(self._heading_parts).split())
            self.events.append(
                {
                    "type": "heading",
                    "level": int(tag[1]),
                    "title": heading_text,
                }
            )
            self._heading_tag = None
            self._heading_parts = []
            return
        if self._heading_tag:
            return
        if tag == "li":
            self._text_parts.append(".\n")
        elif tag in self._BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._heading_tag:
            self._heading_parts.append(data)
            return
        self._text_parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush_text()


class PiperTTSPlugin(BasePlugin):
    """MkDocs plugin that generates TTS audio for Markdown pages using Piper."""

    config_scheme = (
        ("asset_dir", config_options.Type(str, default="assets/piper-tts")),
        ("audio_dir", config_options.Type(str, default="audio")),
        ("model_dir", config_options.Type(str, default="models/piper-tts")),
        ("languages", config_options.Type(dict, default={})),
        ("button_class", config_options.Type(str, default="piper-tts-button")),
        ("ffmpeg_path", config_options.Type(Path, default=Path("ffmpeg"))),
        ("generate_audio", config_options.Type(bool, default=True)),
        ("use_cuda", config_options.Type(bool, default=False)),
        ("batch_size", config_options.Type(int, default=1)),
    )

    def on_config(self, config: dict) -> None:
        self._timing_started = time.perf_counter()
        generate_audio = os.environ.get("PIPER_TTS_GENERATE_AUDIO")
        if generate_audio is None:
            self._generate_audio = self.config["generate_audio"]
        elif generate_audio.lower() in {"1", "true", "yes", "on"}:
            self._generate_audio = True
        elif generate_audio.lower() in {"0", "false", "no", "off"}:
            self._generate_audio = False
        else:
            raise PluginError("PIPER_TTS_GENERATE_AUDIO must be a boolean value")
        self._page_scan_elapsed = 0.0
        self._eligible_pages = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._hash_elapsed = 0.0
        self._text_elapsed = 0.0
        self._cache_check_elapsed = 0.0
        config_path = getattr(config, "config_file_path", None)
        self._project_dir = Path(config_path).parent.resolve() if config_path else Path.cwd().resolve()
        self._docs_dir = Path(config["docs_dir"])
        if not self._docs_dir.is_absolute():
            self._docs_dir = self._project_dir / self._docs_dir
        self._asset_dir = self.config["asset_dir"].strip("/")
        self._audio_dir = self.config["audio_dir"].strip("/")
        self._model_dir = Path(self.config["model_dir"])
        if not self._model_dir.is_absolute():
            self._model_dir = self._project_dir / self._model_dir
        self._languages = self._configured_languages()
        self._audio_cache_dir = self._docs_dir / self._asset_dir / self._audio_dir
        self._audio_by_page = {}
        self._playlist_by_page = {}
        self._pending_audio = {}
        self._file_hashes = {}
        self._plugin_hash = self._hash_file(Path(__file__))
        self._cache_index = {}
        cache_index_started = time.perf_counter()
        for metadata_path in self._audio_cache_dir.rglob("*.mp3.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            audio_path = metadata_path.with_suffix("")
            if audio_path.is_file() and audio_path.stat().st_size:
                self._cache_index[self._metadata_key(self._cache_lookup_metadata(metadata))] = (
                    audio_path,
                    metadata_path,
                )
        log.info(
            "Piper TTS timing: plugin configured at=%s",
            datetime.now().isoformat(timespec="milliseconds"),
        )
        _log_timing(
            f"audio cache index ({len(self._cache_index)} files)",
            cache_index_started,
            plugin_started=self._timing_started,
        )
        return config

    def on_env(self, env, config, files) -> None:
        env.globals["piper_tts_button"] = self.render_button
        env.globals["piper_tts_playlist"] = self.render_button
        env.globals["piper_tts_reading_time"] = self.render_reading_time
        return env

    def on_page_content(self, html_content: str, *, page, config, files) -> str:
        page_started = time.perf_counter()
        metadata = getattr(page, "meta", {}) or {}
        language = str(metadata.get("lang") or "").lower().split("-", maxsplit=1)[0]
        voice = self._languages.get(language)
        if voice is None or not getattr(page, "file", None):
            return html_content

        self._eligible_pages += 1

        source_path = Path(getattr(page.file, "abs_src_path", ""))
        if not source_path.is_file():
            source_path = self._docs_dir / page.file.src_path
        if not source_path.is_file():
            raise PluginError(f"Piper TTS source Markdown is missing: {source_path}")

        model_path, config_path = self._voice_files(voice, language)
        speaker_id = self._resolve_speaker_id(voice, config_path, language)
        hash_started = time.perf_counter()
        source_hash = self._hash_file(source_path)
        model_hash = self._hash_file(model_path)
        config_hash = self._hash_file(config_path)
        self._hash_elapsed += time.perf_counter() - hash_started
        voice_metadata = {
            "voice_language": language,
            "voice_model": model_path.name,
            "voice_config": config_path.name,
            "voice_speaker_id": speaker_id,
            "voice_model_hash": model_hash,
            "voice_config_hash": config_hash,
            "voice_runtime": "cuda" if self.config["use_cuda"] else "cpu",
        }
        text_started = time.perf_counter()
        sections = self._extract_sections(
            html_content,
            fallback_title=str(getattr(page, "title", "") or "Page"),
        )
        self._text_elapsed += time.perf_counter() - text_started
        section_filenames = self._section_filenames([section["title"] for section in sections])
        playlist_entries = []
        cache_logs = []

        for index, section in enumerate(sections):
            section_text = section["text"]
            section_title = section["title"]
            section_hash = hashlib.sha256(section_text.encode("utf-8")).hexdigest()
            audio_path, metadata_path = self._cache_paths(page.file.src_path, section_filenames[index])
            expected_metadata = {
                "plugin_hash": self._plugin_hash,
                "source_hash": source_hash,
                "section_hash": section_hash,
                "section_index": index,
                "section_count": len(sections),
                "section_title": section_title,
                **voice_metadata,
            }

            cache_started = time.perf_counter()
            cache_valid, cache_reason = self._cache_status(
                audio_path,
                metadata_path,
                expected_metadata,
            )
            if not cache_valid and cache_reason == "audio missing or empty":
                cached_paths = self._cache_index.get(self._metadata_key(expected_metadata))
                if cached_paths is not None:
                    cached_audio_path, cached_metadata_path = cached_paths
                    cache_valid, cache_reason = self._cache_status(
                        cached_audio_path,
                        cached_metadata_path,
                        expected_metadata,
                    )
                    if cache_valid:
                        log.info(
                            "Piper TTS cache reuse: %s [%s] -> %s",
                            page.file.src_path,
                            section_title,
                            cached_audio_path.name,
                        )
                        audio_path = cached_audio_path
                        metadata_path = cached_metadata_path
            self._cache_check_elapsed += time.perf_counter() - cache_started
            if not cache_valid:
                self._cache_misses += 1
                self._pending_audio[audio_path] = {
                    "model_path": model_path,
                    "config_path": config_path,
                    "text": section_text,
                    "speaker_id": speaker_id,
                    "expected_metadata": expected_metadata,
                    "source_path": page.file.src_path,
                    "section_title": section_title,
                }
                cache_logs.append((section_title, cache_reason))
            else:
                self._cache_hits += 1
                log.debug("Reusing cached Piper TTS MP3 for %s [%s]", page.file.src_path, section_title)

            section_metadata = self._load_json_metadata(metadata_path)
            duration_seconds = self._estimate_duration_seconds(section_text)
            if isinstance(section_metadata.get("duration_seconds"), (int, float)):
                duration_seconds = float(section_metadata["duration_seconds"])
            playlist_entries.append(
                {
                    "title": section_title,
                    "audio_path": audio_path,
                    "duration_seconds": duration_seconds,
                }
            )

        self._playlist_by_page[page.file.src_path] = playlist_entries
        self._audio_by_page[page.file.src_path] = playlist_entries[0]["audio_path"] if playlist_entries else None
        self._page_scan_elapsed += time.perf_counter() - page_started
        for section_title, cache_reason in cache_logs:
            log.info(
                "Piper TTS cache miss: %s [%s] reason=%s",
                page.file.src_path,
                section_title,
                cache_reason,
            )
        if cache_logs:
            _log_timing(
                f"cache miss evaluation for {page.file.src_path}",
                page_started,
                plugin_started=self._timing_started,
            )
        return html_content

    def on_post_build(self, config: dict) -> None:
        post_build_started = time.perf_counter()
        log.info(
            "Piper TTS cache summary: eligible_pages=%d cache_hits=%d " "cache_misses=%d pending=%d scan_time=%.3fs",
            self._eligible_pages,
            self._cache_hits,
            self._cache_misses,
            len(self._pending_audio),
            self._page_scan_elapsed,
        )
        log.info(
            "Piper TTS cache timing: file_hashes=%.3fs text_extraction=%.3fs " "cache_validation=%.3fs",
            self._hash_elapsed,
            self._text_elapsed,
            self._cache_check_elapsed,
        )
        if self._pending_audio and not self._generate_audio:
            pending_pages = ", ".join(
                sorted(
                    {
                        self._normalize_pending_task(audio_path, task)["source_path"]
                        for audio_path, task in self._pending_audio.items()
                    }
                )
            )
            raise PluginError(
                "Piper TTS cache-only build found missing or stale audio for: "
                f"{pending_pages}. Generate audio locally and publish an updated runtime asset bundle."
            )
        if self._pending_audio:
            self._generate_pending_audio()
        else:
            log.info(
                "Piper TTS: all %d audio files are cached; skipping " "Piper/ONNX Runtime initialization",
                self._cache_hits,
            )
        _log_timing(
            "post-build audio generation decision",
            post_build_started,
            plugin_started=self._timing_started,
        )

        copy_started = time.perf_counter()
        site_audio_dir = Path(config.site_dir) / self._asset_dir / self._audio_dir
        site_audio_dir.mkdir(parents=True, exist_ok=True)
        referenced_audio_paths = sorted(
            {track["audio_path"] for playlist in self._playlist_by_page.values() for track in playlist},
            key=lambda path: str(path),
        )
        copied = 0
        for audio_path in referenced_audio_paths:
            if not audio_path.is_file() or audio_path.stat().st_size == 0:
                log.warning("Skipping missing or empty Piper TTS MP3 referenced by playlist: %s", audio_path)
                continue
            relative_path = audio_path.relative_to(self._audio_cache_dir)
            destination = site_audio_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio_path, destination)
            copied += 1
        self._remove_stale_audio(site_audio_dir)
        log.info("Copied %d Piper TTS MP3 files to %s", copied, site_audio_dir)
        _log_timing(
            f"copy audio to site ({copied} files)",
            copy_started,
            plugin_started=self._timing_started,
        )

    def _generate_pending_audio(self):
        if not self._pending_audio:
            return

        generation_started = time.perf_counter()
        batch_size = self._batch_size()
        use_cuda = self.config["use_cuda"]
        total = len(self._pending_audio)
        log.info(
            "Generating %d Piper TTS MP3 files with batch size %d%s",
            total,
            batch_size,
            " using CUDA" if use_cuda else "",
        )
        grouped = {}
        for audio_path, task in self._pending_audio.items():
            task = self._normalize_pending_task(audio_path, task)
            key = (str(task["model_path"]), str(task["config_path"]), task["speaker_id"])
            grouped.setdefault(key, []).append(
                {
                    "audio_path": audio_path,
                    "model_path": task["model_path"],
                    "config_path": task["config_path"],
                    "text": task["text"],
                    "speaker_id": task["speaker_id"],
                    "source_path": task["source_path"],
                    "expected_metadata": task["expected_metadata"],
                    "section_title": task["section_title"],
                }
            )
        _log_timing(
            f"group pending audio ({total} files, {len(grouped)} voices)",
            generation_started,
            plugin_started=self._timing_started,
        )

        completed = 0
        started = time.monotonic()

        def report_progress(_audio_path: Path, source_path: str) -> None:
            nonlocal completed
            completed += 1
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed else 0.0
            eta = (total - completed) / rate if rate else 0.0
            log.info(
                "Piper TTS progress: %d/%d (%.1f%%), %.2f files/s, " "ETA %s: %s",
                completed,
                total,
                completed * 100 / total,
                rate,
                self._format_duration(eta),
                source_path,
            )

        for key, group in grouped.items():
            model_path, config_path, _ = key
            try:
                inference_reported = -1

                def report_inference(done: int, total_batches: int, segments_done: int, elapsed: float) -> None:
                    nonlocal inference_reported
                    if not total_batches:
                        return
                    step = max(1, total_batches // 100)
                    if done not in {0, total_batches} and done < inference_reported + step:
                        return
                    inference_reported = done
                    percent = done * 100 / total_batches
                    filled = round(30 * done / total_batches)
                    bar = "#" * filled + "-" * (30 - filled)
                    rate = done / elapsed if elapsed else 0.0
                    eta = (total_batches - done) / rate if rate else 0.0
                    log.info(
                        "Piper TTS inference: [%s] %d/%d batches " "(%.1f%%), %d segments, %.2f batches/s, ETA %s",
                        bar,
                        done,
                        total_batches,
                        percent,
                        segments_done,
                        rate,
                        self._format_duration(eta),
                    )

                voice = _load_voice(
                    Path(model_path),
                    Path(config_path),
                    use_cuda=use_cuda,
                    plugin_started=self._timing_started,
                )
                log.info(
                    "Piper execution providers for %s: %s",
                    model_path,
                    ", ".join(voice.session.get_providers()),
                )
                worker_tasks = [
                    {
                        "audio_path": task["audio_path"],
                        "text": task["text"],
                        "speaker_id": task["speaker_id"],
                        "source_path": task["source_path"],
                    }
                    for task in group
                ]
                worker_elapsed, durations = _generate_audio_batch(
                    worker_tasks,
                    voice,
                    batch_size,
                    self.config["ffmpeg_path"],
                    progress_callback=report_progress,
                    plugin_started=self._timing_started,
                    inference_progress_callback=report_inference,
                )
            except Exception as error:
                source_path = group[0]["source_path"]
                raise PluginError(f"Piper TTS synthesis failed for {source_path}: {error}") from error

            for task in group:
                audio_path = task["audio_path"]
                metadata_path = audio_path.with_suffix(".mp3.json")
                expected_metadata = dict(task["expected_metadata"])
                expected_metadata["duration_seconds"] = round(float(durations.get(str(audio_path), 0.0)), 3)
                metadata_path.write_text(
                    json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self._cache_index[self._metadata_key(task["expected_metadata"])] = (audio_path, metadata_path)
            log.info(
                "Finished Piper TTS batch for %s in %.1fs",
                model_path,
                worker_elapsed,
            )
        self._pending_audio.clear()
        _log_timing(
            f"all pending audio generation ({total} files)",
            generation_started,
            plugin_started=self._timing_started,
        )

    def _batch_size(self) -> int:
        batch_size = self.config["batch_size"]
        if batch_size < 1:
            raise PluginError("Piper TTS batch_size must be one or greater")
        return batch_size

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h{minutes:02d}m"
        if minutes:
            return f"{minutes}m{seconds:02d}s"
        return f"{seconds}s"

    def render_button(self, page: Page | None = None, label: str | None = None) -> Markup:
        if page is None:
            return Markup("")

        source_path = getattr(page.file, "src_path", "")
        playlist = self._playlist_by_page.get(source_path, [])
        if not playlist:
            return Markup("")

        metadata = getattr(page, "meta", {}) or {}
        language = str(metadata.get("lang") or "").lower().split("-", maxsplit=1)[0]
        voice = self._languages.get(language, {})
        audio_label = label or voice.get("label") or language
        playlist_urls = []
        for track in playlist:
            audio_rel_path = self._audio_rel_path(track["audio_path"])
            playlist_urls.append(
                {
                    "title": track["title"],
                    "url": self._relative_url(page, audio_rel_path),
                }
            )

        if not playlist_urls:
            return Markup("")

        player_id = f"piper-tts-{hashlib.sha1(source_path.encode('utf-8')).hexdigest()[:12]}"
        player_container_id = f"{player_id}-container"
        audio_class = self.config["button_class"].strip() or "piper-tts-button"
        playlist_json = json.dumps(playlist_urls)
        list_items = "".join(
            "<li>" f'<button type="button" data-track-index="{index}">' f"{html.escape(str(track['title']))}</button>" "</li>"
            for index, track in enumerate(playlist_urls)
        )
        attributes = {
            "id": player_id,
            "class": audio_class,
            "controls": "",
            "preload": "none",
            "aria-label": audio_label,
            "title": audio_label,
            "src": playlist_urls[0]["url"],
            "data-playlist": playlist_json,
        }
        rendered_attributes = " ".join(
            f'{name}="{html.escape(str(value), quote=True)}"' if value else name for name, value in attributes.items()
        )
        return Markup(
            f'<div id="{player_container_id}" class="{html.escape(audio_class, quote=True)}-wrapper">'
            f"<audio {rendered_attributes}>{html.escape(str(audio_label))}</audio>"
            f'<ol class="{html.escape(audio_class, quote=True)}-playlist">{list_items}</ol>'
            "</div>"
            "<script>"
            "(function () {"
            f"var audio = document.getElementById('{player_id}');"
            f"var container = document.getElementById('{player_container_id}');"
            "if (!audio || !container) { return; }"
            "var playlist;"
            "try { playlist = JSON.parse(audio.dataset.playlist || '[]'); } catch (_error) { return; }"
            "if (!playlist.length) { return; }"
            "var buttons = Array.prototype.slice.call(container.querySelectorAll('button[data-track-index]'));"
            "var currentIndex = 0;"
            "function markActive(index) {"
            "  currentIndex = index;"
            "  buttons.forEach(function (button, buttonIndex) {"
            "    var active = buttonIndex === index;"
            "    button.setAttribute('aria-current', active ? 'true' : 'false');"
            "  });"
            "}"
            "function loadTrack(index, autoplay) {"
            "  if (index < 0 || index >= playlist.length) { return; }"
            "  markActive(index);"
            "  if (audio.getAttribute('src') !== playlist[index].url) {"
            "    audio.setAttribute('src', playlist[index].url);"
            "    audio.load();"
            "  }"
            "  if (autoplay) {"
            "    var playPromise = audio.play();"
            "    if (playPromise && typeof playPromise.catch === 'function') {"
            "      playPromise.catch(function () {});"
            "    }"
            "  }"
            "}"
            "buttons.forEach(function (button, index) {"
            "  button.addEventListener('click', function () { loadTrack(index, true); });"
            "});"
            "audio.addEventListener('ended', function () {"
            "  if (currentIndex + 1 < playlist.length) {"
            "    loadTrack(currentIndex + 1, true);"
            "  }"
            "});"
            "markActive(0);"
            "})();"
            "</script>"
        )

    def render_reading_time(self, page: Page | None = None, label: str = "approximate reading time:") -> Markup:
        if page is None:
            return Markup("")

        source_path = getattr(page.file, "src_path", "")
        playlist = self._playlist_by_page.get(source_path, [])
        if not playlist:
            return Markup("")

        total_duration = sum(max(0.0, float(track.get("duration_seconds") or 0.0)) for track in playlist)
        if total_duration <= 0:
            return Markup("")

        rendered_label = html.escape(str(label).strip() or "approximate reading time:")
        return Markup(
            '<span class="piper-tts-reading-time">' f"{rendered_label} {self._format_duration(total_duration)}" "</span>"
        )

    def _audio_rel_path(self, audio_path: Path) -> str:
        relative_path = audio_path.relative_to(self._audio_cache_dir).as_posix()
        return posixpath.join(self._asset_dir, self._audio_dir, relative_path)

    def _configured_languages(self) -> dict[str, dict]:
        languages = {language: dict(voice) for language, voice in DEFAULT_LANGUAGES.items()}
        for language, configured_voice in (self.config["languages"] or {}).items():
            normalized_language = str(language).lower().split("-", maxsplit=1)[0]
            voice = dict(configured_voice or {})
            if "model" not in voice:
                raise PluginError(f"Piper TTS language {language!r} must define a model")
            languages[normalized_language] = voice

        for voice in languages.values():
            model = Path(str(voice["model"]))
            voice.setdefault("config", f"{model.name}.json")
        return languages

    def _voice_files(self, voice: dict, language: str) -> tuple[Path, Path]:
        model_value = Path(str(voice["model"]))
        model_path = model_value if model_value.is_absolute() else self._model_dir / model_value
        config_value = Path(str(voice["config"]))
        config_path = config_value if config_value.is_absolute() else self._model_dir / config_value
        missing_paths = [path for path in (model_path, config_path) if not path.is_file()]
        if missing_paths:
            locations = "\n".join(f"  - {path}" for path in (model_path, config_path))
            download_url = str(voice.get("download_url") or "").strip()
            download_hint = ""
            if download_url:
                download_hint = "\nDownload the matching files from:\n" f"  - {download_url}\n" f"  - {download_url}.json"
            raise PluginError(
                f"Piper TTS: there is no ONNX voice model and/or JSON configuration for "
                f"language {language!r}.\nModel directory: {self._model_dir}\n"
                f"Required files:\n{locations}{download_hint}"
            )
        return model_path.resolve(), config_path.resolve()

    def _resolve_speaker_id(self, voice: dict, config_path: Path, language: str) -> int | None:
        speaker = voice.get("speaker")
        if speaker is None:
            return None

        try:
            speaker_map = json.loads(config_path.read_text(encoding="utf-8"))["speaker_id_map"]
        except (OSError, json.JSONDecodeError, KeyError) as error:
            raise PluginError(f"Could not read Piper speaker map for language {language!r}") from error

        if str(speaker) in speaker_map:
            return int(speaker_map[str(speaker)])
        try:
            speaker_id = int(speaker)
        except (TypeError, ValueError) as error:
            raise PluginError(f"Unknown Piper speaker {speaker!r} for language {language!r}") from error

        if speaker_id not in speaker_map.values():
            raise PluginError(f"Unknown Piper speaker {speaker!r} for language {language!r}")
        return speaker_id

    @staticmethod
    def _extract_text(html_content: str) -> str:
        parser = _PageTextExtractor()
        parser.feed(html_content)
        parser.close()
        return PiperTTSPlugin._normalize_extracted_text("".join(parser.parts))

    @staticmethod
    def _normalize_extracted_text(text: str) -> str:
        def paragraph_pause(match):
            preceding_text = match.group(1).rstrip()
            if preceding_text.endswith((".", "!", "?")):
                return preceding_text + " "
            return preceding_text + ". "

        text = re.sub(r"(\S)\s*\n[ \t]*\n+", paragraph_pause, text)
        return " ".join(text.split())

    def _extract_sections(self, html_content: str, *, fallback_title: str) -> list[dict]:
        parser = _PageSectionExtractor()
        parser.feed(html_content)
        parser.close()
        events = parser.events
        if not events:
            return []

        sections = []
        h1_indices = [
            index
            for index, event in enumerate(events)
            if event["type"] == "heading" and event.get("level") == 1 and event.get("title")
        ]

        if len(h1_indices) > 1:
            for index, start in enumerate(h1_indices):
                end = h1_indices[index + 1] if index + 1 < len(h1_indices) else len(events)
                title = str(events[start].get("title") or f"{fallback_title} part {index + 1}")
                text = self._events_to_text(events[start:end])
                if text:
                    sections.append({"title": title, "text": text})
            return sections

        if len(h1_indices) == 1:
            h1_start = h1_indices[0]
            h1_title = str(events[h1_start].get("title") or fallback_title)
            h2_indices = [
                index
                for index, event in enumerate(events[h1_start + 1 :], start=h1_start + 1)
                if event["type"] == "heading" and event.get("level") == 2 and event.get("title")
            ]
            if h2_indices:
                intro_text = self._events_to_text(events[h1_start : h2_indices[0]])
                if intro_text:
                    sections.append({"title": f"(Intro) {h1_title}", "text": intro_text})
                for index, start in enumerate(h2_indices):
                    end = h2_indices[index + 1] if index + 1 < len(h2_indices) else len(events)
                    title = str(events[start].get("title") or f"{h1_title} part {index + 1}")
                    text = self._events_to_text(events[start:end])
                    if text:
                        sections.append({"title": title, "text": text})
                return sections

            total_text = self._events_to_text(events[h1_start:])
            if total_text:
                return [{"title": h1_title, "text": total_text}]

        fallback_text = self._events_to_text(events)
        if fallback_text:
            return [{"title": fallback_title, "text": fallback_text}]
        return []

    def _events_to_text(self, events: list[dict]) -> str:
        parts = []
        for event in events:
            if event["type"] == "heading":
                heading_title = str(event.get("title") or "").strip()
                if heading_title:
                    parts.append(f"\n{heading_title}.\n")
                continue
            parts.append(str(event.get("raw") or ""))
        return self._normalize_extracted_text("".join(parts))

    @staticmethod
    def _slugify_title(title: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", title.lower()).strip("-")
        return slug or "section"

    def _section_filenames(self, section_titles: list[str]) -> list[str]:
        counts = {}
        filenames = []
        for title in section_titles:
            base_slug = self._slugify_title(title)
            counts[base_slug] = counts.get(base_slug, 0) + 1
            if counts[base_slug] == 1:
                filenames.append(base_slug)
            else:
                filenames.append(f"{base_slug}-{counts[base_slug]}")
        return filenames

    @staticmethod
    def _estimate_duration_seconds(text: str) -> float:
        words = len(re.findall(r"\b\w+\b", text))
        if not words:
            return 0.0
        # Conservative estimate for speech pace before exact durations exist.
        return (words / 160.0) * 60.0

    @staticmethod
    def _source_slug(source_rel_path: str) -> str:
        source_path = Path(source_rel_path).with_suffix("").as_posix().strip("/")
        if not source_path:
            return "page"
        parts = [quote(part, safe="-_.~") for part in source_path.split("/")]
        return "/".join(part or "page" for part in parts)

    def _cache_paths(self, source_rel_path: str, section_slug: str) -> tuple[Path, Path]:
        source_slug = self._source_slug(source_rel_path)
        audio_path = self._audio_cache_dir / source_slug / f"{section_slug}.mp3"
        return audio_path, audio_path.with_suffix(".mp3.json")

    @staticmethod
    def _load_json_metadata(metadata_path: Path) -> dict:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return metadata if isinstance(metadata, dict) else {}

    def _normalize_pending_task(self, audio_path: Path, task) -> dict:
        if not isinstance(task, dict):
            raise PluginError(f"Piper TTS pending task for {audio_path} has unsupported format")
        return {
            "model_path": Path(task["model_path"]),
            "config_path": Path(task["config_path"]),
            "text": str(task["text"]),
            "speaker_id": task.get("speaker_id"),
            "expected_metadata": dict(task.get("expected_metadata") or {}),
            "source_path": str(task.get("source_path") or audio_path.name),
            "section_title": str(task.get("section_title") or "section"),
        }

    @staticmethod
    def _metadata_key(metadata):
        return json.dumps(metadata, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _cache_lookup_metadata(metadata: dict) -> dict:
        normalized = dict(metadata)
        normalized.pop("duration_seconds", None)
        return normalized

    def _cache_status(self, audio_path: Path, metadata_path: Path, expected: dict) -> tuple[bool, str]:
        if not audio_path.is_file() or audio_path.stat().st_size == 0:
            return False, "audio missing or empty"
        try:
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False, "metadata missing or invalid"
        if not isinstance(actual, dict):
            return False, "metadata is not an object"
        if all(actual.get(key) == value for key, value in expected.items()):
            return True, "valid"
        changed = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        return False, f"metadata mismatch ({', '.join(changed)})"

    def _remove_stale_audio(self, site_audio_dir: Path) -> None:
        current_paths = {
            track["audio_path"].relative_to(self._audio_cache_dir).as_posix()
            for playlist in self._playlist_by_page.values()
            for track in playlist
        }
        for published_path in site_audio_dir.rglob("*.mp3"):
            published_relative = published_path.relative_to(site_audio_dir).as_posix()
            if published_relative in current_paths:
                continue
            published_path.unlink(missing_ok=True)

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._file_hashes[str(path.resolve())] = value
        return value

    def _relative_url(self, page: Page, target_path: str) -> str:
        page_url = "/" + str(getattr(page, "url", "") or "").lstrip("/")
        if page_url == "/" or page_url.endswith("/"):
            page_directory = page_url
        else:
            page_directory = posixpath.dirname(page_url) + "/"
        return posixpath.relpath(f"/{target_path}", page_directory)
