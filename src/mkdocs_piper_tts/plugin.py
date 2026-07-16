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
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

from markupsafe import Markup
from mkdocs.config import config_options
from mkdocs.exceptions import PluginError
from mkdocs.plugins import BasePlugin, get_plugin_logger


log = get_plugin_logger(__name__)

DEFAULT_LANGUAGES = {
    "de": {"model": "de_DE-thorsten-medium.onnx", "label": "Vorlesen"},
    "en": {"model": "en_US-lessac-medium.onnx", "label": "Listen"},
}


def _log_timing(label, started, *, plugin_started=None, level="info"):
    """Log wall-clock and monotonic timing so build phases can be correlated."""
    elapsed = time.perf_counter() - started
    since_plugin = ""
    if plugin_started is not None:
        since_plugin = f", since_plugin={time.perf_counter() - plugin_started:.3f}s"
    message = "Piper TTS timing: %s at=%s duration=%.3fs%s" % (
        label,
        datetime.now().isoformat(timespec="milliseconds"),
        elapsed,
        since_plugin,
    )
    getattr(log, level)(message)


def _log_duration(label, duration, *, plugin_started=None):
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
    model_path,
    config_path,
    *,
    use_cuda,
    plugin_started=None,
):
    """Load Piper with either the CUDA or CPU execution provider."""
    started = time.perf_counter()
    import onnxruntime
    from piper import PiperVoice
    from piper.config import PiperConfig

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


def _audio_from_batch(voice, phoneme_ids, speaker_id, timing=None):
    """Run one padded batch and return one trimmed float waveform per item."""
    import numpy as np

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
            if window > 1:
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
    tasks,
    voice,
    batch_size,
    ffmpeg_path,
    progress_callback=None,
    plugin_started=None,
    inference_progress_callback=None,
):
    """Generate and encode a group of pages sharing one Piper model/voice."""
    import numpy as np

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_dir = Path(temporary_dir)
        wav_paths = [temporary_dir / f"{index}.wav" for index in range(len(tasks))]
        wav_files = []
        segments = []
        phonemize_started = time.perf_counter()
        for index, (_, _, _, text, speaker_id, _) in enumerate(tasks):
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
        for index, (audio_path, _, _, _, _, _) in enumerate(tasks):
            audio_path = Path(audio_path)
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
            if progress_callback is not None:
                progress_callback(audio_path)
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
    return time.perf_counter() - started


class _PageTextExtractor(HTMLParser):

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.ignored_depth += 1
        elif not self.ignored_depth and tag in {"br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif not self.ignored_depth and tag in {"h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            self.parts.append(".\n")
        elif not self.ignored_depth and tag in {"div", "li", "p", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.ignored_depth:
            self.parts.append(data)


class PiperTTSPlugin(BasePlugin):
    config_scheme = (
        ("asset_dir", config_options.Type(str, default="assets/piper-tts")),
        ("audio_dir", config_options.Type(str, default="audio")),
        ("model_dir", config_options.Type(str, default="models/piper-tts")),
        ("languages", config_options.Type(dict, default={})),
        ("button_class", config_options.Type(str, default="piper-tts-button")),
        ("ffmpeg_path", config_options.Type(str, default="ffmpeg")),
        ("generate_audio", config_options.Type(bool, default=True)),
        ("use_cuda", config_options.Type(bool, default=False)),
        ("batch_size", config_options.Type(int, default=1)),
    )

    def on_config(self, config):
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
        self._pending_audio = {}
        self._file_hashes = {}
        self._plugin_hash = self._hash_file(Path(__file__))
        self._cache_index = {}
        cache_index_started = time.perf_counter()
        for metadata_path in self._audio_cache_dir.glob("*.mp3.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            audio_path = metadata_path.with_suffix("")
            if audio_path.is_file() and audio_path.stat().st_size:
                self._cache_index[self._metadata_key(metadata)] = (
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

    def on_env(self, env, config, files):
        env.globals["piper_tts_button"] = self.render_button
        return env

    def on_page_content(self, html_content, *, page, config, files):
        page_started = time.perf_counter()
        metadata = getattr(page, "meta", {}) or {}
        language = str(metadata.get("lang") or "").lower().split("-")[0]
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
        self._hash_elapsed += time.perf_counter() - hash_started
        text_started = time.perf_counter()
        text = self._extract_text(html_content)
        self._text_elapsed += time.perf_counter() - text_started
        audio_path, metadata_path = self._cache_paths(page.file.src_path, source_hash)
        expected_metadata = {
            "plugin_hash": self._plugin_hash,
            "source_hash": source_hash,
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
                        "Piper TTS cache reuse: %s -> %s",
                        page.file.src_path,
                        cached_audio_path.name,
                    )
                    audio_path = cached_audio_path
                    metadata_path = cached_metadata_path
        self._cache_check_elapsed += time.perf_counter() - cache_started
        if not cache_valid:
            self._cache_misses += 1
            self._pending_audio[audio_path] = (
                model_path,
                config_path,
                text,
                speaker_id,
                expected_metadata,
                page.file.src_path,
            )
        else:
            self._cache_hits += 1
            log.debug("Reusing cached Piper TTS MP3 for %s", page.file.src_path)

        self._audio_by_page[page.file.src_path] = audio_path
        self._page_scan_elapsed += time.perf_counter() - page_started
        if not cache_valid:
            log.info(
                "Piper TTS cache miss: %s reason=%s",
                page.file.src_path,
                cache_reason,
            )
            _log_timing(
                f"cache miss evaluation for {page.file.src_path}",
                page_started,
                plugin_started=self._timing_started,
            )
        return html_content

    def on_post_build(self, config):
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
            pending_pages = ", ".join(sorted(task[5] for task in self._pending_audio.values()))
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
        copied = 0
        for audio_path in self._audio_cache_dir.glob("*.mp3"):
            if audio_path.stat().st_size:
                shutil.copy2(audio_path, site_audio_dir / audio_path.name)
                copied += 1
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
        for audio_path, (
            model_path,
            config_path,
            text,
            speaker_id,
            metadata,
            source_path,
        ) in self._pending_audio.items():
            key = (str(model_path), str(config_path), speaker_id)
            grouped.setdefault(key, []).append(
                (
                    str(audio_path),
                    model_path,
                    config_path,
                    text,
                    speaker_id,
                    source_path,
                    metadata,
                )
            )
        _log_timing(
            f"group pending audio ({total} files, {len(grouped)} voices)",
            generation_started,
            plugin_started=self._timing_started,
        )

        completed = 0
        started = time.monotonic()

        def report_progress(audio_path):
            nonlocal completed
            source_path = next(task[5] for task in group if str(audio_path) == task[0])
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

        for (model_path, config_path, speaker_id), group in grouped.items():
            try:
                inference_reported = -1

                def report_inference(done, total_batches, segments_done, elapsed):
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
                    model_path,
                    config_path,
                    use_cuda=use_cuda,
                    plugin_started=self._timing_started,
                )
                log.info(
                    "Piper execution providers for %s: %s",
                    model_path,
                    ", ".join(voice.session.get_providers()),
                )
                # The batch worker only needs the first six fields; metadata is
                # written by the parent after the MP3 has been created.
                worker_tasks = [task[:6] for task in group]
                worker_elapsed = _generate_audio_batch(
                    worker_tasks,
                    voice,
                    batch_size,
                    self.config["ffmpeg_path"],
                    progress_callback=report_progress,
                    plugin_started=self._timing_started,
                    inference_progress_callback=report_inference,
                )
            except Exception as error:
                source_path = group[0][5]
                raise PluginError(f"Piper TTS synthesis failed for {source_path}: {error}") from error

            for audio_path, _, _, _, _, source_path, metadata in group:
                Path(audio_path).with_suffix(".mp3.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            log.info(
                "Finished Piper TTS batch for %s in %.1fs",
                model_path,
                worker_elapsed,
            )
        _log_timing(
            f"all pending audio generation ({total} files)",
            generation_started,
            plugin_started=self._timing_started,
        )

    def _batch_size(self):
        batch_size = self.config["batch_size"]
        if batch_size < 1:
            raise PluginError("Piper TTS batch_size must be one or greater")
        return batch_size

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h{minutes:02d}m"
        if minutes:
            return f"{minutes}m{seconds:02d}s"
        return f"{seconds}s"

    def render_button(self, page=None, label=None):
        if page is None:
            return Markup("")

        audio_path = self._audio_by_page.get(getattr(page.file, "src_path", ""))
        if audio_path is None:
            return Markup("")

        metadata = getattr(page, "meta", {}) or {}
        language = str(metadata.get("lang") or "").lower().split("-")[0]
        voice = self._languages.get(language, {})
        audio_label = label or voice.get("label") or language
        audio_rel_path = posixpath.join(self._asset_dir, self._audio_dir, audio_path.name)
        audio_url = self._relative_url(page, audio_rel_path)
        audio_class = self.config["button_class"].strip() or "piper-tts-button"
        attributes = {
            "class": audio_class,
            "controls": "",
            "preload": "none",
            "aria-label": audio_label,
            "title": audio_label,
        }
        rendered_attributes = " ".join(
            f'{name}="{html.escape(str(value), quote=True)}"' if value else name for name, value in attributes.items()
        )
        return Markup(
            f"<audio {rendered_attributes}>"
            f'<source src="{html.escape(audio_url, quote=True)}" type="audio/mpeg">'
            f"{html.escape(str(audio_label))}</audio>"
        )

    def _configured_languages(self):
        languages = {language: dict(voice) for language, voice in DEFAULT_LANGUAGES.items()}
        for language, configured_voice in (self.config["languages"] or {}).items():
            normalized_language = str(language).lower().split("-")[0]
            voice = dict(configured_voice or {})
            if "model" not in voice:
                raise PluginError(f"Piper TTS language {language!r} must define a model")
            languages[normalized_language] = voice

        for voice in languages.values():
            model = Path(str(voice["model"]))
            voice.setdefault("config", f"{model.name}.json")
        return languages

    def _voice_files(self, voice, language):
        model_value = Path(str(voice["model"]))
        model_path = model_value if model_value.is_absolute() else self._model_dir / model_value
        config_value = Path(str(voice["config"]))
        config_path = config_value if config_value.is_absolute() else self._model_dir / config_value
        if not model_path.is_file() or not config_path.is_file():
            raise PluginError(f"Piper TTS model for language {language!r} is missing: " f"{model_path} and {config_path}")
        return model_path.resolve(), config_path.resolve()

    def _resolve_speaker_id(self, voice, config_path, language):
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

    def _extract_text(self, html_content):
        parser = _PageTextExtractor()
        parser.feed(html_content)
        parser.close()
        return " ".join("".join(parser.parts).split())

    def _cache_paths(self, source_rel_path, source_hash):
        source_path = Path(source_rel_path)
        source_id = hashlib.sha256(source_rel_path.encode("utf-8")).hexdigest()[:12]
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", source_path.with_suffix("").as_posix())
        slug = slug.strip("-")[-80:] or "page"
        filename = f"{slug}-{source_id}-{source_hash[:12]}.mp3"
        audio_path = self._audio_cache_dir / filename
        return audio_path, audio_path.with_suffix(".mp3.json")

    @staticmethod
    def _metadata_key(metadata):
        return json.dumps(metadata, sort_keys=True, separators=(",", ":"))

    def _cache_status(self, audio_path, metadata_path, expected):
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

    def _remove_stale_audio(self, site_audio_dir):
        current_paths = set(self._audio_by_page.values())
        for published_path in site_audio_dir.glob("*.mp3"):
            if published_path.name in {path.name for path in current_paths}:
                continue
            published_path.unlink(missing_ok=True)

    def _hash_file(self, path):
        cache_key = str(path)
        if cache_key in self._file_hashes:
            return self._file_hashes[cache_key]
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._file_hashes[cache_key] = value
        return value

    def _relative_url(self, page, target_path):
        page_url = "/" + str(getattr(page, "url", "") or "").lstrip("/")
        if page_url == "/" or page_url.endswith("/"):
            page_directory = page_url
        else:
            page_directory = posixpath.dirname(page_url) + "/"
        return posixpath.relpath(f"/{target_path}", page_directory)
