Updated todo list

Reviewing the library now with a code-audit focus, starting by reading the core plugin implementation, tests, and any existing repo memory notes so I can report concrete bugs, design flaws, and hacky patterns with precise locations.

I am gathering baseline context first by checking stored repo notes and scanning the main source files so the review targets real code risks instead of generic feedback.

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#1-1), lines 1 to 420

Read memory

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/__init__.py)

Read memory [](file:///home/wer/.vscode-server/data/User/workspaceStorage/cd07255db0ab23632f8f3cf5ce015805/GitHub.copilot-chat/memory-tool/memories/repo/plugin-notes.md)

I have the main implementation loaded and I am now reading the remainder plus tests to map concrete findings to observed behavior and coverage gaps.

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#421-421), lines 421 to 920

Read [](file:///home/wer/Code/mkdocs_piper_tts/tests/test_plugin_unit.py)

Read [](file:///home/wer/Code/mkdocs_piper_tts/tests/test_plugin_unit.py#348-348), lines 348 to 348

Read [](file:///home/wer/Code/mkdocs_piper_tts/tests/test_text_extraction.py)

Read [](file:///home/wer/Code/mkdocs_piper_tts/tests/test_example_e2e.py)

Next I am pulling the rest of the plugin source and project metadata so I can verify whether the risky patterns are isolated or systemic.

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#900-900), lines 900 to 1237

Read [](file:///home/wer/Code/mkdocs_piper_tts/pyproject.toml)

Read [](file:///home/wer/Code/mkdocs_piper_tts/README.md)

I am running the existing test suite now to surface behavior regressions and confirm whether likely flaws are already covered or currently slipping through.

23/23 tests passed (100%)

Tests are green, so I am now checking diagnostics and deeper code paths for latent bugs and architectural issues that tests can miss.

Checked workspace, 94 problems found in [](file:///home/wer/Code/mkdocs_piper_tts/tests/test_text_extraction.py), [](file:///home/wer/Code/mkdocs_piper_tts/tests/test_example_e2e.py), [](file:///home/wer/Code/mkdocs_piper_tts/tests/test_plugin_unit.py), [](vscode-terminal:/terminal3.py)

I found a potentially critical playlist serialization issue and will quickly validate it in the configured Python environment so the report includes confirmed behavior, not speculation.

Configuring a Python Environment
Completed with input: {
  "resourcePath": "/home/wer/Code/mkdocs_piper_tts"
}

I am running a minimal runtime check of the rendered HTML attribute to confirm whether the playlist JSON is parseable in the browser.

Ran terminal command:  /home/wer/Code/mkdocs_piper_tts/.venv/bin/python - <<'PY'
from types import SimpleNamespace
from pathlib import Path
import re, html, json
from mkdocs_piper_tts.plugin import PiperTTSPlugin

plugin = PiperTTSPlugin()
plugin.config = {
    "asset_dir": "assets/piper-tts",
    "audio_dir": "audio",
    "model_dir": "models",
    "languages": {},
    "button_class": "piper-tts-button",
    "ffmpeg_path": Path("ffmpeg"),
    "generate_audio": True,
    "use_cuda": False,
    "batch_size": 1,
}
plugin._audio_cache_dir = Path("/tmp/audio-cache")
plugin._playlist_by_page = {
    "p.md": [{"title": "A", "audio_path": Path("/tmp/audio-cache/p/intro.mp3"), "duration_seconds": 1.0}]
}
plugin._languages = {"en": {"label": "Listen"}}
page = SimpleNamespace(file=SimpleNamespace(src_path="p.md"), meta={"lang": "en"}, url="p/")
rendered = str(plugin.render_button(page))
m = re.search(r'data-playlist="([^"]*)"', rendered)
print("attr:", m.group(1))
val = html.unescape(m.group(1))
print("decoded once:", val)
try:
    obj = json.loads(val)
    print("json ok", obj)
except Exception as e:
    print("json fail", type(e).__name__, e)
PY

The first probe failed because the plugin state was not initialized, so I am rerunning with a proper on_config setup to get a valid result.

Ran terminal command:  /home/wer/Code/mkdocs_piper_tts/.venv/bin/python - <<'PY'
from types import SimpleNamespace
from pathlib import Path
import re, html, json, tempfile
from mkdocs_piper_tts.plugin import PiperTTSPlugin

with tempfile.TemporaryDirectory() as d:
    docs = Path(d) / "docs"
    docs.mkdir()
    cfg = {"docs_dir": str(docs)}
    cfg = type("Cfg", (dict,), {"config_file_path": str(Path(d)/"mkdocs.yml")})(cfg)

    plugin = PiperTTSPlugin()
    plugin.config = {
        "asset_dir": "assets/piper-tts",
        "audio_dir": "audio",
        "model_dir": "models",
        "languages": {},
        "button_class": "piper-tts-button",
        "ffmpeg_path": Path("ffmpeg"),
        "generate_audio": True,
        "use_cuda": False,
        "batch_size": 1,
    }
    plugin.on_config(cfg)
    plugin._playlist_by_page = {
        "p.md": [{"title": "A", "audio_path": plugin._audio_cache_dir / "p" / "intro.mp3", "duration_seconds": 1.0}]
    }
    plugin._languages = {"en": {"label": "Listen"}}
    page = SimpleNamespace(file=SimpleNamespace(src_path="p.md"), meta={"lang": "en"}, url="p/")
    rendered = str(plugin.render_button(page))
    m = re.search(r'data-playlist="([^"]*)"', rendered)
    print("attr:", m.group(1))
    val = html.unescape(m.group(1))
    print("decoded once:", val)
    try:
        obj = json.loads(val)
        print("json ok", obj)
    except Exception as e:
        print("json fail", type(e).__name__, e)
PY

I am collecting exact line references for each candidate finding so the summary can be actionable and easy to verify.

Searched for regex `playlist_json = html.escape\(json.dumps\(playlist_urls\), quote=True\)` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `rendered_attributes = " ".join\(` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `JSON.parse\(audio.dataset.playlist \|\| '\[\]'\)` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `expected_metadata = \{` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `def _cache_paths\(` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for text `source_slug = re.sub` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `def _hash_file\(` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for text `if cache_key in self._file_hashes` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `for audio_path in self._audio_cache_dir.rglob\("\*\.mp3"\)` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `def _remove_stale_audio\(` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

I am verifying whether the stale-audio cleanup helper is actually wired into the build flow or left as dead code.

Searched for regex `_remove_stale_audio\(` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `self._plugin_hash = self._hash_file\(Path\(__file__\)\)` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for text `cache_valid, cache_reason = self._cache_status` (`**/src/mkdocs_piper_tts/plugin.py`), 2 results

Searched for regex `self._cache_index\[self._metadata_key\(task\["expected_metadata"\]\)\]` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for text `duration_seconds` (`**/src/mkdocs_piper_tts/plugin.py`), 8 results

Searched for regex `self._pending_audio\[audio_path\] = \{` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `source_slug = source_slug.strip\("/-"\) or "page"` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `audio_path = self._audio_cache_dir / source_slug / f"\{section_slug\}\.mp3"` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

I am checking repository hygiene next to identify any generated-artifact or release-process hacks that can cause drift from source.

Ran terminal command:  git -C /home/wer/Code/mkdocs_piper_tts status --short

Progress update: I confirmed the working tree is already dirty and I am keeping this review strictly read-only while I inspect integration templates for additional flaws.

Read [](file:///home/wer/Code/mkdocs_piper_tts/examples/simple-site/overrides/main.html)

I am doing a quick pass over the utility scripts to catch release or deployment shortcuts that could undermine reliability beyond core runtime code.

Read [](file:///home/wer/Code/mkdocs_piper_tts/scripts/build_example_pages.py)

Read [](file:///home/wer/Code/mkdocs_piper_tts/scripts/release_package.py)

Read [](file:///home/wer/Code/mkdocs_piper_tts/scripts/example_voice_asset.py)

I am validating another high-risk scenario: whether changing the configured voice model fails to invalidate cached audio for unchanged page text.

Ran terminal command:  /home/wer/Code/mkdocs_piper_tts/.venv/bin/python - <<'PY'
from pathlib import Path
from types import SimpleNamespace
import json, tempfile
from mkdocs_piper_tts.plugin import PiperTTSPlugin

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    docs = root / "docs"
    docs.mkdir()
    source = docs / "page.md"
    source.write_text("---\nlang: en\n---\n# Hello\nBody", encoding="utf-8")

    cfg = {"docs_dir": str(docs)}
    cfg = type("Cfg", (dict,), {"config_file_path": str(root/"mkdocs.yml")})(cfg)

    model_dir = root / "models"
    model_dir.mkdir()
    for name in ["voiceA.onnx", "voiceA.onnx.json", "voiceB.onnx", "voiceB.onnx.json"]:
        p = model_dir / name
        if p.suffix == ".json":
            p.write_text("{}", encoding="utf-8")
        else:
            p.write_bytes(b"x")

    plugin = PiperTTSPlugin()
    plugin.config = {
        "asset_dir": "assets/piper-tts",
        "audio_dir": "audio",
        "model_dir": str(model_dir),
        "languages": {},
        "button_class": "piper-tts-button",
        "ffmpeg_path": Path("ffmpeg"),
        "generate_audio": True,
        "use_cuda": False,
        "batch_size": 1,
    }
    plugin.on_config(cfg)

    plugin._languages = {"en": {"model": "voiceA.onnx", "config": "voiceA.onnx.json"}}
    page = SimpleNamespace(meta={"lang": "en"}, file=SimpleNamespace(abs_src_path=str(source), src_path="page.md"), title="Hello")

    html = "<h1>Hello</h1><p>Body</p>"
    plugin.on_page_content(html, page=page, config=cfg, files=[])
    audio_path, task = next(iter(plugin._pending_audio.items()))
    metadata = dict(task["expected_metadata"])
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"mp3")
    audio_path.with_suffix(".mp3.json").write_text(json.dumps(metadata), encoding="utf-8")
    plugin._pending_audio.clear()
    plugin._cache_hits = 0
    plugin._cache_misses = 0

    plugin._languages = {"en": {"model": "voiceB.onnx", "config": "voiceB.onnx.json"}}
    plugin.on_page_content(html, page=page, config=cfg, files=[])

    print("cache_hits", plugin._cache_hits)
    print("cache_misses", plugin._cache_misses)
    print("pending", len(plugin._pending_audio))
PY

I am checking path-slug collisions now to confirm whether distinct source files can map to the same cache path and overwrite each other.

Ran terminal command:  /home/wer/Code/mkdocs_piper_tts/.venv/bin/python - <<'PY'
from pathlib import Path
import tempfile
from mkdocs_piper_tts.plugin import PiperTTSPlugin

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    docs = root / "docs"
    docs.mkdir()
    cfg = {"docs_dir": str(docs)}
    cfg = type("Cfg", (dict,), {"config_file_path": str(root/"mkdocs.yml")})(cfg)
    plugin = PiperTTSPlugin()
    plugin.config = {
        "asset_dir": "assets/piper-tts",
        "audio_dir": "audio",
        "model_dir": "models",
        "languages": {},
        "button_class": "piper-tts-button",
        "ffmpeg_path": Path("ffmpeg"),
        "generate_audio": True,
        "use_cuda": False,
        "batch_size": 1,
    }
    plugin.on_config(cfg)
    a,_ = plugin._cache_paths("guide/Über.md", "intro")
    b,_ = plugin._cache_paths("guide/ber.md", "intro")
    print(a)
    print(b)
    print("collision", a == b)
PY

I found one collision test that did not reproduce, so I am running a broader slug probe with non-Latin filenames where the normalization is most lossy.

Ran terminal command:  /home/wer/Code/mkdocs_piper_tts/.venv/bin/python - <<'PY'
from pathlib import Path
import tempfile
from mkdocs_piper_tts.plugin import PiperTTSPlugin

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    docs = root / "docs"
    docs.mkdir()
    cfg = {"docs_dir": str(docs)}
    cfg = type("Cfg", (dict,), {"config_file_path": str(root/"mkdocs.yml")})(cfg)
    plugin = PiperTTSPlugin()
    plugin.config = {
        "asset_dir": "assets/piper-tts",
        "audio_dir": "audio",
        "model_dir": "models",
        "languages": {},
        "button_class": "piper-tts-button",
        "ffmpeg_path": Path("ffmpeg"),
        "generate_audio": True,
        "use_cuda": False,
        "batch_size": 1,
    }
    plugin.on_config(cfg)
    for src in ["你好.md", "こんにちは.md", "🙂.md", "--.md", "___.md"]:
        audio,_ = plugin._cache_paths(src, "intro")
        print(src, "=>", audio.relative_to(plugin._audio_cache_dir))
PY

I have several concrete findings now; I am collecting exact line anchors for each one so the final review is severity-ranked and directly actionable.

Searched for regex `normalized_language = str\(language\)\.lower\(\)\.split\("-", maxsplit=1\)\[0\]` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `language = str\(metadata.get\("lang"\) or ""\)\.lower\(\)\.split\("-", maxsplit=1\)\[0\]` (`**/src/mkdocs_piper_tts/plugin.py`), 2 results

Searched for text `if cache_key in self._file_hashes` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `self._file_hashes = \{\}` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `def _extract_text\(html_content: str, maybe_html_content: str \| None = None\)` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `if maybe_html_content is not None:\n\s*html_content = maybe_html_content` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `def _normalize_pending_task\(` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Searched for regex `if isinstance\(task, tuple\) and len\(task\) >= 6` (`**/src/mkdocs_piper_tts/plugin.py`), 1 result

Ran terminal command:  git -C /home/wer/Code/mkdocs_piper_tts ls-files build | sed -n '1,40p'

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Searched for text `def test_render_button_urls_escapes_labels_and_registers_template_helper` (`**/tests/test_plugin_unit.py`), 1 result

Searched for text `assert "data-playlist" in rendered` (`**/tests/test_plugin_unit.py`), 1 result

Searched for text `def test_cache_paths_status_and_hashing` (`**/tests/test_plugin_unit.py`), 1 result

Searched for regex `first_hash = plugin._hash_file\(source\)|source.write_text\("changed"|assert plugin._hash_file\(source\) == first_hash` (`**/tests/test_plugin_unit.py`), 3 results

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#558-558), lines 558 to 614

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#676-676), lines 676 to 710

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#1202-1202), lines 1202 to 1218

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#1138-1138), lines 1138 to 1150

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#1031-1031), lines 1031 to 1045

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#1154-1154), lines 1154 to 1178

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#1218-1218), lines 1218 to 1234

Read [](file:///home/wer/Code/mkdocs_piper_tts/src/mkdocs_piper_tts/plugin.py#1180-1180), lines 1180 to 1195

I have enough findings for core behavior, and I am running a quick strict type-check pass to catch additional latent defects that tests may not execute.

Ran terminal command:  /home/wer/Code/mkdocs_piper_tts/.venv/bin/python -m mypy src -q

Ran terminal command:  /home/wer/Code/mkdocs_piper_tts/.venv/bin/python -m mypy src

Read [](file:///home/wer/.vscode-server/data/User/workspaceStorage/cd07255db0ab23632f8f3cf5ce015805/GitHub.copilot-chat/chat-session-resources/ef9a09de-49ea-48be-831d-c8c601b84789/call_iUMHzTVjpFd2H6JQoZ5e1btk__vscode-1784632132085/content.txt#1-1), lines 1 to 260

Read [](file:///home/wer/.vscode-server/data/User/workspaceStorage/cd07255db0ab23632f8f3cf5ce015805/GitHub.copilot-chat/chat-session-resources/ef9a09de-49ea-48be-831d-c8c601b84789/call_iUMHzTVjpFd2H6JQoZ5e1btk__vscode-1784632132085/content.txt#260-260), lines 260 to 276

Inserted into memory file [](file:///home/wer/.vscode-server/data/User/workspaceStorage/cd07255db0ab23632f8f3cf5ce015805/GitHub.copilot-chat/memory-tool/memories/repo/plugin-notes.md)

Progress update: I completed a read-only audit of plugin logic, tests, and scripts, and validated two high-impact bugs with runtime probes. I am now compiling the findings in severity order with direct file links and concrete evidence.

Searched for regex `\[tool.mypy\]` (`**/pyproject.toml`), 1 result

Searched for text `strict = true` (`**/pyproject.toml`), 1 result

Searched for text `files = "src tests"` (`**/pyproject.toml`), 1 result

Findings ordered by severity

1. Severity: Critical. Playlist JavaScript is effectively broken by double-escaped JSON in the audio element.
The playlist payload is escaped once before being put into attributes, then escaped again during attribute rendering, so the browser sees entities like ampersand-quot instead of real quotes. The script then hits JSON.parse failure and exits early, which disables playlist button behavior and auto-advance silently.
References: plugin.py, plugin.py, plugin.py, test_plugin_unit.py.
Evidence from runtime probe: rendered data-playlist contained entity-escaped quotes and failed JSON decoding.

2. Severity: High. Cache invalidation ignores voice identity, so model/speaker changes can reuse wrong audio.
The cache key metadata tracks plugin/source/section hashes and section position/title, but not model path, config path, speaker, language variant, or synthesis mode. If text stays the same and only voice config changes, stale audio is treated as valid.
References: plugin.py, plugin.py, plugin.py, plugin.py.
Evidence from runtime probe: switching from voiceA to voiceB still produced cache hit with zero pending regeneration.

3. Severity: High. Source path slugging causes real collisions for non-ASCII or symbol-only page names.
Path normalization strips non-ASCII aggressively and falls back to page when empty. Different filenames can collapse to the same cache path and overwrite each other.
References: plugin.py, plugin.py, plugin.py, plugin.py.
Evidence from runtime probe: 你好.md, こんにちは.md, and 🙂.md all mapped to page/intro.mp3.

4. Severity: Medium. Stale audio cleanup exists but is dead code; post-build publishes entire cache directory.
The build copies every mp3 in the cache tree to site output, regardless of whether the current build references that file. A cleanup helper exists but is never invoked.
References: plugin.py, plugin.py, plugin.py, plugin.py.
Impact: output bloat and lingering obsolete audio after content removals/renames.

5. Severity: Medium. File hash memoization is process-lifetime and can mask file changes in long-lived workflows.
Hashes are cached by path and never invalidated in-process. A unit test explicitly expects stale hash after file mutation.
References: plugin.py, plugin.py, test_plugin_unit.py, test_plugin_unit.py.
Risk: if plugin instances persist across rebuild cycles, cache decisions can be wrong after source edits.

6. Severity: Low. There are clear dirty-hack maintainability patterns.
Two examples:
- A compatibility-style static method signature that accepts an optional second argument to support odd call forms.
References: plugin.py, plugin.py.
- Pending task normalization accepts both dict and positional tuple formats, increasing ambiguity and runtime branching.
References: plugin.py, plugin.py.

Additional review notes

1. Runtime tests currently pass: 23 passed.
2. Static typing quality is inconsistent with declared strict policy: mypy reports 59 errors in source while strict mode is enabled.
References: pyproject.toml, pyproject.toml, pyproject.toml.

If you want, I can follow up with a targeted fix PR plan in this order:
1. Fix playlist serialization/parsing bug and add a regression test that parses data-playlist.
2. Expand cache metadata to include voice identity and synthesis parameters.
3. Replace lossy slugging with collision-safe encoding and wire stale-audio pruning into post-build.

Created 3 todos