# mkdocs-piper-tts

An MkDocs plugin that generates Piper text-to-speech audio for pages and adds
an HTML audio playlist control through the `piper_tts_button` template helper.

## Install

```bash
pip install mkdocs-piper-tts
```

For CUDA synthesis, install the CUDA extra on a compatible CUDA 12 system:

```bash
pip install 'mkdocs-piper-tts[cuda]'
```

The plugin invokes `ffmpeg` to encode MP3 files, so `ffmpeg` must be available
on `PATH` when generating audio.

## Piper Version Policy

This package intentionally pins `piper-tts==1.2.0`. That release is MIT
licensed and is compatible with this package's MIT license. Do not upgrade
Piper without first reviewing the license of the target release and its
runtime dependencies.

## Configure

Store Piper `.onnx` models and their JSON configuration files outside the
published documentation source, then enable the plugin in `mkdocs.yml`:

```yaml
plugins:
  - piper-tts:
      model_dir: models/piper-tts
      asset_dir: assets/piper-tts
      audio_dir: audio
      use_cuda: true
      batch_size: 2
      languages:
        en:
          model: en_US-amy-medium.onnx
          label: Listen
          download_url: https://example.invalid/en_US-amy-medium.onnx
        de:
          model: de_DE-thorsten-medium.onnx
          label: Vorlesen
```

Set `lang` in a page's front matter. The plugin caches generated MP3 files and
sidecar metadata under `<docs_dir>/<asset_dir>/<audio_dir>`. Cached files are
reused when both page source and plugin code are unchanged.

When generation needs a missing model or configuration, the build fails before
initializing Piper and prints each exact expected path. Add `download_url` to a
language to include direct URLs for the `.onnx` and `.onnx.json` files in that
error; the bundled German and English defaults already provide them.

Set `generate_audio: false`, or `PIPER_TTS_GENERATE_AUDIO=false`, for
cache-only builds. In this mode, missing or stale audio fails the build instead
of initializing Piper; use it in CI after restoring a verified audio cache
artifact.

Render the control in an MkDocs template with:

```jinja2
{{ piper_tts_button(page) }}
```

Render an additional reading-time element with:

```jinja2
{{ piper_tts_reading_time(page) }}
```

Both helpers are also available as:

```jinja2
{{ piper_tts_playlist(page) }}
{{ piper_tts_reading_time(page, "approximate reading time:") }}
```

The reading-time label is language-aware by default: it falls back to each
language's `reading_time_label` (configured alongside `label` in `languages`,
or the bundled German/English default) when no explicit label argument is
given, so templates covering multiple languages do not need to pass a label at
all.

Pages with only a few seconds of audio rarely warrant a reading-time badge.
Rather than parsing the rendered duration text, set a threshold once in
`mkdocs.yml`:

```yaml
plugins:
  - piper-tts:
      reading_time_min_seconds: 60
```

or pass `min_seconds` at the call site to override it per template:

```jinja2
{{ piper_tts_reading_time(page, min_seconds=60) }}
{{ piper_tts_controls(page, reading_time_min_seconds=60) }}
```

`piper_tts_reading_time` returns an empty string whenever the total duration is
below the effective threshold (the call-site argument if given, otherwise
`reading_time_min_seconds`, which defaults to `0`, i.e. always shown).

Use the combined helper to render the reading time and the audio player as a
single block, in the order they should appear, without any client-side DOM
manipulation:

```jinja2
{{ piper_tts_controls(page) }}
```

To place that combined block immediately after a page's first `h1` without any
template markup or JavaScript repositioning, set:

```yaml
plugins:
  - piper-tts:
      insert_reading_time_after_heading: true
```

With this enabled, the plugin injects the reading-time markup directly into
the rendered page HTML during the build; do not also call
`piper_tts_reading_time(page)` in the template, or it will render twice.
`piper_tts_button`/`piper_tts_playlist` are unaffected and can still be placed
anywhere (for example, in a fixed player bar) independently of this setting.

### Compact Player Mode

By default `piper_tts_button`/`piper_tts_playlist` render the browser's native
`<audio controls>` element plus an ordered list of track buttons (`mode="list"`).
Pass `mode="compact"` for a title+prev/next+play/pause bar rendered and driven
entirely by the plugin, so consuming themes do not need to reimplement a
player against internal markup:

```jinja2
{{ piper_tts_playlist(page, mode="compact") }}
{{ piper_tts_controls(page, mode="compact") }}
```

In compact mode, the native `<audio>` element is rendered without the
`controls` attribute (so browsers do not show their own UI); the compact bar
is the only visible control surface and does not require any CSS to hide the
native player. Sites are still responsible for the bar's visual styling using
the stable classes below.

Prev/next/play/pause accessibility labels are language-aware: configure
`prev_label`, `next_label`, `play_label`, and `pause_label` per language in
`languages` (alongside `label` and `reading_time_label`), the same way the
bundled German and English defaults do. No labels need to be threaded through
the consuming theme's own translation dictionary.

### Theming API

The following classes and attributes are a stable, documented contract for
styling the player; they are safe to target from a theme's CSS and will not
change without a note in the changelog:

- `.piper-tts-button-wrapper` — outer container for both modes (the base
  class name follows the configured `button_class`, default `piper-tts-button`).
- `.piper-tts-mode-compact` — added to the wrapper only in compact mode.
- `.piper-tts-button-playlist` — the `<ol>` of track buttons (present in both
  modes); `button[data-track-index]` are its items, and the active item has
  `aria-current="true"`.
- `.piper-tts-playlist-expanded` — toggled onto the playlist `<ol>` in compact
  mode to reveal it as a dropdown; absent/removed means collapsed.
- `.piper-tts-compact-bar` — the compact mode bar container, with
  `.piper-tts-prev`, `.piper-tts-play-pause`, `.piper-tts-now-playing`, and
  `.piper-tts-next` buttons inside it.
- `.piper-tts-reading-time` — the reading-time `<span>`.
- `.piper-tts-controls` — the wrapper `<div>` emitted by `piper_tts_controls`.

## Section And Playlist Rules

Audio is generated per page section and then played as an ordered playlist:

- When a page has multiple `h1` elements, one audio file is generated per `h1`
  section.
- When a page has exactly one `h1` and one or more `h2` elements:
  - the first audio file covers the `h1` content until the first `h2` and is
    named from the title as `(Intro) <h1 title>`
  - one audio file is generated for each `h2` section
- When a page has exactly one `h1` and no `h2`, one audio file is generated for
  the full page (not an intro), named from the `h1` title.

Generated audio files are stored under a page-specific directory and named from
their section titles (slugged and made unique when duplicates exist).

The rendered player auto-advances through the playlist track-by-track.

For a more extensive production example, see [retoweber.info](https://retoweber.info/),
which uses this plugin for its English and German pages.

## Example And Tests

[`examples/simple-site`](examples/simple-site) is a complete, minimal MkDocs
project. It is in the source repository, not the published wheel. Its Piper
model and matching `.onnx.json` configuration are stored in the
`example-voice-v1` GitHub Release asset, not in Git or package distributions.
Restore the verified asset before a local build:

```bash
python scripts/example_voice_asset.py restore
```

Then run:

```bash
cd examples/simple-site
mkdocs build --strict
```

The example uses CPU synthesis by default. Set `use_cuda: true` in its
`mkdocs.yml` after installing the CUDA extra on a system with a compatible GPU.

## Deploying An Example

The example is deployed from the repository's `gh-pages` branch with MkDocs'
standard `gh-deploy` command. Its `site_url` and `remote_branch` are already
configured in `examples/simple-site/mkdocs.yml`.

### Standard GitHub Actions Deployment

Push a release tag or use the **Deploy Example Pages** workflow manually to
restore the checked release asset, synthesize the example on CPU, and run
`mkdocs gh-deploy`. The workflow never downloads a voice from the Piper source;
it restores the versioned GitHub Release artifact and verifies its checksum.

### Local Precomputation And Deployment

Prefer local synthesis when a compatible CUDA GPU is available, when the site
has substantial audio, or when GitHub Actions minutes are limited. Generate
the audio on the machine that has the model and accelerator, then deploy the
cached static output with MkDocs:

```bash
cd examples/simple-site
# Set use_cuda: true in mkdocs.yml when using the CUDA extra.
mkdocs build --strict
mkdocs gh-deploy --strict --force
```

`mkdocs gh-deploy` rebuilds the site but reuses valid cached audio, then pushes
the resulting static artifact, including MP3 files, to `gh-pages`. The example
voice is kept only as a GitHub Release artifact; it is never included in Git or
package distributions.

The repository's end-to-end tests build the example once with CPU and once with
CUDA using the restored example voice. Set `PIPER_TTS_TEST_MODEL_DIR` to test
with a different directory containing a model and matching configuration:

```bash
pip install -e '.[test]'
pytest -m 'not cuda'
PIPER_TTS_TEST_MODEL_DIR=/path/to/models pytest -m cuda
```

The CUDA test skips when ONNX Runtime cannot create a CUDA execution provider.
`ffmpeg` must be available on `PATH` for either test.

## Published Example

Every release runs the CPU end-to-end test and uses `mkdocs gh-deploy` to deploy
the result to
[mkdocs-piper-tts.retoweber.info](https://mkdocs-piper-tts.retoweber.info/). The deployed site
includes an E2E Build Status page with the release tag and build timestamp.

To replace the stored example voice, put the `.onnx` and `.onnx.json` files in
`examples/simple-site/models/` and run `python scripts/example_voice_asset.py
publish`. This updates the `example-voice-v1` Release asset and checksum; it
does not add model files to Git or a package distribution.

## Releases

Tags named `v*` build and publish a wheel and source distribution with PyPI
trusted publishing. For each release, update the version in `pyproject.toml`,
commit it, and push a matching tag such as `v0.2.0`. PyPI versions are
immutable, so never reuse a published version or tag.
