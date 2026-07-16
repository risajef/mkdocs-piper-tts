# mkdocs-piper-tts

An MkDocs plugin that generates Piper text-to-speech audio for pages and adds
an HTML audio control through the `piper_tts_button` template helper.

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
        de:
          model: de_DE-thorsten-medium.onnx
          label: Vorlesen
```

Set `lang` in a page's front matter. The plugin caches generated MP3 files and
sidecar metadata under `<docs_dir>/<asset_dir>/<audio_dir>`. Cached files are
reused when both page source and plugin code are unchanged.

Set `generate_audio: false`, or `PIPER_TTS_GENERATE_AUDIO=false`, for
cache-only builds. In this mode, missing or stale audio fails the build instead
of initializing Piper; use it in CI after restoring a verified audio cache
artifact.

Render the control in an MkDocs template with:

```jinja2
{{ piper_tts_button(page) }}
```

For a more extensive production example, see [retoweber.info](https://retoweber.info/),
which uses this plugin for its English and German pages.

## Example And Tests

[`examples/simple-site`](examples/simple-site) is a complete, minimal MkDocs
project. It is in the source repository, not the published wheel. Put a Piper
model and matching `.onnx.json` file in `examples/simple-site/models`, then run:

```bash
cd examples/simple-site
mkdocs build --strict
```

The example uses CPU synthesis by default. Set `use_cuda: true` in its
`mkdocs.yml` after installing the CUDA extra on a system with a compatible GPU.

The repository's end-to-end tests build the example once with CPU and once with
CUDA. They download a cached test voice unless `PIPER_TTS_TEST_MODEL_DIR` names
a directory containing a model and matching configuration:

```bash
pip install -e '.[test]'
pytest -m 'not cuda'
PIPER_TTS_TEST_MODEL_DIR=/path/to/models pytest -m cuda
```

The CUDA test skips when ONNX Runtime cannot create a CUDA execution provider.
`ffmpeg` must be available on `PATH` for either test.

## Published Example

Every release rebuilds the CPU example from an empty generated-output directory,
runs its CPU end-to-end test, and deploys the result to
[mkdocs-piper-tts.retoweber.info](https://mkdocs-piper-tts.retoweber.info/). The deployed site
includes an E2E Build Status page with the release tag and build timestamp. The
workflow deliberately does not cache models, generated audio, or site output.

## Releases

Tags named `v*` build and publish a wheel and source distribution with PyPI
trusted publishing. For each release, update the version in `pyproject.toml`,
commit it, and push a matching tag such as `v0.2.0`. PyPI versions are
immutable, so never reuse a published version or tag.
