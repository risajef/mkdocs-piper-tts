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

For TensorRT synthesis support, install:

```bash
pip install 'mkdocs-piper-tts[tensorrt]'
```

The plugin invokes `ffmpeg` to encode MP3 files, so `ffmpeg` must be available
on `PATH` when generating audio.

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

Render the control in an MkDocs template with:

```jinja2
{{ piper_tts_button(page) }}
```

## Releases

Tags named `v*` build and publish a wheel and source distribution with PyPI
trusted publishing. Before the first release, configure PyPI's pending trusted
publisher for the `risajef/mkdocs-piper-tts` repository and the
`publish-pypi.yml` workflow.
