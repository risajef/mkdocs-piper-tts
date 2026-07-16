# Voice files

The example's `en_US-lessac-medium.onnx` and matching
`en_US-lessac-medium.onnx.json` are ignored local files. They are stored in the
`example-voice-v1` GitHub Release asset, not in Git or either PyPI distribution.

Restore them before local tests or builds:

```bash
python scripts/example_voice_asset.py restore
```

After deliberately updating the voice, publish a new checked release asset:

```bash
python scripts/example_voice_asset.py publish
```

The upstream source files are available at:

- <https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx>
- <https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json>
