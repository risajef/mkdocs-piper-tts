from mkdocs_piper_tts.plugin import PiperTTSPlugin


def test_extract_text_adds_pauses_for_paragraphs_and_line_breaks() -> None:
    html = "<p>First paragraph</p><p>Second line<br>continues here</p><p>Done.</p>"

    assert PiperTTSPlugin._extract_text(None, html) == "First paragraph. Second line, continues here. Done."