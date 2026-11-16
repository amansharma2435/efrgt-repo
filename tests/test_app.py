"""Streamlit smoke tests, including a real PDF upload through the widgets."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from docqa.sample_document import build_sample_pdf_bytes

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_upload_and_does_not_show_a_key():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()

    assert not app.exception
    assert app.title[0].value == "Document question answering"
    assert app.sidebar.file_uploader
    captions = [block.value for block in app.sidebar.caption]
    assert any("GEMINI_API_KEY" in caption for caption in captions)
    assert any("top 3" in caption for caption in captions)
    joined = "\n".join(captions)
    assert "unit-test-key" not in joined
    assert "AIza" not in joined
    assert app.chat_input[0].disabled


def test_upload_indexes_the_sample_and_a_missing_key_is_shown():
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.run()
    app.sidebar.file_uploader[0].set_value(
        ("campus_handbook.pdf", build_sample_pdf_bytes(), "application/pdf")
    ).run()

    assert not app.exception
    success_text = " ".join(block.value for block in app.success)
    assert "Indexed campus_handbook.pdf" in success_text
    assert "3 chunks" in success_text
    assert app.chat_input[0].disabled is False

    app.chat_input[0].set_value("When does the central library close?").run()

    assert not app.exception
    errors = " ".join(block.value for block in app.error)
    assert "GEMINI_API_KEY" in errors
    assert "AIza" not in errors
    assert "unit-test-key" not in errors
