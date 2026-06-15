# tests/test_piper_downloader.py
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import trainer.piper_downloader as dl_mod


FAKE_VOICES = {
    "es_ES/davefx/medium": {
        "name": "davefx",
        "language": {"code": "es_ES", "name_english": "Spanish (Spain)"},
        "quality": "medium",
        "files": {
            "es_ES/davefx/medium/es_ES-davefx-medium.onnx": {"size_bytes": 61000000},
            "es_ES/davefx/medium/es_ES-davefx-medium.onnx.json": {"size_bytes": 4000},
        },
    },
    "en_US/lessac/medium": {
        "name": "lessac",
        "language": {"code": "en_US", "name_english": "English (US)"},
        "quality": "medium",
        "files": {
            "en_US/lessac/medium/en_US-lessac-medium.onnx": {"size_bytes": 63000000},
            "en_US/lessac/medium/en_US-lessac-medium.onnx.json": {"size_bytes": 4500},
        },
    },
}


def test_fetch_voices_index_filters_by_lang():
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_VOICES

    with patch("httpx.get", return_value=mock_resp):
        result = dl_mod.fetch_voices_index(lang_filter="es")

    assert "es_ES/davefx/medium" in result
    assert "en_US/lessac/medium" not in result


def test_fetch_voices_index_no_filter_returns_all():
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_VOICES

    with patch("httpx.get", return_value=mock_resp):
        result = dl_mod.fetch_voices_index(lang_filter=None)

    assert len(result) == 2


def test_download_voice_creates_files(tmp_path):
    file_paths = list(FAKE_VOICES["es_ES/davefx/medium"]["files"].keys())

    fake_content = b"fake-model-bytes"
    mock_response = MagicMock()
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.iter_bytes.return_value = [fake_content]

    with patch("httpx.stream", return_value=mock_response):
        downloaded = dl_mod.download_voice(file_paths, tmp_path)

    assert len(downloaded) == 2
    for p in downloaded:
        assert p.exists()
        assert p.read_bytes() == fake_content
