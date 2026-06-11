# tests/test_synthesizer.py
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from trainer.synthesizer import (
    synthesize_piper, synthesize_openai, list_voices_openai, list_voices_piper,
    run_synthesis,
)
from trainer.state import TtsSource


def make_wav_bytes(sr: int = 16000, duration: float = 0.5) -> bytes:
    import io
    data = (np.random.randn(int(duration * sr)) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_synthesize_piper_calls_binary(tmp_path):
    out = tmp_path / "clip.wav"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        synthesize_piper(
            text="ok jota",
            voice_path="piper/voices/es_ES.onnx",
            output_path=out,
            speed=1.0,
            piper_binary="piper/piper",
        )
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "piper/piper" in args
    assert "piper/voices/es_ES.onnx" in args


def test_synthesize_piper_raises_on_error(tmp_path):
    out = tmp_path / "clip.wav"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="model not found")
        with pytest.raises(RuntimeError, match="Piper"):
            synthesize_piper("ok jota", "bad.onnx", out, piper_binary="piper/piper")


def test_list_voices_piper(tmp_path):
    (tmp_path / "es_ES_female.onnx").touch()
    (tmp_path / "es_ES_male.onnx").touch()
    (tmp_path / "config.json").touch()  # debe ignorarse
    voices = list_voices_piper(str(tmp_path))
    assert len(voices) == 2
    assert all(v.endswith(".onnx") for v in voices)


@pytest.mark.asyncio
async def test_synthesize_openai_posts_to_endpoint(tmp_path):
    out = tmp_path / "clip.wav"
    wav_bytes = make_wav_bytes()

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = wav_bytes
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        await synthesize_openai(
            text="ok jota",
            voice="Rachel",
            url="https://api.example.com/v1",
            token="sk-test",
            output_path=out,
            speed=1.0,
        )

    assert out.exists()


@pytest.mark.asyncio
async def test_list_voices_openai_parses_response(tmp_path):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={"voices": [
        {"voice_id": "Rachel", "name": "Rachel", "labels": {"gender": "female", "language": "es"}},
        {"voice_id": "Josh",   "name": "Josh",   "labels": {"gender": "male",   "language": "en"}},
    ]})

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        voices = await list_voices_openai("https://api.example.com/v1", "sk-test")

    assert len(voices) == 2
    assert voices[0]["name"] == "Rachel"


@pytest.mark.asyncio
async def test_list_voices_openai_returns_empty_on_error():
    mock_response = AsyncMock()
    mock_response.status_code = 404

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        voices = await list_voices_openai("https://api.example.com/v1", "sk-test")

    assert voices == []
