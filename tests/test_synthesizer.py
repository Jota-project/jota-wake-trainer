# tests/test_synthesizer.py
import sys
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from trainer.synthesizer import (
    synthesize_piper, synthesize_openai, list_voices_openai, list_voices_piper,
    synthesize_google, list_voices_google,
    run_synthesis, ApiLimitReached,
)
from trainer.state import TtsSource, Project, SynthesisState, TrainingState, Voice


def make_wav_bytes(sr: int = 16000, duration: float = 0.5) -> bytes:
    import io
    data = (np.random.randn(int(duration * sr)) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _fake_piper_run(out_path: Path, sr: int = 22050):
    """
    Simula lo que hace realmente Piper: escribir un WAV en `--output_file` al
    sample rate nativo de la voz (que puede no ser 16kHz — 22050 por
    defecto aquí, justo el caso que rompía el entrenamiento).
    """
    def _run(cmd, *a, **kw):
        data = (np.random.randn(int(0.5 * sr)) * 0.3).astype(np.float32)
        sf.write(str(out_path), data, sr, subtype="PCM_16")
        return MagicMock(returncode=0, stderr="")
    return _run


def test_synthesize_piper_invokes_module_by_default(tmp_path):
    """
    Sin binario configurado (el caso normal en macOS, donde rhasspy/piper
    nunca publicó releases), debe invocar el paquete piper-tts como módulo
    Python con el intérprete actual, no un ejecutable 'piper' suelto.
    """
    out = tmp_path / "clip.wav"
    with patch("subprocess.run", side_effect=_fake_piper_run(out)) as mock_run:
        synthesize_piper(
            text="ok jota",
            voice_path="piper/voices/es_ES.onnx",
            output_path=out,
            speed=1.0,
        )
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[:3] == [sys.executable, "-m", "piper"]
    assert "piper/voices/es_ES.onnx" in args


def test_synthesize_piper_normalizes_to_16k_mono(tmp_path):
    """
    Regresión: distintas voces Piper tienen distinto sample rate nativo
    (algunas 16000 Hz, otras 22050 Hz). openwakeword exige 16000 Hz exacto
    y falla sin decir qué fichero está mal — el resultado debe normalizarse
    siempre, sea cual sea el sample rate que devuelva Piper.
    """
    out = tmp_path / "clip.wav"
    with patch("subprocess.run", side_effect=_fake_piper_run(out, sr=22050)):
        synthesize_piper(
            text="ok jota",
            voice_path="piper/voices/es_ES-davefx-medium.onnx",
            output_path=out,
            speed=1.0,
        )
    info = sf.info(str(out))
    assert info.samplerate == 16000
    assert info.channels == 1


def test_synthesize_piper_uses_real_binary_when_available(tmp_path):
    """Si se configura una ruta a un ejecutable real (p.ej. Linux), se usa tal cual."""
    out = tmp_path / "clip.wav"
    real_binary = tmp_path / "piper"
    real_binary.write_text("#!/bin/sh\n")
    real_binary.chmod(0o755)

    with patch("subprocess.run", side_effect=_fake_piper_run(out)) as mock_run:
        synthesize_piper(
            text="ok jota",
            voice_path="piper/voices/es_ES.onnx",
            output_path=out,
            speed=1.0,
            piper_binary=str(real_binary),
        )
    args = mock_run.call_args[0][0]
    assert args[0] == str(real_binary)


def test_synthesize_piper_falls_back_when_configured_binary_missing(tmp_path):
    """
    Regresión: un 'binary' configurado que apunta a una ruta inexistente
    (p.ej. el antiguo default 'piper/piper', que nunca existió en macOS)
    debe caer de vuelta a invocar el módulo en vez de fallar con
    'No such file or directory'.
    """
    out = tmp_path / "clip.wav"
    with patch("subprocess.run", side_effect=_fake_piper_run(out)) as mock_run:
        synthesize_piper(
            text="ok jota",
            voice_path="piper/voices/es_ES.onnx",
            output_path=out,
            speed=1.0,
            piper_binary="piper/piper",
        )
    args = mock_run.call_args[0][0]
    assert args[:3] == [sys.executable, "-m", "piper"]


def test_synthesize_piper_raises_on_error(tmp_path):
    out = tmp_path / "clip.wav"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="model not found")
        with pytest.raises(RuntimeError, match="Piper"):
            synthesize_piper("ok jota", "bad.onnx", out)


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
async def test_synthesize_openai_normalizes_non_16k_response(tmp_path):
    """
    Regresión: la propia API de OpenAI TTS devuelve WAV a 24kHz salvo que se
    pida lo contrario. Cualquier endpoint compatible con OpenAI podría
    devolver un sample rate distinto de 16kHz, así que debe normalizarse
    igual que la salida de Piper.
    """
    out = tmp_path / "clip.wav"
    wav_bytes = make_wav_bytes(sr=24000)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = wav_bytes
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        await synthesize_openai(
            text="ok jota",
            voice="alloy",
            url="https://api.example.com/v1",
            token="sk-test",
            output_path=out,
            speed=1.0,
        )

    info = sf.info(str(out))
    assert info.samplerate == 16000
    assert info.channels == 1


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


def _make_project(tmp_path, sources, monkeypatch, model_name="ok_jota"):
    import trainer.state as state_mod
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", tmp_path)
    return Project(
        wake_word="ok jota",
        model_name=model_name,
        created_at="2026-01-01T00:00:00+00:00",
        voices=[Voice(name="Sito")],
        synthesis=SynthesisState(sources=sources),
        training=TrainingState(),
    )


def test_run_synthesis_continues_with_next_source_after_api_limit(tmp_path, monkeypatch):
    """
    Regresión: si una fuente (p.ej. ElevenLabs) agota su cuota, las fuentes
    siguientes (p.ej. Piper, sin relación con esa API) deben seguir
    procesándose en vez de abortar la síntesis entera.
    """
    exhausted = TtsSource(type="openai", url="https://api.elevenlabs.io/v1",
                          token_header="xi-api-key", selected_voices=["voz1"], speeds=[1.0])
    working = TtsSource(type="piper", binary="piper/piper", voices_dir="piper/voices",
                        selected_voices=["voz_piper"], speeds=[1.0])

    project = _make_project(tmp_path, [exhausted, working], monkeypatch)

    def fake_elevenlabs(*a, **kw):
        raise ApiLimitReached("Límite de API alcanzado (ElevenLabs)")

    def fake_piper(text, voice_path, output_path, speed=1.0, piper_binary="piper/piper"):
        Path(output_path).write_bytes(make_wav_bytes())

    with patch("trainer.synthesizer.synthesize_elevenlabs", new=AsyncMock(side_effect=fake_elevenlabs)), \
         patch("trainer.synthesizer.synthesize_piper", side_effect=fake_piper), \
         patch("trainer.synthesizer._resolve_token", return_value="dummy-token"), \
         patch("trainer.synthesizer.save_project"):
        total = run_synthesis(project)

    # La fuente piper (independiente del límite de ElevenLabs) debe haberse generado
    assert total == 1
    assert project.synthesis.status == "pending"  # ElevenLabs se quedó sin completar


def test_run_synthesis_marks_done_when_all_sources_complete(tmp_path, monkeypatch):
    working = TtsSource(type="piper", binary="piper/piper", voices_dir="piper/voices",
                        selected_voices=["voz_piper"], speeds=[1.0])
    project = _make_project(tmp_path, [working], monkeypatch)

    def fake_piper(text, voice_path, output_path, speed=1.0, piper_binary="piper/piper"):
        Path(output_path).write_bytes(make_wav_bytes())

    with patch("trainer.synthesizer.synthesize_piper", side_effect=fake_piper), \
         patch("trainer.synthesizer.save_project"):
        run_synthesis(project)

    assert project.synthesis.status == "done"


@pytest.mark.asyncio
async def test_list_voices_openai_returns_empty_on_error():
    mock_response = AsyncMock()
    mock_response.status_code = 404

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        voices = await list_voices_openai("https://api.example.com/v1", "sk-test")

    assert voices == []


# ── Google Cloud TTS ──────────────────────────────────────────────────────────

def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


@pytest.mark.asyncio
async def test_synthesize_google_posts_to_synthesize_endpoint(tmp_path):
    out = tmp_path / "clip.wav"
    wav_bytes = make_wav_bytes(sr=16000)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={"audioContent": _b64(wav_bytes)})
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_post = AsyncMock(return_value=mock_response)
        MockClient.return_value.__aenter__.return_value.post = mock_post
        await synthesize_google(
            text="ok jota", voice="es-ES-Neural2-A", token="api-key-test",
            output_path=out, speed=1.0,
        )

    assert out.exists()
    call = mock_post.call_args
    assert call.args[0] == "https://texttospeech.googleapis.com/v1/text:synthesize"
    assert call.kwargs["headers"]["X-Goog-Api-Key"] == "api-key-test"
    body = call.kwargs["json"]
    assert body["input"]["text"] == "ok jota"
    assert body["voice"]["name"] == "es-ES-Neural2-A"
    # languageCode se deriva del propio nombre de la voz.
    assert body["voice"]["languageCode"] == "es-ES"
    assert body["audioConfig"]["sampleRateHertz"] == 16000
    assert body["audioConfig"]["audioEncoding"] == "LINEAR16"


@pytest.mark.asyncio
async def test_synthesize_google_normalizes_non_16k_response(tmp_path):
    """
    Aunque se pida sampleRateHertz=16000 explícitamente, no nos fiamos
    ciegamente — mismo principio que con Piper/OpenAI: siempre se normaliza
    lo que llegue de verdad.
    """
    out = tmp_path / "clip.wav"
    wav_bytes = make_wav_bytes(sr=24000)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={"audioContent": _b64(wav_bytes)})
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        await synthesize_google(
            text="ok jota", voice="es-ES-Standard-A", token="api-key-test", output_path=out,
        )

    info = sf.info(str(out))
    assert info.samplerate == 16000
    assert info.channels == 1


@pytest.mark.asyncio
async def test_synthesize_google_raises_api_limit_on_429(tmp_path):
    out = tmp_path / "clip.wav"
    mock_response = AsyncMock()
    mock_response.status_code = 429

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        with pytest.raises(ApiLimitReached):
            await synthesize_google(
                text="ok jota", voice="es-ES-Neural2-A", token="api-key-test", output_path=out,
            )


@pytest.mark.asyncio
async def test_list_voices_google_filters_by_language(tmp_path):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={"voices": [
        {"name": "es-ES-Neural2-A", "ssmlGender": "FEMALE", "naturalSampleRateHertz": 24000},
        {"name": "es-US-Neural2-B", "ssmlGender": "MALE", "naturalSampleRateHertz": 24000},
    ]})

    with patch("httpx.AsyncClient") as MockClient:
        mock_get = AsyncMock(return_value=mock_response)
        MockClient.return_value.__aenter__.return_value.get = mock_get
        voices = await list_voices_google("api-key-test", language_code="es")

    assert len(voices) == 2
    call = mock_get.call_args
    assert call.kwargs["headers"]["X-Goog-Api-Key"] == "api-key-test"
    assert call.kwargs["params"] == {"languageCode": "es"}


@pytest.mark.asyncio
async def test_list_voices_google_returns_empty_on_error():
    mock_response = AsyncMock()
    mock_response.status_code = 403

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        voices = await list_voices_google("bad-key")

    assert voices == []


def test_run_synthesis_dispatches_to_google(tmp_path, monkeypatch):
    """Regresión de wiring: un source type='google' debe llamar a synthesize_google,
    no caer por accidente en la rama 'openai' (o no hacer nada)."""
    source = TtsSource(type="google", token_env="GOOGLE_TTS_API_KEY",
                       selected_voices=["es-ES-Neural2-A"], speeds=[1.0])
    project = _make_project(tmp_path, [source], monkeypatch)

    calls = []

    async def fake_google(text, voice, token, output_path, speed=1.0):
        calls.append((text, voice, token, speed))
        Path(output_path).write_bytes(make_wav_bytes())

    with patch("trainer.synthesizer.synthesize_google", new=AsyncMock(side_effect=fake_google)), \
         patch("trainer.synthesizer._resolve_token", return_value="dummy-google-key"), \
         patch("trainer.synthesizer.save_project"):
        total = run_synthesis(project)

    assert total == 1
    assert len(calls) == 1
    assert calls[0][1] == "es-ES-Neural2-A"
    assert calls[0][2] == "dummy-google-key"
