# trainer/synthesizer.py
from __future__ import annotations
import asyncio
import json
import shutil
import subprocess
import sys
import io
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from trainer.audio_utils import write_wav_16k_mono
from trainer.state import TtsSource, Project, save_project

console = Console()
SAMPLE_RATE = 16000
CATALOG_FILE = ".catalog.json"


class ApiLimitReached(Exception):
    """Límite de cuenta/plan que afecta a TODA la fuente (p.ej. cuota de
    caracteres agotada): no tiene sentido seguir probando más voces de esta
    fuente, así que run_synthesis aborta la fuente entera y pasa a la
    siguiente."""
    pass


class VoiceUnavailable(Exception):
    """Una voz concreta es inutilizable de forma permanente en esta cuenta/
    plan (p.ej. voz de la Voice Library de ElevenLabs bloqueada en el plan
    free) — a diferencia de ApiLimitReached, esto NO afecta a las demás
    voces de la misma fuente, así que run_synthesis descarta solo esta voz
    (la quita de selected_voices para no reintentarla en próximas
    ejecuciones) y sigue con el resto."""
    pass


# ── Catálogo ──────────────────────────────────────────────────────────────────

def _catalog_path(sintetizados: Path) -> Path:
    return sintetizados / CATALOG_FILE


def _load_catalog(sintetizados: Path) -> dict[str, str]:
    """Devuelve {clave → filename} de clips ya sintetizados."""
    p = _catalog_path(sintetizados)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_catalog(sintetizados: Path, catalog: dict[str, str]) -> None:
    _catalog_path(sintetizados).write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False)
    )


def _catalog_key(source_url: str, voice: str, speed: float) -> str:
    return f"{source_url}|{voice}|{speed}"


# ── Auth ──────────────────────────────────────────────────────────────────────

def _auth_headers(token: str, token_header: str | None = None) -> dict:
    if token_header:
        return {token_header: token}
    return {"Authorization": f"Bearer {token}"}


# ── Piper ─────────────────────────────────────────────────────────────────────

def list_voices_piper(voices_dir: str) -> list[str]:
    return sorted(str(p) for p in Path(voices_dir).glob("*.onnx"))


def synthesize_piper(
    text: str,
    voice_path: str,
    output_path: Path,
    speed: float = 1.0,
    piper_binary: str | None = None,
) -> None:
    """
    Invoca el motor Piper TTS.

    rhasspy/piper (el binario standalone al que apuntaba la doc original) está
    archivado y nunca publicó binarios para macOS — solo Linux (amd64/arm64/
    armv7). El proyecto activo, OHF-Voice/piper1-gpl, se distribuye como
    paquete Python (`pip install piper-tts`) y se invoca como módulo
    (`python3 -m piper`), no como ejecutable suelto en PATH.

    Si `piper_binary` apunta a un ejecutable real (p.ej. en Linux, donde sí
    pueden existir binarios standalone), se usa tal cual. Si no se indica, o
    no existe como ejecutable, se invoca el módulo `piper` con el intérprete
    Python actual — esta es la ruta correcta para una instalación vía pip.

    Piper escribe el WAV al sample rate nativo del modelo de voz elegido, que
    varía de una voz a otra (algunas voces en español están a 16000 Hz, otras
    a 22050 Hz — Piper no resamplea). openWakeWord exige 16kHz mono estricto,
    así que el resultado se normaliza siempre antes de devolver el control.
    """
    length_scale = str(round(1.0 / speed, 3))
    if piper_binary and shutil.which(piper_binary):
        base_cmd = [piper_binary]
    else:
        base_cmd = [sys.executable, "-m", "piper"]
    cmd = base_cmd + ["--model", voice_path, "--output_file", str(output_path),
                       "--length_scale", length_scale]
    result = subprocess.run(cmd, input=text, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Piper falló: {result.stderr}")

    data, sr = sf.read(str(output_path), dtype="float32")
    write_wav_16k_mono(data, sr, output_path)


# ── OpenAI-compatible ─────────────────────────────────────────────────────────

async def list_voices_openai(url: str, token: str, token_header: str | None = None) -> list[dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{url}/voices",
                headers=_auth_headers(token, token_header),
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            if isinstance(data, list):
                return data
            return data.get("voices", [])
        except Exception:
            return []


async def synthesize_openai(
    text: str,
    voice: str,
    url: str,
    token: str,
    output_path: Path,
    speed: float = 1.0,
    token_header: str | None = None,
) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{url}/audio/speech",
            headers=_auth_headers(token, token_header),
            json={
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": "wav",
            },
        )
        if resp.status_code == 402:
            raise ApiLimitReached(f"Límite de API alcanzado ({url})")
        resp.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = resp.content
    try:
        # Endpoints compatibles con OpenAI no garantizan 16kHz (la propia API
        # de OpenAI TTS devuelve 24kHz salvo que se indique lo contrario), así
        # que se normaliza igual que con Piper.
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        write_wav_16k_mono(data, sr, output_path)
    except Exception:
        output_path.write_bytes(raw)


# ── ElevenLabs ────────────────────────────────────────────────────────────────

async def check_elevenlabs_quota(url: str, token: str) -> dict | None:
    """
    Consulta el uso real de la cuenta vía GET /v1/user/subscription
    (character_count / character_limit / next_character_count_reset_unix).
    Devuelve None si la petición falla — no debe tumbar el flujo de síntesis
    por no poder mostrar un dato informativo de más.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{url}/user/subscription", headers={"xi-api-key": token})
        if resp.status_code == 200:
            data = resp.json()
            return {
                "character_count": data.get("character_count"),
                "character_limit": data.get("character_limit"),
                "next_reset_unix": data.get("next_character_count_reset_unix"),
            }
    except Exception:
        pass
    return None


async def synthesize_elevenlabs(
    text: str,
    voice_id: str,
    url: str,
    token: str,
    output_path: Path,
    speed: float = 1.0,
) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{url}/text-to-speech/{voice_id}",
            headers={"xi-api-key": token},
            params={"output_format": "pcm_16000"},
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"speed": speed},
            },
        )
        # Dos causas muy distintas ambas "no se puede sintetizar", pero con
        # alcance opuesto — una es de la VOZ, la otra es de toda la CUENTA:
        #
        #   402 payment_required / code=paid_plan_required:
        #     "Free users cannot use library voices via the API." — ESTA voz
        #     de la Voice Library de ElevenLabs es inaccesible vía API en el
        #     plan free, pero otras voces (propias, o no-library) pueden
        #     seguir funcionando perfectamente. Se trata como
        #     VoiceUnavailable: se descarta solo esta voz y se sigue con el
        #     resto — no tiene sentido tratarlo como límite de toda la
        #     fuente cuando es una restricción por-voz.
        #   401 con detail.status == "quota_exceeded":
        #     esto sí es cuota real agotada, afecta a TODAS las voces de la
        #     cuenta por igual — se trata como ApiLimitReached (aborta la
        #     fuente entera). Distinto de un 401 por API key inválida (sin
        #     ese status), que debe abortar de verdad como error normal.
        #
        # (elevenlabs.io/docs/developers/resources/error-messages)
        detail = {}
        if resp.status_code in (401, 402, 403):
            try:
                detail = resp.json().get("detail") or {}
            except Exception:
                detail = {}
        detail_status = detail.get("status", "")
        detail_message = detail.get("message", "")

        if resp.status_code == 402 or detail_status == "payment_required":
            raise VoiceUnavailable(
                detail_message
                or "Voz de la Voice Library no disponible vía API en el plan free de ElevenLabs "
                   "(paid_plan_required) — hace falta plan de pago o usar otra voz."
            )
        if detail_status == "quota_exceeded":
            quota = await check_elevenlabs_quota(url, token)
            if quota and quota.get("character_limit"):
                msg = (
                    f"{detail_message or 'Límite de caracteres alcanzado (ElevenLabs)'} — uso real: "
                    f"{quota['character_count']}/{quota['character_limit']} caracteres de este periodo"
                )
            else:
                msg = detail_message or "Límite de caracteres alcanzado (ElevenLabs)"
            raise ApiLimitReached(msg)
        resp.raise_for_status()

    raw = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32768.0
    write_wav_16k_mono(raw, 16000, output_path)


# ── Google Cloud TTS ──────────────────────────────────────────────────────────

GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1"


async def list_voices_google(token: str, language_code: str = "es") -> list[dict]:
    """
    Consulta GET /v1/voices?languageCode=es. Se usa en vez de mantener una
    lista de nombres de voces a mano (el catálogo de Google cambia con
    frecuencia — Neural2/Studio/Chirp3 se han ido añadiendo con el tiempo,
    y una lista fija se queda desactualizada o, peor, con nombres que ya no
    existen).
    """
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{GOOGLE_TTS_URL}/voices",
                headers={"X-Goog-Api-Key": token},
                params={"languageCode": language_code} if language_code else {},
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("voices", [])
        except Exception:
            return []


async def synthesize_google(
    text: str,
    voice: str,
    token: str,
    output_path: Path,
    speed: float = 1.0,
) -> None:
    """
    Sintetiza con Google Cloud Text-to-Speech (`text:synthesize`). Auth vía
    header `X-Goog-Api-Key` (API key simple, no cuenta de servicio/OAuth).

    A diferencia de Piper/OpenAI-compatible, aquí SÍ podemos pedirle a la API
    que nos devuelva el audio directamente en el formato que necesitamos
    (`audioEncoding: LINEAR16`, `sampleRateHertz: 16000`) en vez de tener que
    fiarnos de cuál sea el sample rate nativo de origen — la lección de
    Piper (voces con distinto sample rate nativo, normalización obligatoria
    después) aquí se evita pidiéndolo bien desde el principio. Aun así se
    vuelve a pasar por `write_wav_16k_mono` como red de seguridad, igual que
    el resto de fuentes: no asumimos ciegamente que lo que llega ya es
    perfecto.

    El `languageCode` se deriva del propio nombre de la voz (p.ej.
    "es-ES-Neural2-A" -> "es-ES"), que es como Google nombra sus voces.
    """
    language_code = "-".join(voice.split("-")[:2]) or "es-ES"
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{GOOGLE_TTS_URL}/text:synthesize",
            headers={"X-Goog-Api-Key": token, "Content-Type": "application/json"},
            json={
                "input": {"text": text},
                "voice": {"languageCode": language_code, "name": voice},
                "audioConfig": {
                    "audioEncoding": "LINEAR16",
                    "sampleRateHertz": SAMPLE_RATE,
                    "speakingRate": speed,
                },
            },
        )
        if resp.status_code == 429:
            raise ApiLimitReached("Límite de cuota alcanzado (Google Cloud TTS)")
        resp.raise_for_status()

    import base64
    raw = base64.b64decode(resp.json().get("audioContent", ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    write_wav_16k_mono(data, sr, output_path)


# ── Síntesis principal ────────────────────────────────────────────────────────

def run_synthesis(project: Project) -> int:
    """
    Genera clips sintéticos, saltando los ya catalogados, para cada fuente
    TTS del proyecto. Si una fuente alcanza su límite de API, se salta al
    resto de fuentes en vez de abortar la síntesis entera — antes, un único
    provider agotado (p.ej. ElevenLabs sin cuota) bloqueaba también a
    cualquier otra fuente configurada después en la lista (p.ej. Piper),
    aunque no tuviera nada que ver con esa API.
    """
    sintetizados = project.sintetizados_path
    sintetizados.mkdir(parents=True, exist_ok=True)

    catalog = _load_catalog(sintetizados)
    existing_files = {f.name for f in sintetizados.glob("*.wav")}
    clip_number = len(existing_files) + 1
    total_generated = 0
    any_source_incomplete = False

    # Resuelve todos los tokens ANTES de entrar en el renderizado en vivo de
    # las barras de progreso. `_resolve_token` puede caer a un `console.input(...)`
    # interactivo si falta la variable de entorno — y un prompt bloqueante
    # lanzado dentro de un contexto `Progress`/`Live` queda invisible bajo el
    # redibujado automático: el proceso se queda esperando una entrada por
    # teclado que nunca se llega a ver, y da la impresión de estar colgado
    # (barra en 0, sin logs, sin error) cuando en realidad solo falta pulsar
    # una tecla que no se puede leer en pantalla.
    tokens = {
        id(source): _resolve_token(source)
        for source in project.synthesis.sources
        if source.type in ("openai", "google")
    }

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}")) as progress:

        for source in project.synthesis.sources:
            source_limit_hit = False
            blocked_voices: set[str] = set()

            pending = [
                (v, s) for v in source.selected_voices for s in source.speeds
                if _catalog_key(source.url or source.type, v, s) not in catalog
            ]
            done_count = (
                len(source.selected_voices) * len(source.speeds) - len(pending)
            )
            total_clips = len(source.selected_voices) * len(source.speeds)
            label = source.url.split("//")[-1].split("/")[0] if source.url else source.type
            task = progress.add_task(f"[cyan]{label}[/cyan]", total=total_clips, completed=done_count)

            token = tokens.get(id(source), "")

            for voice, speed in pending:
                if source_limit_hit:
                    break
                if voice in blocked_voices:
                    # Ya sabemos que esta voz está bloqueada (VoiceUnavailable
                    # en una velocidad anterior) — no tiene sentido repetir la
                    # misma llamada 4 veces más solo para variar la velocidad.
                    progress.advance(task)
                    continue

                out_path = sintetizados / f"{clip_number:03d}.wav"
                key = _catalog_key(source.url or source.type, voice, speed)
                try:
                    if source.type == "piper":
                        synthesize_piper(project.wake_word, voice, out_path,
                                         speed=speed, piper_binary=source.binary)
                    elif source.type == "google":
                        asyncio.run(synthesize_google(project.wake_word, voice, token, out_path, speed))
                    elif source.type == "openai":
                        if source.token_header == "xi-api-key":
                            asyncio.run(synthesize_elevenlabs(project.wake_word, voice,
                                                               source.url, token, out_path, speed))
                        else:
                            asyncio.run(synthesize_openai(project.wake_word, voice,
                                                          source.url, token, out_path, speed,
                                                          token_header=source.token_header))
                    catalog[key] = out_path.name
                    _save_catalog(sintetizados, catalog)
                    clip_number += 1
                    total_generated += 1
                except ApiLimitReached as exc:
                    progress.console.print(f"  [yellow]⚠️  {label}: {escape(str(exc))}[/yellow]")
                    progress.console.print(
                        f"  '{label}' se queda pendiente — continuando con el resto de fuentes."
                    )
                    source_limit_hit = True
                except VoiceUnavailable as exc:
                    progress.console.print(
                        f"  [yellow]⏭  {label}/{voice}: {escape(str(exc))} "
                        f"— se descarta esta voz, se sigue con el resto.[/yellow]"
                    )
                    blocked_voices.add(voice)
                    if voice in source.selected_voices:
                        source.selected_voices.remove(voice)
                        save_project(project)
                except Exception as exc:
                    progress.console.print(f"  [red]✗ {label}/{voice}/{speed}:[/red] {escape(str(exc))}")
                finally:
                    progress.advance(task)

            still_pending = [
                (v, s) for v in source.selected_voices for s in source.speeds
                if _catalog_key(source.url or source.type, v, s) not in catalog
            ]
            if still_pending:
                any_source_incomplete = True

    existing_total = len(list(sintetizados.glob("*.wav")))
    project.synthesis.clips = existing_total
    project.synthesis.status = "pending" if any_source_incomplete else "done"
    save_project(project)
    return total_generated


def _resolve_token(source: TtsSource) -> str:
    import os
    if source.token_env:
        token = os.environ.get(source.token_env, "")
        if token:
            return token
    return console.input(f"  Token para {source.url}: ").strip()
