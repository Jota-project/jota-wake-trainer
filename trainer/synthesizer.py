# trainer/synthesizer.py
from __future__ import annotations
import asyncio
import subprocess
import io
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from trainer.state import TtsSource, Project, save_project

console = Console()
SAMPLE_RATE = 16000


def list_voices_piper(voices_dir: str) -> list[str]:
    return sorted(str(p) for p in Path(voices_dir).glob("*.onnx"))


def synthesize_piper(
    text: str,
    voice_path: str,
    output_path: Path,
    speed: float = 1.0,
    piper_binary: str = "piper/piper",
) -> None:
    length_scale = str(round(1.0 / speed, 3))
    cmd = [piper_binary, "--model", voice_path, "--output_file", str(output_path),
           "--length_scale", length_scale]
    result = subprocess.run(cmd, input=text, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Piper falló: {result.stderr}")


async def list_voices_openai(url: str, token: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{url}/voices",
                headers={"Authorization": f"Bearer {token}"},
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
) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{url}/audio/speech",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": "wav",
            },
        )
        resp.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = resp.content
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        sf.write(str(output_path), data, sr, subtype="PCM_16")
    except Exception:
        output_path.write_bytes(raw)


def run_synthesis(project: Project) -> int:
    """Genera todos los clips sintéticos definidos en project.synthesis.sources."""
    sintetizados = project.sintetizados_path
    sintetizados.mkdir(parents=True, exist_ok=True)
    existing = len(list(sintetizados.glob("*.wav")))
    clip_number = existing + 1
    total_generated = 0

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}")) as progress:

        for source in project.synthesis.sources:
            total_clips = len(source.selected_voices) * len(source.speeds)
            task = progress.add_task(f"[cyan]{source.type}[/cyan]", total=total_clips)

            for voice in source.selected_voices:
                for speed in source.speeds:
                    out_path = sintetizados / f"{clip_number:03d}.wav"
                    try:
                        if source.type == "piper":
                            synthesize_piper(project.wake_word, voice, out_path,
                                             speed=speed, piper_binary=source.binary or "piper/piper")
                        elif source.type == "openai":
                            token = _resolve_token(source)
                            asyncio.run(synthesize_openai(project.wake_word, voice,
                                                          source.url, token, out_path, speed))
                        clip_number += 1
                        total_generated += 1
                    except Exception as exc:
                        console.print(f"  [red]x {source.type}/{voice}/{speed}:[/red] {exc}")
                    finally:
                        progress.advance(task)

    project.synthesis.clips = existing + total_generated
    project.synthesis.status = "done"
    save_project(project)
    return total_generated


def _resolve_token(source: TtsSource) -> str:
    import os
    if source.token_env:
        token = os.environ.get(source.token_env, "")
        if token:
            return token
    return console.input(f"  Token para {source.url}: ").strip()
