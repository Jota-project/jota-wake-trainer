# trainer/recorder.py
from __future__ import annotations
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from rich import box
from rich.console import Console
from rich.panel import Panel

console = Console()

SAMPLE_RATE = 16000
CHANNELS = 1
SUBTYPE = "PCM_16"
RECORD_DURATION = 3.0
MIN_DURATION = 0.5
MIN_LEVEL = 0.01   # ~-40 dBFS
MAX_LEVEL = 0.99   # ~-0.1 dBFS

CONDITIONS: list[dict] = [
    {"id": 1,  "name": "Distancia normal · silencio",  "desc": "1-1.5 m del dispositivo, habitación en silencio",  "clips": 5},
    {"id": 2,  "name": "Distancia cercana · silencio", "desc": "30-50 cm del dispositivo, volumen normal",          "clips": 3},
    {"id": 3,  "name": "Distancia larga · voz alzada", "desc": "3-4 m del dispositivo, voz ligeramente más alta",  "clips": 3},
    {"id": 4,  "name": "Ruido TV/radio",               "desc": "TV o radio de fondo a volumen moderado",            "clips": 4},
    {"id": 5,  "name": "Ruido de conversación",        "desc": "Otra persona hablando en la misma habitación",      "clips": 3},
    {"id": 6,  "name": "Música de fondo",              "desc": "Música a volumen normal",                           "clips": 3},
    {"id": 7,  "name": "Voz rápida",                   "desc": "Dicho con prisa",                                   "clips": 3},
    {"id": 8,  "name": "Voz lenta",                    "desc": "Pausado, sobrearticulado",                          "clips": 2},
    {"id": 9,  "name": "Voz baja / susurro",           "desc": "Tono bajo, sin proyectar",                         "clips": 2},
    {"id": 10, "name": "Ángulo lateral",               "desc": "Hablando de lado al dispositivo (~45°)",            "clips": 2},
]


def validate_clip(path: Path) -> tuple[bool, str]:
    data, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        return False, f"Sample rate incorrecto: {sr} Hz (esperado {SAMPLE_RATE} Hz)"
    if data.ndim > 1 and data.shape[1] > 1:
        return False, "El audio no es mono"
    duration = len(data) / sr
    if duration < MIN_DURATION:
        return False, f"Clip demasiado corto: {duration:.2f}s (mínimo {MIN_DURATION}s)"
    flat = data.ravel()
    peak = float(np.abs(flat).max())
    if peak < MIN_LEVEL:
        return False, "Nivel de señal demasiado bajo — habla más fuerte o acércate"
    # Saturation: más del 1% de muestras en el límite de clipping
    clipped_fraction = float((np.abs(flat) >= MAX_LEVEL).sum()) / len(flat)
    if clipped_fraction > 0.01:
        return False, "Señal saturada — baja el volumen o aléjate del micrófono"
    return True, ""


def _record_raw(output_path: Path, countdown: int = 3) -> None:
    for i in range(countdown, 0, -1):
        console.print(f"  [dim]Preparado...[/dim]  [bold]{i}[/bold]", end="\r")
        time.sleep(1)
    console.print("  [bold red]● GRABANDO[/bold red]          ", end="\r")
    for attempt in range(3):
        try:
            audio = sd.rec(
                int(RECORD_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
            )
            sd.wait()
            break
        except sd.PortAudioError:
            if attempt == 2:
                raise
            # macOS CoreAudio puede reportar kAudioHardwareNotRunningError ('stop')
            # si otra app (Siri, dictado) acaba de liberar el micrófono.
            # Reinicializar PortAudio resuelve el estado inconsistente.
            sd._terminate()
            sd._initialize()
            time.sleep(0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, SAMPLE_RATE, subtype=SUBTYPE)


def record_voice(voice_dir: Path, wake_word: str) -> int:
    """
    Guía la grabación de los 30 clips para una voz.
    Devuelve el número de clips grabados correctamente.
    """
    voice_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(voice_dir.glob("*.wav"))
    clip_number = len(existing) + 1

    for cond in CONDITIONS:
        console.print(Panel(
            f"[bold]Condición {cond['id']}/10:[/bold] {cond['name']}\n"
            f"[dim]{cond['desc']}[/dim]",
            box=box.SIMPLE,
        ))
        clips_done = 0
        while clips_done < cond["clips"]:
            clip_path = voice_dir / f"{clip_number:03d}.wav"
            console.print(f"\n  Clip {clips_done+1}/{cond['clips']}")
            console.print(f'  Di claramente: [bold cyan]"{wake_word}"[/bold cyan]')

            _record_raw(clip_path)
            valid, error = validate_clip(clip_path)

            if not valid:
                clip_path.unlink(missing_ok=True)
                console.print(f"  [red]✗ {error}[/red] — reintentando...")
                time.sleep(0.5)
                continue

            data, _ = sf.read(str(clip_path), dtype="float32")
            dur = len(data) / SAMPLE_RATE
            console.print(f"  [green]✓ Guardado[/green]  ({dur:.2f}s · {SAMPLE_RATE}Hz · mono)")

            action = console.input("\n  [↵ continuar]  [r repetir]  [q pausar]: ").strip().lower()
            if action == "r":
                clip_path.unlink(missing_ok=True)
                continue
            if action == "q":
                return clip_number - 1

            clip_number += 1
            clips_done += 1

    return clip_number - 1
