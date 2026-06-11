# trainer/trainer_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from rich.console import Console

console = Console()

try:
    import openwakeword.train as oww_train
except ImportError:
    oww_train = None  # type: ignore — se instala con pip install ".[train]"


@dataclass
class TrainingConfig:
    model_name: str
    output_dir: Path
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    augmentation_factor: int = 10


def collect_positive_clips(positivos_path: Path) -> list[Path]:
    """Devuelve todos los WAVs bajo positivos_path (recursivo)."""
    return sorted(positivos_path.rglob("*.wav"))


def run_training(positivos_path: Path, config: TrainingConfig) -> Path:
    """
    Lanza el entrenamiento con openWakeWord.
    Devuelve la ruta al .tflite resultante.
    """
    if oww_train is None:
        raise RuntimeError(
            "openWakeWord no está instalado con soporte de entrenamiento. "
            "Ejecuta: pip install openWakeWord[train]"
        )

    clips = collect_positive_clips(positivos_path)
    if not clips:
        raise ValueError(f"No se encontraron clips en {positivos_path}")

    console.print(f"\n  [dim]Clips positivos:[/dim] {len(clips)}")
    console.print(f"  [dim]Dataset estimado tras augmentación:[/dim] ~{len(clips) * config.augmentation_factor}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    oww_train.train_custom_model(
        custom_model_data={
            "positive_data": [str(c) for c in clips],
        },
        model_name=config.model_name,
        output_dir=str(config.output_dir),
        num_steps=config.epochs * 100,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
    )

    tflite_path = config.output_dir / f"{config.model_name}.tflite"
    if not tflite_path.exists():
        candidates = list(config.output_dir.rglob("*.tflite"))
        if candidates:
            tflite_path = candidates[0]

    return tflite_path
