# trainer/evaluator.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import openwakeword
    import soundfile as sf
except ImportError:
    openwakeword = None  # type: ignore
    sf = None  # type: ignore


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    false_positives: int
    threshold: float

    def passed(self) -> bool:
        return self.precision >= 0.9 and self.recall >= 0.85


def evaluate_model(
    model_path: Path,
    positivos_path: Path,
    threshold: float = 0.3,
) -> EvaluationResult:
    """
    Evalúa el modelo cargando todos los WAVs positivos y calculando
    precisión y recall. Los falsos positivos se miden en fragmentos
    de audio ambiente (silencio sintético).
    """
    if openwakeword is None:
        raise RuntimeError("openWakeWord no instalado.")

    model = openwakeword.Model(
        wakeword_models=[str(model_path)],
        inference_framework="tflite",
    )
    model_key = model_path.stem

    clips = sorted(positivos_path.rglob("*.wav"))
    true_positives = 0
    false_negatives = 0

    for clip_path in clips:
        data, sr = sf.read(str(clip_path), dtype="int16", always_2d=False)
        scores = model.predict(data)
        max_score = max(scores.get(model_key, [0.0]))
        if max_score >= threshold:
            true_positives += 1
        else:
            false_negatives += 1

    total = len(clips) or 1
    # Test con 10s de silencio sintético (ceros)
    silence = np.zeros(16000 * 10, dtype=np.int16)
    silence_scores = model.predict(silence)
    fp = sum(1 for s in silence_scores.get(model_key, []) if s >= threshold)

    precision = true_positives / max(true_positives + fp, 1)
    recall = true_positives / total

    return EvaluationResult(
        precision=round(precision, 3),
        recall=round(recall, 3),
        false_positives=fp,
        threshold=threshold,
    )
