# trainer/evaluator.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time
import numpy as np
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

try:
    import openwakeword
    import soundfile as sf
except ImportError:
    openwakeword = None  # type: ignore
    sf = None  # type: ignore


# Barrido de thresholds que se reporta siempre, además del "primario". Cubre
# el rango típico que se usa en la práctica (0.5 es el valor por defecto más
# habitual en configuraciones de Home Assistant/Wyoming; el resto sirve para
# ver cómo se mueven precisión/recall/FP si se sube o baja la sensibilidad
# sin tener que volver a evaluar el modelo entero por cada valor).
DEFAULT_THRESHOLD_SWEEP: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _max_score_over_clip(model, data: np.ndarray, model_key: str) -> float:
    """
    Devuelve el score máximo del modelo a lo largo de todo el clip.

    Por qué existe: `model.predict(x)` de openWakeWord NO devuelve una lista
    de scores por frame — devuelve un único float por modelo (ver
    `openwakeword/model.py::Model.predict`, línea final de cada iteración:
    `predictions[mdl] = prediction[0][0][0]`), incluso cuando `x` es un clip
    entero de varios segundos (en ese caso el propio `predict()` ya colapsa
    internamente todos los frames con `.max(axis=0)` y solo te da el máximo).
    El código anterior hacía `max(scores.get(model_key, [0.0]))` asumiendo
    que era una lista — funcionaba por casualidad con el valor por defecto
    `[0.0]` (sí es iterable), pero en cuanto el modelo tenía una entrada real
    para `model_key` (un float), `max(float)` explota con
    `TypeError: 'float' object is not iterable`. Aquí usamos
    `model.predict_clip()`, que si itera internamente frame a frame (80 ms)
    y devuelve una lista de dicts — la forma correcta de recorrer un clip
    completo — y reseteamos el estado del modelo antes de cada clip para que
    no arrastre buffers de predicción del clip anterior.
    """
    model.reset()
    frame_predictions = model.predict_clip(data)
    return max((p.get(model_key, 0.0) for p in frame_predictions), default=0.0)


def _detect_runtime_used(model, model_key: str) -> str:
    """
    openwakeword.Model puede pedir 'tflite' y caer solo a 'onnx' por dentro
    si no encuentra el paquete `tflite_runtime` instalado — con un simple
    `logging.warning` fácil de pasar por alto, sin exponer el framework
    final como atributo público. Lo inferimos mirando el tipo real del
    objeto ya cargado para este modelo: onnxruntime usa `InferenceSession`,
    tflite usa `Interpreter`. Así el informe dice siempre la verdad sobre
    qué runtime se usó de verdad, en vez de asumir el que se pidió.
    """
    loaded = model.models.get(model_key)
    if loaded is None:
        return "desconocido"
    return "onnx" if "InferenceSession" in type(loaded).__name__ else "tflite"


def _score_clips_with_progress(
    model, model_key: str, clips: list[Path], description: str
) -> tuple[list[float], float, float]:
    """
    Devuelve (scores, segundos_de_audio_totales, segundos_de_proceso_totales)
    para una lista de clips, con una barra de progreso — recorrer cada clip
    frame a frame puede tardar de verdad (más aún si se ha caído a
    onnxruntime en vez de tflite) y sin progreso visible se ve exactamente
    igual que un cuelgue, como ya ha pasado antes con otros pasos largos de
    este proyecto (síntesis de negativos, descarga de ruido de fondo...).
    """
    scores: list[float] = []
    total_audio_seconds = 0.0
    total_processing_seconds = 0.0
    with Progress(
        SpinnerColumn(), TextColumn("[dim]{task.description}[/dim]"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task(description, total=len(clips) or 1)
        for clip_path in clips:
            data, sr = sf.read(str(clip_path), dtype="int16", always_2d=False)
            start = time.perf_counter()
            score = _max_score_over_clip(model, data, model_key)
            total_processing_seconds += time.perf_counter() - start
            total_audio_seconds += len(data) / sr
            scores.append(score)
            progress.advance(task)
    return scores, total_audio_seconds, total_processing_seconds


@dataclass
class ThresholdStats:
    """Métricas para un valor de threshold concreto, calculadas a partir de
    los scores ya obtenidos (no vuelve a correr el modelo)."""
    threshold: float
    recall: float          # sensibilidad: % de positivos que el modelo detecta
    precision: float        # % de detecciones que son de verdad la wake word
    false_positives: int
    accuracy: float          # % de clips (positivos + negativos) bien clasificados


@dataclass
class PerformanceStats:
    """Rendimiento/velocidad de inferencia, medido durante la propia evaluación."""
    runtime_used: str        # 'tflite' u 'onnx' — el real, no el pedido (ver _detect_runtime_used)
    total_audio_seconds: float
    total_processing_seconds: float
    real_time_factor: float  # cuántas veces más rápido que reproducir el audio en directo
    avg_ms_per_clip: float


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    false_positives: int
    threshold: float
    false_positive_note: str
    threshold_sweep: list[ThresholdStats] = field(default_factory=list)
    performance: Optional[PerformanceStats] = None

    def passed(self) -> bool:
        return self.precision >= 0.9 and self.recall >= 0.85


def _threshold_stats(threshold: float, positive_scores: list[float], negative_scores: list[float]) -> ThresholdStats:
    tp = sum(1 for s in positive_scores if s >= threshold)
    fp = sum(1 for s in negative_scores if s >= threshold)
    tn = len(negative_scores) - fp
    recall = tp / len(positive_scores) if positive_scores else 0.0
    precision = tp / max(tp + fp, 1)
    total = len(positive_scores) + len(negative_scores)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    return ThresholdStats(
        threshold=threshold,
        recall=round(recall, 3),
        precision=round(precision, 3),
        false_positives=fp,
        accuracy=round(accuracy, 3),
    )


def evaluate_model(
    model_path: Path,
    positivos_path: Path,
    negativos_path: Optional[Path] = None,
    threshold: float = 0.5,
    threshold_sweep: tuple[float, ...] = DEFAULT_THRESHOLD_SWEEP,
) -> EvaluationResult:
    """
    Evalúa el modelo cargando todos los WAVs positivos (recall) y, si existen,
    los clips negativos sintéticos generados durante el entrenamiento
    (precisión / falsos positivos). Esto es mucho más representativo que
    probar solo contra silencio digital, que casi nunca dispara ningún
    modelo y por tanto no dice nada sobre falsos positivos reales.

    Si no hay negativos disponibles (proyecto sin fuentes TTS configuradas),
    cae de vuelta a 10s de silencio como antes, dejando claro en el informe
    que la medida es débil.

    Cada clip se puntúa una sola vez (`_score_clips_with_progress`); a partir
    de esos scores ya calculados se derivan tanto el resultado "primario"
    (al `threshold` pedido) como el barrido completo en `threshold_sweep`,
    sin tener que volver a correr el modelo por cada valor de threshold.
    """
    if openwakeword is None:
        raise RuntimeError("openWakeWord no instalado.")

    framework = "tflite" if model_path.suffix == ".tflite" else "onnx"
    model = openwakeword.Model(
        wakeword_models=[str(model_path)],
        inference_framework=framework,
    )
    model_key = model_path.stem
    runtime_used = _detect_runtime_used(model, model_key)

    positive_clips = sorted(positivos_path.rglob("*.wav"))
    positive_scores, pos_audio_s, pos_proc_s = _score_clips_with_progress(
        model, model_key, positive_clips, "Evaluando positivos"
    )

    negative_clips = sorted(negativos_path.rglob("*.wav")) if negativos_path and negativos_path.exists() else []

    if negative_clips:
        negative_scores, neg_audio_s, neg_proc_s = _score_clips_with_progress(
            model, model_key, negative_clips, "Evaluando negativos"
        )
        weak_negative_measure = False
    else:
        # Sin negativos reales disponibles: fallback débil (casi nunca dispara nada).
        # Aquí sí tratamos cada frame de 80 ms como una observación independiente
        # (no un único máximo de clip), para no perder la cuenta de falsos
        # positivos repetidos dentro de los mismos 10 s.
        silence = np.zeros(16000 * 10, dtype=np.int16)
        model.reset()
        start = time.perf_counter()
        frame_predictions = model.predict_clip(silence)
        neg_proc_s = time.perf_counter() - start
        neg_audio_s = 10.0
        negative_scores = [p.get(model_key, 0.0) for p in frame_predictions]
        weak_negative_measure = True

    total_audio_s = pos_audio_s + neg_audio_s
    total_proc_s = pos_proc_s + neg_proc_s
    n_clips = len(positive_clips) + (len(negative_clips) if not weak_negative_measure else 1)
    performance = PerformanceStats(
        runtime_used=runtime_used,
        total_audio_seconds=round(total_audio_s, 1),
        total_processing_seconds=round(total_proc_s, 3),
        real_time_factor=round(total_audio_s / total_proc_s, 1) if total_proc_s > 0 else float("inf"),
        avg_ms_per_clip=round(1000 * total_proc_s / max(n_clips, 1), 1),
    )

    sweep = [_threshold_stats(th, positive_scores, negative_scores) for th in threshold_sweep]
    primary = _threshold_stats(threshold, positive_scores, negative_scores)

    if weak_negative_measure:
        note = (
            f"{primary.false_positives} en 10 s de silencio sintético "
            "(sin negativos reales — medida débil, poco fiable)"
        )
    else:
        note = f"{primary.false_positives} de {len(negative_clips)} clips negativos sintéticos (~{neg_audio_s:.0f} s)"

    return EvaluationResult(
        precision=primary.precision,
        recall=primary.recall,
        false_positives=primary.false_positives,
        threshold=threshold,
        false_positive_note=note,
        threshold_sweep=sweep,
        performance=performance,
    )
