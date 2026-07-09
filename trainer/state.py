from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal
import json
from datetime import datetime, timezone

PROJECTS_ROOT = Path("projects")
SESSION_FILE = "session.json"


class ProjectNotFoundError(FileNotFoundError):
    """
    Se lanza cuando `load_project` no encuentra el proyecto pedido.

    Por qué existe: antes esto era un `FileNotFoundError` crudo de
    `Path.read_text()`, que se propagaba tal cual hasta la terminal como un
    traceback completo de varias pantallas — por ejemplo al pasar por error
    una ruta de fichero ('projects/ok_jota/models/ok_jota.tflite') en vez del
    nombre del proyecto ('ok_jota'), que además se concatenaba con
    PROJECTS_ROOT dando un path sin sentido ('projects/projects/...'). El
    error de fondo era trivial (nombre de proyecto equivocado) pero se veía
    como si algo grave hubiera petado. Este subtipo lleva un mensaje ya
    pensado para mostrarse directamente al usuario (sin traceback), y
    trainer/cli.py lo captura en cada comando para eso.
    """


def _looks_like_a_path(model_name: str) -> bool:
    return (
        "/" in model_name
        or "\\" in model_name
        or model_name.endswith((".tflite", ".onnx", ".json", ".wav"))
    )


def _guess_intended_model_name(model_name: str) -> Optional[str]:
    """Si `model_name` parece una ruta que incluye 'projects/<algo>/...',
    intenta recuperar el nombre de proyecto real que el usuario quería."""
    parts = Path(model_name).parts
    if "projects" in parts:
        idx = parts.index("projects")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # Sin 'projects/' de por medio: el primer segmento de la ruta suele ser
    # el propio nombre (p.ej. 'ok_jota/models/ok_jota.tflite').
    if len(parts) > 1:
        return parts[0]
    return None


def available_project_names() -> list[str]:
    if not PROJECTS_ROOT.exists():
        return []
    return sorted(
        p.name for p in PROJECTS_ROOT.iterdir() if (p / SESSION_FILE).exists()
    )


def _project_not_found_message(model_name: str) -> str:
    lines = [f"No se encontró el proyecto '{model_name}'."]

    if _looks_like_a_path(model_name):
        lines.append(
            "Este comando espera solo el NOMBRE del proyecto (el mismo que "
            "aparece en 'wake-trainer status'), no una ruta a un fichero ni "
            "a la carpeta del modelo."
        )
        guess = _guess_intended_model_name(model_name)
        if guess and guess != model_name:
            lines.append(f"¿Quisiste decir '{guess}'?")

    available = available_project_names()
    if available:
        lines.append("Proyectos existentes: " + ", ".join(available))
    else:
        lines.append("No hay ningún proyecto creado todavía.")

    return "\n".join(lines)


@dataclass
class TtsSource:
    type: Literal["piper", "openai", "google"]
    binary: Optional[str] = None
    voices_dir: Optional[str] = None
    url: Optional[str] = None
    token_env: Optional[str] = None
    token_header: Optional[str] = None
    selected_voices: list[str] = field(default_factory=list)
    speeds: list[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])


@dataclass
class Voice:
    name: str
    mode: Optional[Literal["record", "import"]] = None
    clips: int = 0
    status: Literal["pending", "done"] = "pending"


@dataclass
class SynthesisState:
    status: Literal["pending", "in_progress", "done"] = "pending"
    clips: int = 0
    sources: list[TtsSource] = field(default_factory=list)


@dataclass
class TrainingState:
    status: Literal["pending", "in_progress", "done"] = "pending"
    epochs_completed: int = 0


@dataclass
class Project:
    wake_word: str
    model_name: str
    created_at: str
    voices: list[Voice]
    synthesis: SynthesisState
    training: TrainingState
    model_path: Optional[str] = None

    @property
    def root(self) -> Path:
        return PROJECTS_ROOT / self.model_name

    @property
    def positivos_path(self) -> Path:
        return self.root / "data" / "positivos"

    @property
    def sintetizados_path(self) -> Path:
        return self.root / "data" / "sintetizados"

    @property
    def negativos_path(self) -> Path:
        return self.root / "data" / "negativos"

    @property
    def models_path(self) -> Path:
        return self.root / "models"


def _to_dict(project: Project) -> dict:
    return asdict(project)


def _from_dict(d: dict) -> Project:
    d = dict(d)  # copia superficial para no mutar el argumento
    voices = [Voice(**v) for v in d.pop("voices", [])]
    synth_raw = d.pop("synthesis", {})
    sources = [TtsSource(**s) for s in synth_raw.pop("sources", [])]
    synthesis = SynthesisState(**synth_raw, sources=sources)
    training = TrainingState(**d.pop("training", {}))
    return Project(**d, voices=voices, synthesis=synthesis, training=training)


def load_project(model_name: str) -> Project:
    path = PROJECTS_ROOT / model_name / SESSION_FILE
    if not path.exists():
        raise ProjectNotFoundError(_project_not_found_message(model_name))
    return _from_dict(json.loads(path.read_text()))


def save_project(project: Project) -> None:
    project.root.mkdir(parents=True, exist_ok=True)
    path = project.root / SESSION_FILE
    path.write_text(json.dumps(_to_dict(project), indent=2, ensure_ascii=False))


def list_projects() -> list[Project]:
    if not PROJECTS_ROOT.exists():
        return []
    result = []
    for session_path in sorted(PROJECTS_ROOT.glob(f"*/{SESSION_FILE}")):
        try:
            result.append(load_project(session_path.parent.name))
        except Exception:
            pass
    return result


def create_project(wake_word: str, model_name: str, voice_names: list[str]) -> Project:
    session_path = PROJECTS_ROOT / model_name / SESSION_FILE
    if session_path.exists():
        raise FileExistsError(
            f"El proyecto '{model_name}' ya existe en {session_path}. "
            "Usa load_project() para cargarlo."
        )
    project = Project(
        wake_word=wake_word,
        model_name=model_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        voices=[Voice(name=n) for n in voice_names],
        synthesis=SynthesisState(),
        training=TrainingState(),
    )
    project.positivos_path.mkdir(parents=True, exist_ok=True)
    project.sintetizados_path.mkdir(parents=True, exist_ok=True)
    project.models_path.mkdir(parents=True, exist_ok=True)
    for voice in project.voices:
        (project.positivos_path / voice.name).mkdir(exist_ok=True)
    save_project(project)
    return project


AUGMENTATION_FACTOR = 10
MINIMUM_DATASET_SIZE = 1000
CLIPS_PER_VOICE = 30


def calculate_dataset(project: Project) -> dict:
    real_clips = len(project.voices) * CLIPS_PER_VOICE
    synth_clips = sum(
        len(src.selected_voices) * len(src.speeds)
        for src in project.synthesis.sources
    )
    real_augmented = real_clips * AUGMENTATION_FACTOR
    synth_augmented = synth_clips * AUGMENTATION_FACTOR
    total = real_augmented + synth_augmented
    return {
        "real_clips": real_clips,
        "synth_clips": synth_clips,
        "real_augmented": real_augmented,
        "synth_augmented": synth_augmented,
        "total": total,
        "meets_minimum": total >= MINIMUM_DATASET_SIZE,
    }
