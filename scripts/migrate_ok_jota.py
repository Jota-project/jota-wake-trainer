# scripts/migrate_ok_jota.py
"""
Script de migración one-shot: mueve la estructura plana de ok_jota
a projects/ok_jota/ compatible con el nuevo sistema.
Idempotente — si el proyecto ya existe, no hace nada.
"""
from pathlib import Path
import shutil, json
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
PROJECTS_ROOT = ROOT / "projects"
TARGET = PROJECTS_ROOT / "ok_jota"
SESSION = TARGET / "session.json"


def migrate():
    if SESSION.exists():
        print("✓ projects/ok_jota/ ya existe — sin cambios.")
        return

    print("Migrando ok_jota a projects/ok_jota/ ...")

    # Crear estructura
    (TARGET / "data" / "positivos").mkdir(parents=True, exist_ok=True)
    (TARGET / "data" / "sintetizados").mkdir(parents=True, exist_ok=True)
    (TARGET / "models").mkdir(parents=True, exist_ok=True)

    # Mover datos si existen
    for src_name, dst_name in [
        ("data/positivos", "data/positivos"),
        ("data/sintetizados", "data/sintetizados"),
    ]:
        src = ROOT / src_name
        dst = TARGET / dst_name
        if src.exists() and any(src.iterdir()):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  ✓ {src_name} → projects/ok_jota/{dst_name}")

    # Mover modelo si existe
    model_src = ROOT / "models" / "ok_jota.tflite"
    if model_src.exists():
        shutil.copy2(model_src, TARGET / "models" / "ok_jota.tflite")
        print("  ✓ models/ok_jota.tflite → projects/ok_jota/models/ok_jota.tflite")

    # Crear session.json desde configs/ok_jota.yaml
    session = {
        "wake_word": "ok jota",
        "model_name": "ok_jota",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "voices": [],
        "synthesis": {"status": "pending", "clips": 0, "sources": []},
        "training": {"status": "pending", "epochs_completed": 0},
        "model_path": None,
    }

    # Detectar voces existentes en positivos migrados
    positivos_dir = TARGET / "data" / "positivos"
    for person_dir in sorted(positivos_dir.iterdir()):
        if person_dir.is_dir():
            clips = list(person_dir.glob("*.wav"))
            session["voices"].append({
                "name": person_dir.name,
                "mode": "record",
                "clips": len(clips),
                "status": "done" if len(clips) >= 30 else "pending",
            })

    SESSION.write_text(json.dumps(session, indent=2, ensure_ascii=False))
    print(f"  ✓ session.json creado con {len(session['voices'])} voces detectadas")
    print("\n✅ Migración completada. Usa `wake-trainer status ok_jota` para ver el estado.")


if __name__ == "__main__":
    migrate()
