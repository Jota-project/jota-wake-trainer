# tests/test_trainer_core.py
import logging
import warnings
import numpy as np
import pytest
from pathlib import Path
from trainer.trainer_core import (
    collect_positive_clips,
    TrainingConfig,
    run_training,
    _split_train_val,
    _reshape_windows,
    _validate_feature_file,
    _silence_known_noisy_warnings,
    _configure_training_logging,
    _compute_n_per_class,
    MIN_NEGATIVE_CLIPS,
)
from trainer.state import Project, Voice, SynthesisState, TrainingState


def _make_project(root: Path, monkeypatch, wake_word: str = "ok jota") -> Project:
    # Los paths de Project son relativos a PROJECTS_ROOT ("projects/<model>"),
    # así que para los tests apuntamos trainer.state.PROJECTS_ROOT a tmp_path
    # (con monkeypatch, para que se revierta solo al terminar el test).
    import trainer.state as state_mod
    monkeypatch.setattr(state_mod, "PROJECTS_ROOT", root)

    return Project(
        wake_word=wake_word,
        model_name="ok_jota",
        created_at="2026-01-01T00:00:00+00:00",
        voices=[Voice(name="Sito")],
        synthesis=SynthesisState(),
        training=TrainingState(),
    )


def test_collect_positive_clips_finds_wavs(tmp_path):
    person_dir = tmp_path / "Alfonso"
    person_dir.mkdir()
    (person_dir / "001.wav").touch()
    (person_dir / "002.wav").touch()
    (tmp_path / "ignored.txt").touch()

    clips = collect_positive_clips(tmp_path)
    assert len(clips) == 2
    assert all(str(c).endswith(".wav") for c in clips)


def test_collect_positive_clips_recurses_subdirectories(tmp_path):
    (tmp_path / "persona1").mkdir()
    (tmp_path / "persona2").mkdir()
    (tmp_path / "persona1" / "001.wav").touch()
    (tmp_path / "persona2" / "001.wav").touch()
    (tmp_path / "persona2" / "002.wav").touch()

    clips = collect_positive_clips(tmp_path)
    assert len(clips) == 3


def test_collect_positive_clips_empty_dir(tmp_path):
    assert collect_positive_clips(tmp_path) == []


def test_collect_positive_clips_missing_dir(tmp_path):
    assert collect_positive_clips(tmp_path / "no_existe") == []


def test_training_config_defaults():
    cfg = TrainingConfig(model_name="ok_jota", output_dir=Path("models"))
    assert cfg.negative_mode == "quick"
    assert cfg.window_steps == 16
    assert cfg.val_fraction == 0.15


def test_training_config_matches_openwakeword_own_defaults():
    """
    Regresión: steps=5000/layer_size=32 producían un modelo de 207 KB que
    nunca detectaba la wake word (ver known-bugs.md, bug #5). Deben
    coincidir con los defaults de la propia librería
    (openwakeword.train.Model layer_dim=128, Model.auto_train steps=50000),
    no con valores arbitrarios más pequeños para que las pruebas vayan rápido.
    """
    cfg = TrainingConfig(model_name="ok_jota", output_dir=Path("models"))
    assert cfg.steps == 50000
    assert cfg.layer_size == 128


def test_training_config_augmentation_rounds_bumped_after_confirmed_recall_zero():
    """
    Regresión: con n_per_class ya arreglado (ver test_compute_n_per_class_*)
    y logging activado, un reentrenamiento real siguió dando
    'Final Model Recall: 0.0'. Investigado hasta la fuente: openWakeWord
    necesita típicamente 20.000-50.000 clips positivos (confirmado por el
    propio mantenedor en GitHub), y este proyecto solo tiene ~195 antes de
    augmentar. augmentation_rounds=8 (1.328 instancias) era claramente
    insuficiente; 40 (~6.640) es el paliativo barato inmediato.
    """
    cfg = TrainingConfig(model_name="ok_jota", output_dir=Path("models"))
    assert cfg.augmentation_rounds == 40


def test_split_train_val_holds_out_fraction():
    clips = [Path(f"{i}.wav") for i in range(20)]
    train, val = _split_train_val(clips, val_fraction=0.2)
    assert len(val) == 4
    assert len(train) == 16
    assert set(train).isdisjoint(set(val))


def test_split_train_val_too_few_clips_returns_empty_val():
    clips = [Path(f"{i}.wav") for i in range(3)]
    train, val = _split_train_val(clips, val_fraction=0.2)
    assert val == []
    assert len(train) == 3


def test_reshape_windows_passthrough_when_already_correct_shape():
    x = np.zeros((10, 16, 96), dtype=np.float32)
    result = _reshape_windows(x, n=16)
    assert result is x


def test_reshape_windows_regroups_longer_sequences():
    # 4 clips de 32 pasos cada uno -> debe poder partirse en ventanas de 16
    x = np.zeros((4, 32, 96), dtype=np.float32)
    result = _reshape_windows(x, n=16)
    assert result.shape[1:] == (16, 96)
    assert result.shape[0] > 0


def test_reshape_windows_raises_when_not_enough_data():
    x = np.zeros((1, 5, 96), dtype=np.float32)
    with pytest.raises(ValueError):
        _reshape_windows(x, n=16)


def test_validate_feature_file_passes_for_valid_3d_array(tmp_path):
    p = tmp_path / "ok.npy"
    np.save(p, np.zeros((10, 16, 96), dtype=np.float32))
    _validate_feature_file(p, "positive")  # no debe lanzar


def test_validate_feature_file_raises_for_2d_array(tmp_path):
    p = tmp_path / "bad_2d.npy"
    np.save(p, np.zeros((10, 96), dtype=np.float32))
    with pytest.raises(ValueError, match="se esperaba"):
        _validate_feature_file(p, "positive")


def test_validate_feature_file_raises_for_empty_array(tmp_path):
    p = tmp_path / "empty.npy"
    np.save(p, np.zeros((0, 16, 96), dtype=np.float32))
    with pytest.raises(ValueError, match="se esperaba"):
        _validate_feature_file(p, "adversarial_negative")


def test_min_negative_clips_threshold_is_positive():
    assert MIN_NEGATIVE_CLIPS > 0


def test_compute_n_per_class_ignores_disk_size_ratio():
    """
    Regresión de la causa raíz confirmada del bug real: 'Final Model Recall'
    salía en 0.0 durante TODO el entrenamiento (no solo mal calibrado al
    final) porque, sin n_per_class explícito, mmap_batch_generator reparte
    cada batch proporcional al número de filas en disco de cada clase — con
    negative_mode='full' (ACAV100M) el negativo general tiene órdenes de
    magnitud más filas que los positivos reales, así que 'positive' se
    quedaba con ~1 muestra de 128 por batch durante los 50.000+ pasos
    enteros, y la señal positiva nunca llegaba a aprenderse. Esta función NO
    debe mirar el tamaño real de los ficheros — el reparto es fijo, por
    número de clases, con independencia de qué tan desproporcionado sea el
    dataset de negativos en disco.
    """
    files = {
        "positive": "positive.npy",
        "general_negative": "general_negative.npy",
        "adversarial_negative": "adversarial_negative.npy",
    }
    result = _compute_n_per_class(files, batch_size=126)
    assert result == {"positive": 42, "general_negative": 42, "adversarial_negative": 42}


def test_compute_n_per_class_without_adversarial_negative():
    files = {"positive": "positive.npy", "general_negative": "general_negative.npy"}
    result = _compute_n_per_class(files, batch_size=128)
    assert result == {"positive": 64, "general_negative": 64}


def test_compute_n_per_class_never_gives_zero_even_with_many_classes():
    files = {f"class_{i}": f"{i}.npy" for i in range(10)}
    result = _compute_n_per_class(files, batch_size=4)
    assert all(n >= 1 for n in result.values())


def test_configure_training_logging_enables_info_level():
    """
    Regresión: openwakeword.train.Model reporta sus métricas reales
    (incluida la línea "Final Model Accuracy: X") por logging.info(), pero
    el logging raíz de Python es WARNING por defecto — sin configurar esto,
    esas líneas se descartaban en silencio y no había forma de saber, mirando
    el log de un entrenamiento, si el modelo aprendió algo de verdad.
    """
    logging.getLogger().setLevel(logging.WARNING)
    try:
        _configure_training_logging()
        assert logging.getLogger().getEffectiveLevel() <= logging.INFO
    finally:
        logging.getLogger().setLevel(logging.WARNING)


def test_prepare_noise_augmentation_falls_back_silently_on_error(monkeypatch):
    """
    Si falta la dependencia 'datasets', no hay red, o HuggingFace está
    caído, _prepare_noise_augmentation no debe romper el entrenamiento —
    solo avisar y devolver listas vacías (el mismo comportamiento que había
    antes de que existiera la augmentación con audio real).
    """
    from trainer.trainer_core import _prepare_noise_augmentation
    import trainer.noise_data as noise_data_mod

    def boom():
        raise ImportError("No module named 'datasets'")

    monkeypatch.setattr(noise_data_mod, "ensure_rir_clips", lambda: boom())
    monkeypatch.setattr(noise_data_mod, "ensure_background_noise_clips", lambda full: boom())
    rir_paths, background_paths = _prepare_noise_augmentation(full=False)
    assert rir_paths == []
    assert background_paths == []


def test_prepare_noise_augmentation_keeps_rir_when_only_background_fails(monkeypatch):
    """
    Regresión del bug real que se dio en producción: el RIR (270 clips del
    MIT) se descargó bien, pero el ruido de fondo (FMA) falló con
    'Cannot seek streaming HTTP file' — y como antes ambos vivían en el
    mismo try/except, el fallo del segundo tiraba también el resultado ya
    válido del primero. Deben tratarse de forma independiente.
    """
    from trainer.trainer_core import _prepare_noise_augmentation
    import trainer.noise_data as noise_data_mod

    monkeypatch.setattr(noise_data_mod, "ensure_rir_clips", lambda: ["rir1.wav", "rir2.wav"])

    def boom_background(full):
        raise RuntimeError("Cannot seek streaming HTTP file")

    monkeypatch.setattr(noise_data_mod, "ensure_background_noise_clips", boom_background)

    rir_paths, background_paths = _prepare_noise_augmentation(full=False)
    assert rir_paths == ["rir1.wav", "rir2.wav"]
    assert background_paths == []


def test_prepare_noise_augmentation_returns_real_paths_when_available(monkeypatch):
    from trainer.trainer_core import _prepare_noise_augmentation
    import trainer.noise_data as noise_data_mod

    monkeypatch.setattr(noise_data_mod, "ensure_rir_clips", lambda: ["a.wav", "b.wav"])
    monkeypatch.setattr(noise_data_mod, "ensure_background_noise_clips", lambda full: ["c.wav"])
    rir_paths, background_paths = _prepare_noise_augmentation(full=False)
    assert rir_paths == ["a.wav", "b.wav"]
    assert background_paths == ["c.wav"]


def test_synthesize_negative_clips_uses_catalog_and_is_idempotent(tmp_path, monkeypatch):
    from trainer.trainer_core import _synthesize_negative_clips
    from trainer.state import TtsSource
    import trainer.synthesizer as synth_mod

    calls = []

    def fake_piper(text, voice, out_path, speed=1.0, piper_binary=None):
        calls.append(text)
        out_path.write_bytes(b"fake-wav-bytes")

    monkeypatch.setattr(synth_mod, "synthesize_piper", fake_piper)

    project = _make_project(tmp_path, monkeypatch)
    project.synthesis.sources = [
        TtsSource(type="piper", voices_dir="piper/voices", selected_voices=["voz1"], speeds=[1.0])
    ]
    out_dir = tmp_path / "negativos"

    clips1 = _synthesize_negative_clips(project, out_dir, n_phrases=10)
    assert len(calls) == 10
    assert len(clips1) == 10
    assert (out_dir / ".negative_catalog.json").exists()

    # Repetir con el mismo n_phrases no debe volver a sintetizar nada — ya
    # están en el catálogo (a diferencia del check antiguo de "cuenta total
    # de la carpeta", esto no depende de qué otros ficheros haya en out_dir).
    calls.clear()
    clips2 = _synthesize_negative_clips(project, out_dir, n_phrases=10)
    assert len(calls) == 0
    assert len(clips2) == 10

    # Negativos importados a mano en la misma carpeta no deben bloquear la
    # síntesis de frases nuevas (el bug que reportó el usuario).
    (out_dir / "neg_hand_001.wav").write_bytes(b"hand-imported")

    # Aumentar n_phrases genera solo las frases nuevas que faltan.
    clips3 = _synthesize_negative_clips(project, out_dir, n_phrases=15)
    assert len(calls) == 5
    assert len(clips3) == 16  # 10 + 1 importado a mano + 5 nuevos


def test_synthesize_negative_clips_dispatches_to_google(tmp_path, monkeypatch):
    """Regresión de wiring: un source type='google' debe usar synthesize_google
    (con el token resuelto), no caer en la rama openai/elevenlabs por defecto."""
    from trainer.trainer_core import _synthesize_negative_clips
    from trainer.state import TtsSource
    import trainer.synthesizer as synth_mod
    from unittest.mock import AsyncMock

    calls = []

    async def fake_google(text, voice, token, out_path, speed=1.0):
        calls.append((text, voice, token))
        out_path.write_bytes(b"fake-wav-bytes")

    monkeypatch.setattr(synth_mod, "synthesize_google", AsyncMock(side_effect=fake_google))
    monkeypatch.setattr(synth_mod, "_resolve_token", lambda source: "dummy-google-key")

    project = _make_project(tmp_path, monkeypatch)
    project.synthesis.sources = [
        TtsSource(type="google", token_env="GOOGLE_TTS_API_KEY",
                 selected_voices=["es-ES-Neural2-A"], speeds=[1.0])
    ]
    out_dir = tmp_path / "negativos"

    clips = _synthesize_negative_clips(project, out_dir, n_phrases=5)

    assert len(calls) == 5
    assert len(clips) == 5
    assert all(c[2] == "dummy-google-key" for c in calls)


def test_run_training_raises_when_no_positive_clips(tmp_path, monkeypatch):
    project = _make_project(tmp_path, monkeypatch)
    cfg = TrainingConfig(model_name="ok_jota", output_dir=tmp_path / "models")

    with pytest.raises(ValueError, match="clips positivos"):
        run_training(project, cfg)


# ── _silence_known_noisy_warnings ────────────────────────────────────────────
#
# Los tres warnings que revisamos con Sito tras el primer entrenamiento
# exitoso (pkg_resources/torchmetrics, dtype float64/audiomentations, y el
# ya existente output_type/torch_audiomentations) son puramente informativos
# — investigados uno por uno, ninguno indica pérdida de datos ni un problema
# real. Se silencian por mensaje+módulo exactos, nunca por categoría global,
# para no ocultar avisos nuevos que sí podrían importar.

def _warn_from(message: str, category, module: str):
    warnings.warn_explicit(message, category, filename=f"{module}.py", lineno=1, module=module)


def test_silences_torchmetrics_pkg_resources_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _silence_known_noisy_warnings()
        _warn_from(
            "pkg_resources is deprecated as an API. See https://setuptools.pypa.io/...",
            UserWarning, "torchmetrics.utilities.imports",
        )
    assert len(caught) == 0


def test_silences_audiomentations_float64_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _silence_known_noisy_warnings()
        _warn_from(
            "Warning: input samples dtype is np.float64. Converting to np.float32",
            UserWarning, "audiomentations.core.transforms_interface",
        )
    assert len(caught) == 0


def test_silences_torch_audiomentations_output_type_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _silence_known_noisy_warnings()
        _warn_from(
            "Transforms now expect an `output_type` argument",
            FutureWarning, "torch_audiomentations.utils.object_dict",
        )
    assert len(caught) == 0


def test_does_not_silence_unrelated_warnings():
    """
    Regresión de que el silenciado es específico (mensaje+módulo), no una
    categoría global — un warning nuevo y genuinamente distinto debe seguir
    apareciendo.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _silence_known_noisy_warnings()
        _warn_from("Something genuinely new and unexpected", UserWarning, "some.other.module")
    assert len(caught) == 1


def test_run_training_raises_helpful_error_when_deps_missing(tmp_path, monkeypatch):
    import trainer.trainer_core as trainer_core_mod

    monkeypatch.setattr(trainer_core_mod, "HAS_TRAIN_DEPS", False)

    project = _make_project(tmp_path, monkeypatch)
    (project.positivos_path / "Sito").mkdir(parents=True)
    (project.positivos_path / "Sito" / "001.wav").touch()

    cfg = TrainingConfig(model_name="ok_jota", output_dir=tmp_path / "models")

    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[train\]'"):
        run_training(project, cfg)


def test_run_training_warns_when_positive_dataset_far_below_recommended_scale(tmp_path, monkeypatch, capsys):
    """
    Regresión: confirmamos (con evidencia de GitHub, no solo teoría) que
    openWakeWord necesita típicamente 20.000-50.000 clips positivos y que
    con ~195 (nuestro caso real) el modelo puede quedarse en recall 0 pase
    lo que pase con pasos/capacidad/pesos de negativos. Este aviso debe
    aparecer SIEMPRE que el dataset esté muy por debajo de esa escala, no
    solo cuando ya hemos visto fallar el entrenamiento — para que el
    problema se detecte antes de gastar un entrenamiento entero.
    """
    import trainer.trainer_core as trainer_core_mod

    monkeypatch.setattr(trainer_core_mod, "HAS_TRAIN_DEPS", False)

    project = _make_project(tmp_path, monkeypatch)
    (project.positivos_path / "Sito").mkdir(parents=True)
    (project.positivos_path / "Sito" / "001.wav").touch()

    cfg = TrainingConfig(model_name="ok_jota", output_dir=tmp_path / "models")

    with pytest.raises(RuntimeError):
        run_training(project, cfg)

    output = capsys.readouterr().out
    assert "20.000-50.000" in output
    assert "1 clips positivos" in output
