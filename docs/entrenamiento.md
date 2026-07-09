# Cómo entrena esta herramienta (y qué cambió)

## El problema que había antes

`trainer/trainer_core.py` llamaba a `openwakeword.train.train_custom_model(...)`,
una función que no existe en openWakeWord (ni en 0.6.0 ni en versiones
anteriores). El diseño original asumía que openWakeWord generaba negativos y
aplicaba augmentación "internamente" a partir solo de clips positivos. Pero
openWakeWord entrena un **clasificador binario**: sin ejemplos negativos
(audio que *no* es la wake word) no hay nada que aprender, sea cual sea la
API que se use. Por eso el entrenamiento se quedaba siempre en `pending`.

## Qué hace ahora

`wake-trainer train <modelo>` sigue estos pasos:

1. **Features positivas** — tus clips grabados + sintéticos se aumentan
   (pitch, ruido, ecualización...) y se pasan por el extractor de
   features de openWakeWord (el mismo AudioSet embedding preinstalado).

2. **Negativos "duros"** — se sintetizan automáticamente con las mismas
   fuentes TTS que ya tengas configuradas (Piper/ElevenLabs/etc.), usando
   variaciones fonéticas cercanas a tu wake word (p. ej. para "ok jota":
   "ok rosa", "ok jose", "hola jota"...) más un conjunto de frases genéricas
   de uso habitual con un asistente. No hace falta grabar nada de esto a mano.

3. **Negativos generales** — audio diverso (voz, ruido, música) que no
   tiene nada que ver con tu wake word. En vez de recopilarlo y procesarlo
   nosotros, se usa el dataset de features precalculadas que publica el
   propio autor de openWakeWord en HuggingFace
   ([`davidscripka/openwakeword_features`](https://huggingface.co/datasets/davidscripka/openwakeword_features)):

   - **Modo `quick`** (por defecto): descarga solo el *set de validación de
     falsos positivos* (~11.3 h, ~190 MB) y lo reutiliza también como
     negativo de entrenamiento. Rápido, poco disco, modelo razonable para
     uso personal — pero menos diverso que el dataset completo.
   - **Modo `--full`**: descarga además ACAV100M completo (~2000 h, ~17 GB)
     como negativo de entrenamiento dedicado. Es la receta "oficial",
     más robusta frente a falsos positivos, pero la descarga inicial puede
     tardar horas según tu conexión.

4. **Entrenamiento** — `openwakeword.train.Model.auto_train(...)`, la misma
   rutina de 3 fases (con negative weighting creciente) que usa el notebook
   oficial. Es una red pequeña (una DNN de un par de capas sobre features de
   16×96), así que entrena en CPU en minutos, no en horas — Apple Silicon no
   aporta aceleración aquí porque openWakeWord no usa MPS, solo CUDA o CPU.

5. **Exportación** — el modelo se guarda en ONNX siempre. La conversión a
   `.tflite` (el formato que carga el addon openWakeWord de Wyoming/Home
   Assistant cuando corre en CPU, que es el caso normal en un Raspberry Pi
   o en el propio Home Assistant OS) usa `onnx2tf` en vez del
   `onnx-tf`/`tensorflow-cpu==2.8.1` que trae openWakeWord por defecto —
   ese combo está abandonado desde 2024 y no tiene wheels para macOS Apple
   Silicon ni Python 3.11+. Si no tienes instalado el extra `tflite`, te
   quedas con el `.onnx` (válido solo si tu instalación de Home Assistant
   corre con GPU/CUDA) y un aviso explicando cómo instalar el extra.

## Instalación

```bash
pip install -e ".[train]"    # entrenamiento (torch, speechbrain, audiomentations...)
pip install -e ".[tflite]"   # conversión final a .tflite (onnx2tf, tensorflow)
```

## Uso

```bash
wake-trainer train ok_jota          # modo rápido, ~200 MB de negativos
wake-trainer train ok_jota --full   # modo full, ~17 GB, más robusto
```

Los datasets de negativos se cachean en `data/negative_features/`
(compartido entre proyectos — no se vuelve a descargar al entrenar otra
wake word). Las features intermedias de cada entrenamiento quedan en
`models/_work/<modelo>/`.

## Problemas conocidos y cómo se resuelven aquí

| Síntoma | Causa | Cómo lo evita este proyecto |
|---|---|---|
| `AttributeError: module 'openwakeword.train' has no attribute 'train_custom_model'` | Esa función nunca existió | Se usa `Model` + `auto_train` directamente |
| `ModuleNotFoundError: onnx_tf` / `tensorflow-probability` | `openwakeword[full]` pin dependencias abandonadas de 2024 | Extra `tflite` propio con `onnx2tf` |
| `ModuleNotFoundError: No module named 'tf_keras'` / `'onnx_graphsurgeon'` / `'psutil'` uno detrás de otro, cada vez que se arregla el anterior (la conversión falla y se queda solo en `.onnx` aunque `pip install -e '.[tflite]'` ya se haya hecho) | Toda la serie 1.x de `onnx2tf` tiene el `METADATA` del wheel vacío (`Requires-Dist` no declara nada real), así que cualquier pin por rango (`>=1.20` o incluso `>=1.29`, que sí declara dependencias pero arrastra `ai_edge_litert` — ver fila siguiente) deja que pip instale una versión cuyo árbol de imports incondicionales no está verificado | Pin exacto `onnx2tf==1.26.9`, la única versión cuyos ~190 ficheros `.py` se recorrieron con un parser AST para listar TODO import incondicional de nivel de módulo — no solo el primero que revienta. El extra `tflite` declara el resultado completo: `tf-keras`, `onnx-graphsurgeon`, `psutil`, `absl-py`, `flatbuffers`, `requests`, `sng4onnx` |
| `ResolutionImpossible` instalando `onnx2tf>=1.29`: `ai-edge-litert` "no matching distributions available for your environment" | A partir de `onnx2tf` 1.27.0, el paquete importa sin condicional `ai_edge_litert`, que en PyPI solo publica wheels `manylinux` (Linux) — ni wheel ni sdist para macOS, de ninguna versión | Se usa `onnx2tf==1.26.9` en vez de una versión más reciente — el último patch de la última serie sin esa dependencia |
| `AttributeError: torchaudio has no attribute 'set_audio_backend'` | Eliminado en torchaudio ≥2.1, algunas libs aún lo llaman | Stub no-op automático en `trainer_core.py` |
| `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools 82.0.0 (feb-2026) eliminó `pkg_resources` por completo; speechbrain aún lo usa | Extra `train` fija `setuptools<82` |
| Entrenamiento sin datos negativos (se queda en `pending`) | Diseño original solo pasaba clips positivos | Pipeline completo de negativos (este documento) |
| `ValueError: Error! Clip does not have the correct sample rate!` | Piper escribe el WAV al sample rate nativo de cada voz, y no todas las voces comparten uno (p. ej. `es_ES-carlfm-x_low` a 16000 Hz frente a `es_ES-davefx-medium` a 22050 Hz); `openwakeword.data.augment_clips` exige 16000 Hz exacto | Toda síntesis (Piper/OpenAI/ElevenLabs) se normaliza a 16kHz mono en `trainer/synthesizer.py`, y `run_training` pasa además cualquier clip ya existente por `trainer/audio_utils.repair_clips` antes de extraer features, por si hay ficheros de una síntesis anterior a este fix |
| `El fichero de features 'general_negative (train)' tiene shape (N, 96), se esperaba (N, pasos, 96)` (o, sin la validación propia, un `IndexError: tuple index out of range` genérico dentro de `mmap_batch_generator`) | `validation_set_features.npy` y el ACAV100M de HuggingFace son un array 2D — una secuencia continua de frames de audio, sin cortar en clips/ventanas — pero `mmap_batch_generator` asume en su constructor que todo fichero que recibe ya es 3D `(N, pasos, 96)` | `trainer/negative_data.py::build_windowed_features` corta esos ficheros en ventanas fijas de 16 pasos una sola vez (cacheado en disco, procesado por bloques para no cargar en RAM los ~17 GB del modo `--full`) antes de dárselos a `mmap_batch_generator` |
