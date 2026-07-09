# Getting started — entorno de entrenamiento

## Requisitos previos

- macOS con Apple Silicon (M1/M2/M3) o Linux con CUDA
- Python 3.11 o superior
- ~5-8 GB de espacio libre en disco

## 1. Clonar el repositorio

```bash
git clone https://github.com/alfonsogarre/jota-wake-trainer.git
cd jota-wake-trainer
```

## 2. Crear el entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -e ".[train]"
```

> `pip install openWakeWord[train]` **no funciona** — openWakeWord no publica
> ningún extra llamado `train` (solo `full`, que además arrastra
> `tensorflow-cpu==2.8.1` y `onnx-tf==1.10.0`, sin wheels para macOS Apple
> Silicon ni Python 3.11+). El extra `train` de este proyecto instala las
> dependencias reales de entrenamiento sin ese lastre.

Nota sobre el entrenamiento en sí: openWakeWord entrena siempre en CPU
(`torch.cuda.is_available()` es `False` en Mac, así que no usa MPS), pero el
modelo es diminuto (una DNN de un par de capas sobre 16×96 features), así
que unos miles de pasos tardan minutos, no horas.

Para exportar el modelo final a `.tflite` (el formato que carga el addon
openWakeWord de Home Assistant en CPU) hace falta un segundo extra:

```bash
pip install -e ".[tflite]"
```

Sin este extra el entrenamiento sigue funcionando igual y deja un `.onnx`
válido — solo no se puede convertir a `.tflite` hasta instalarlo.

## 4. Instalar Piper TTS

Piper genera las muestras sintéticas de voz. El repositorio original
(`rhasspy/piper`, que distribuía binarios standalone) está **archivado** y
nunca publicó binarios para macOS — solo Linux (amd64/arm64/armv7). El
proyecto activo es [OHF-Voice/piper1-gpl](https://github.com/OHF-voice/piper1-gpl),
que se instala como paquete Python:

```bash
pip install -e '.[piper]'
```

No hace falta descargar ni colocar ningún binario manualmente — `wake-trainer`
invoca Piper como módulo (`python3 -m piper`) usando el mismo intérprete con
el que se instaló.

Después descarga los modelos de voz (ver [recording-guide.md](recording-guide.md#voces-piper-recomendadas)).

## 5. Verificar la instalación

```bash
python3 -c "import openwakeword; print('openWakeWord OK')"
python3 -m piper --help
```

Si ambos comandos responden sin errores, el entorno está listo.

## Siguiente paso

→ [recording-guide.md](recording-guide.md) — cómo grabar las muestras reales de voz
