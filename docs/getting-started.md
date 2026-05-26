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
pip install openWakeWord[train]
```

En macOS Apple Silicon, PyTorch incluye soporte MPS (Metal Performance Shaders) automáticamente — no se necesita configuración adicional.

## 4. Descargar Piper TTS

Piper genera las muestras sintéticas de voz. Descarga el binario para tu plataforma desde [github.com/rhasspy/piper/releases](https://github.com/rhasspy/piper/releases) y colócalo en `piper/`:

```bash
mkdir -p piper
# Ejemplo para macOS ARM64:
# curl -L <url-release-piper-macos-aarch64> | tar xz -C piper/
```

Después descarga los modelos de voz (ver [recording-guide.md](recording-guide.md#voces-piper-recomendadas)).

## 5. Verificar la instalación

```bash
python3 -c "import openwakeword; print('openWakeWord OK')"
./piper/piper --version
```

Si ambos comandos responden sin errores, el entorno está listo.

## Siguiente paso

→ [recording-guide.md](recording-guide.md) — cómo grabar las muestras reales de voz
