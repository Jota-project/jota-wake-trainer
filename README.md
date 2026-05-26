# jota-wake-trainer

Herramienta para entrenar wake words personalizadas compatibles con [openWakeWord](https://github.com/dscripka/openWakeWord) y el ecosistema [wyoming](https://github.com/rhasspy/wyoming) de Home Assistant.

Forma parte del ecosistema **J** — una plataforma de asistente de voz local, privado y personalizable.

---

## ¿Qué incluye este repositorio?

- **Pipeline de entrenamiento** — scripts para grabar muestras reales, generar datos sintéticos con Piper TTS y entrenar el modelo en macOS (Apple Silicon) o cualquier máquina con GPU.
- **Guías de integración** — cómo usar el modelo resultante con wyoming-openwakeword y jota-voice.
- **Modelo `ok_jota.tflite`** — se generará aquí una vez completado el entrenamiento y quedará versionado en `models/` para uso directo.

---

## Uso rápido — modelo preentrenado

> **Estado actual:** el modelo `ok_jota.tflite` está en proceso de entrenamiento. Cuando esté disponible, podrás usarlo directamente sin necesidad de entrenar nada.

Una vez disponible, para usar la wake word **"ok jota"**:

```bash
# Copia el modelo al directorio de openWakeWord
cp models/ok_jota.tflite ~/oww-venv/lib/python3.x/site-packages/wyoming_openwakeword/models/

# Lanza wyoming-openwakeword apuntando al modelo nuevo
python3 -m wyoming_openwakeword \
  --uri tcp://0.0.0.0:10401 \
  --preload-model ok_jota \
  --threshold 0.3
```

Ver [docs/integracion-jota-voice.md](docs/integracion-jota-voice.md) para la integración completa con jota-voice.

---

## Entrenar tu propia wake word

Si quieres entrenar una wake word diferente (o mejorar el modelo "ok jota" con más muestras):

1. **Instala las dependencias** → [docs/getting-started.md](docs/getting-started.md)
2. **Graba tus muestras de voz** → [docs/recording-guide.md](docs/recording-guide.md)
3. **Genera muestras sintéticas** → `scripts/generar_sinteticos.sh`
4. **Entrena el modelo** → `scripts/entrenar.sh`

El proceso completo tarda ~1-2 horas en un MacBook Air M2 (incluyendo grabación, síntesis y entrenamiento).

---

## Requisitos

- macOS con Apple Silicon (M1/M2/M3) — o Linux con CUDA
- Python 3.11+
- ~5-8 GB de espacio libre (datos de entrenamiento + modelos Piper)

---

## Arquitectura del modelo

openWakeWord funciona en dos capas:

```
Audio del micrófono
        ↓
Feature extractor (AudioSet embedding, ~30 MB)   ← compartido, ya cargado
        ↓
Clasificador binario (ok_jota.tflite, ~5 KB)     ← este es el archivo que entrenamos
        ↓
Puntuación 0-1  →  threshold 0.3  →  ¡wake word detectada!
```

El modelo que corre en el dispositivo final es solo el clasificador binario — diminuto, sin latencia adicional respecto a modelos preexistentes como `ok_nabu`.

---

## Relación con jota-voice

[jota-voice](https://github.com/alfonsogarre/jota-voice) es el repositorio principal del ecosistema J: configura el satélite Wyoming en Android, el pipeline de Home Assistant y todos los servicios asociados. Este repositorio es la **fuente del modelo de wake word** que jota-voice consume.

---

## Roadmap

- [x] Modelo preentrenado `ok_jota.tflite`
- [ ] Scripts de grabación y síntesis
- [ ] Pipeline de entrenamiento automatizado
- [ ] Guía de integración con jota-voice
- [ ] Speaker identification (identificación de locutor para control de acceso)

---

## Licencia

MIT
