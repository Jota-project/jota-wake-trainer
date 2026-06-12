# Providers TTS

Los providers TTS son fuentes de síntesis de voz que `wake-trainer` usa para generar muestras sintéticas. Se configuran una vez a nivel global y están disponibles para cualquier proyecto.

## Tipos de provider

### openai

Cualquier endpoint compatible con la API de audio de OpenAI. Incluye:

- **ElevenLabs** — alta calidad, voces naturales, requiere cuenta
- **jspeaker** — servidor local compatible con la API de OpenAI
- **OpenAI TTS** — voces de OpenAI (alloy, echo, fable, onyx, nova, shimmer)
- Cualquier otro servidor con endpoint `/audio/speech`

### piper

Síntesis local con [Piper TTS](https://github.com/rhasspy/piper). Sin red, sin coste, completamente privado. Requiere el binario `piper` y modelos de voz `.onnx`.

---

## Configurar un provider

La forma más sencilla es el wizard interactivo:

```bash
wake-trainer providers add
```

El wizard detecta automáticamente las voces disponibles en el endpoint y permite seleccionarlas.

### Ejemplo: ElevenLabs

```
Nombre del provider: elevenlabs
Tipo: openai
URL del endpoint: https://api.elevenlabs.io/v1
Variable de entorno del token: ELEVENLABS_API_KEY
```

El wizard consulta las voces disponibles en tu cuenta y te permite seleccionarlas.

### Ejemplo: Piper local

```
Nombre del provider: piper_local
Tipo: piper
Directorio de voces Piper: piper/voices
Ruta al binario piper: piper/piper
```

El wizard escanea los modelos `.onnx` en el directorio indicado.

### Añadir un provider sin wizard

```bash
wake-trainer providers add \
  --name mi_provider \
  --type openai \
  --url https://mi-servidor/v1 \
  --token-env MI_API_KEY \
  --voice voz1 --voice voz2 \
  --speed 0.9 --speed 1.0 --speed 1.1
```

---

## Modelos de voz recomendados para Piper

Para una wake word en español, estos modelos dan buenos resultados:

| Modelo | Idioma | Tamaño | Calidad |
|--------|--------|--------|---------|
| `es_ES-davefx-medium` | Español (España) | ~60 MB | media |
| `es_ES-sharvard-medium` | Español (España) | ~60 MB | media |
| `es_MX-ald-medium` | Español (México) | ~60 MB | media |

Descarga desde [rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples/) y coloca los archivos `.onnx` y `.onnx.json` en `piper/voices/`.

---

## Dónde se guardan los providers

Los providers configurados se almacenan en `configs/providers.local.json`. Este archivo está en `.gitignore` para que tokens y URLs privadas no se commiteen accidentalmente.

Estructura del fichero:

```json
[
  {
    "name": "elevenlabs",
    "type": "openai",
    "url": "https://api.elevenlabs.io/v1",
    "token_env": "ELEVENLABS_API_KEY",
    "voices": ["Rachel", "Bella"],
    "speeds": [0.9, 1.0, 1.1]
  },
  {
    "name": "piper_local",
    "type": "piper",
    "binary": "piper/piper",
    "voices_dir": "piper/voices",
    "voices": ["piper/voices/es_ES-davefx-medium.onnx"],
    "speeds": [0.8, 1.0, 1.2]
  }
]
```

---

## Velocidades de síntesis

Cada provider tiene una lista de velocidades. Para cada voz y cada velocidad se genera un clip por cada frase del dataset de entrenamiento, multiplicando la cantidad de muestras sintéticas disponibles.

Con 3 voces y 3 velocidades se generan 9× más muestras que con una sola voz a velocidad normal. Para un dataset equilibrado se recomiendan entre 500 y 2.000 muestras sintéticas.

---

## Ver los providers configurados

```bash
wake-trainer providers list
```

Muestra una tabla con todos los providers, sus voces, tipo de autenticación y velocidades.

---

## Eliminar un provider

```bash
wake-trainer providers remove nombre_del_provider
```
