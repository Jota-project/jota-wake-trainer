# Guía de grabación de muestras

## Muestras reales de voz

Necesitas **30 clips por persona** diciendo "ok jota" en condiciones variadas.
Formato requerido: **WAV · 16kHz · mono · 16-bit**.

Deja ~1 segundo de silencio antes y después de cada "ok jota". No recortes el audio manualmente.

### Tabla de condiciones

| # | Condición | Descripción | Clips/persona |
|---|-----------|-------------|:---:|
| 1 | Distancia normal · silencio | 1-1.5 m del dispositivo, habitación en silencio | 5 |
| 2 | Distancia cercana · silencio | 30-50 cm del dispositivo, volumen normal | 3 |
| 3 | Distancia larga · voz alzada | 3-4 m del dispositivo, voz ligeramente más alta | 3 |
| 4 | Ruido TV/radio | TV o radio de fondo a volumen moderado | 4 |
| 5 | Ruido de conversación | Otra persona hablando en la misma habitación | 3 |
| 6 | Música de fondo | Música a volumen normal | 3 |
| 7 | Voz rápida | Dicho con prisa | 3 |
| 8 | Voz lenta | Pausado, sobrearticulado | 2 |
| 9 | Voz baja / susurro | Tono bajo, sin proyectar | 2 |
| 10 | Ángulo lateral | Hablando de lado al dispositivo (~45°) | 2 |
| | **Total por persona** | | **30** |

Guarda cada clip en `data/positivos/<nombre_persona>/`:

```
data/positivos/
├── persona1/
│   ├── 001_normal.wav
│   ├── 002_normal.wav
│   └── ...
├── persona2/
└── persona3/
```

## Muestras sintéticas

El pipeline de entrenamiento lee todos los WAVs de `data/sintetizados/` sin importar su origen.
Puedes combinar libremente cualquier fuente TTS: Piper, ElevenLabs, tu propio servidor local, etc.
El formato debe ser el mismo que las grabaciones reales: **WAV · 16kHz · mono · 16-bit**.

Este repositorio incluye `scripts/generar_sinteticos_piper.sh` como fuente por defecto (gratuita, offline).
Si tienes acceso a servicios adicionales, simplemente deposita los WAVs resultantes en `data/sintetizados/`
antes de lanzar el entrenamiento.

### Síntesis con Piper TTS

El script `scripts/generar_sinteticos_piper.sh` automatiza la síntesis.
Genera 1 clip por voz × por velocidad = 5 clips base por voz.

### Voces Piper recomendadas

Lista los modelos disponibles tras instalar Piper:

```bash
# Ver voces disponibles
ls piper/voices/
```

Selección objetivo: ~6 voces en español (distintos géneros y variantes regionales) + 3 en inglés para cubrir hablantes no nativos.

| Tipo | Variante | Género | Propósito |
|------|----------|--------|-----------|
| Voz A | Español España | F | Acento peninsular femenino |
| Voz B | Español España | M | Acento peninsular masculino |
| Voz C | Español España (calidad distinta) | F | Diversidad de timbre |
| Voz D | Español México | M | Acento latinoamericano |
| Voz E | Español Latinoamérica | F | Diversidad regional |
| Voz F | Español variante adicional | M | Cobertura extra |
| Voz G | Inglés americano | F | Hablantes no nativos |
| Voz H | Inglés americano | M | Hablantes no nativos |
| Voz I | Inglés británico | F | Diversidad adicional |

Para descargar un modelo de voz Piper:

```bash
# Ejemplo (sustituir por el ID del modelo real):
mkdir -p piper/voices
curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/<ruta-modelo>.onnx \
  -o piper/voices/<nombre>.onnx
curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/<ruta-modelo>.onnx.json \
  -o piper/voices/<nombre>.onnx.json
```

Ver catálogo completo: [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)

## Siguiente paso

→ `scripts/generar_sinteticos.sh` para generar los clips sintéticos
→ `scripts/entrenar.sh` para lanzar el entrenamiento
