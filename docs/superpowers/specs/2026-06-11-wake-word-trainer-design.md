# JWake Trainer — Diseño del sistema

**Fecha:** 2026-06-11  
**Estado:** Aprobado

---

## Resumen

JWake Trainer es una herramienta CLI para entrenar modelos de wake words personalizadas compatibles con openWakeWord y el ecosistema Wyoming/Home Assistant. Es de propósito general (cualquier frase) y usa "ok jota" como caso de uso primario.

El objetivo principal es proporcionar una experiencia guiada y elegante que lleve al usuario desde cero hasta un modelo `.tflite` funcional, sin requerir conocimientos técnicos de machine learning.

---

## Modos de uso

### Modo wizard (por defecto)

```
wake-trainer
```

Arranca la interfaz guiada completa. Detecta el estado persistente y sitúa al usuario donde lo dejó. Si no hay proyectos, inicia la creación de uno nuevo.

### Subcomandos (modo experto)

```
wake-trainer new                     # crear nuevo proyecto
wake-trainer status [proyecto]       # ver estado de uno o todos los proyectos
wake-trainer record [proyecto]       # grabar una voz específica
wake-trainer import [proyecto]       # importar WAVs de una voz
wake-trainer synthesize [proyecto]   # generar muestras sintéticas
wake-trainer train [proyecto]        # entrenar el modelo
wake-trainer evaluate [proyecto]     # evaluar el modelo entrenado
```

Cada subcomando es internamente interactivo cuando lo requiere (countdown, selección de voces, etc.).

---

## UI y estilo visual

La interfaz usa **Rich** (Python) para paneles, tablas, barras de progreso, colores y spinners. El objetivo es una experiencia visualmente elegante y clara, no un CLI minimalista.

Principios de diseño:
- Cada pregunta va precedida de una explicación breve de **por qué** se hace
- El estado del proyecto siempre es visible antes de pedir acción
- Los errores y advertencias son descriptivos y accionables
- El progreso se muestra en tiempo real (grabación, síntesis, entrenamiento)

---

## Flujo del wizard

### Pantalla de inicio

```
JWake Trainer
──────────────────────────────────────────────
 1  ok_jota         en curso  — falta voz "María"
 2  hey_asistente   listo     — modelo disponible

 [n] Nuevo proyecto    [q] Salir
```

Si no hay proyectos existentes, salta directamente a la creación.

### Fase 1 — Planificación (nuevo proyecto)

El wizard recoge todos los parámetros antes de grabar nada y calcula el dataset completo.

**Parámetros recogidos:**
- Frase de la wake word
- Nombre del modelo (sugerido automáticamente desde la frase)
- Número de personas y nombre de cada una
- Fuentes TTS (Piper y/o endpoints OpenAI-compatible)
- Voces seleccionadas por fuente

**Explicaciones contextuales antes de cada pregunta clave:**

Antes de preguntar el número de personas:
> "Para que el modelo funcione bien con las personas que lo van a usar, necesitamos grabar su voz directamente. Cada persona graba 30 clips en distintas condiciones (distancia, ruido, velocidad). Cuantas más personas graben, mejor reconocerá el modelo sus voces específicas."

Antes de preguntar sobre síntesis TTS:
> "Además de las voces reales, generaremos muestras sintéticas con servicios de text-to-speech. Esto hace el modelo más robusto para personas que no hayan grabado su voz, cubriendo acentos, géneros y estilos de habla distintos."

**Resumen de planificación antes de crear el proyecto:**

```
─── Resumen del proyecto ────────────────────────────
Frase:    "ok jota"
Modelo:   ok_jota

Grabaciones reales:
  · Alfonso   → 30 clips  (10 condiciones)
  · Carlos    → 30 clips  (10 condiciones)
  · María     → 30 clips  (10 condiciones)
  Total:         90 clips base

Síntesis TTS:
  · Piper:       6 voces × 5 velocidades = 30 clips
  · ElevenLabs:  4 voces × 5 velocidades = 20 clips
  Total:         50 clips sintéticos base

Con augmentación ×10 (ruido, sala, volumen):
  · Reales aumentados:    900 muestras
  · Sintéticos aumentados: 500 muestras
  · Dataset total:       ~1.400 muestras  ✅

Tiempo estimado:
  · Grabación:      ~45 min (3 personas)
  · Síntesis:        ~3 min
  · Entrenamiento:  ~45 min en MacBook M2

¿Crear proyecto? [S/n]
```

Si el dataset estimado cae por debajo de 1.000 muestras post-augmentación (mínimo empírico recomendado por openWakeWord para clasificadores binarios robustos), el wizard avisa y sugiere añadir más voces sintéticas o más personas.

### Fase 2 — Modo de cada persona

Por cada persona definida, se pregunta individualmente:

```
─── Voz de "María" ─────────────────────────────────
¿Cómo quieres añadir sus grabaciones?

  [g] Grabar ahora en este dispositivo
  [i] Importar ficheros WAV que ella te envíe
  [d] Dejar para más tarde

```

### Fase 3 — Grabación guiada

```
─── Grabando voz de "Alfonso" ── ok_jota ───────────
Condición 3/10: ruido de TV o radio de fondo
  Pon una TV o radio a volumen moderado y colócate a 1-1.5 m.
  Clip 2/4

  Preparado...  3  2  1  ● GRABANDO
  Di claramente: "ok jota"
  ■ Guardado ✓  (0.91s · 16kHz · mono · -18 dBFS)

  [↵ continuar]  [r repetir]  [q pausar y salir]
```

Validación automática tras cada clip:
- Duración mínima (> 0.5s)
- Formato correcto (16kHz, mono, 16-bit)
- Nivel de señal no demasiado bajo (> -40 dBFS) ni saturado (< -1 dBFS)

Si falla la validación, el clip se descarta y se ofrece repetir con la causa específica del error.

**Lista de preparación previa:** Antes de iniciar cualquier sesión de grabación, el wizard muestra un resumen completo de todos los audios a recopilar de todas las personas, para que el usuario pueda organizarse:

```
─── Preparación para grabación ─────────────────────
Necesitas recopilar los siguientes audios:

  Alfonso (grabar ahora):
    30 clips en 10 condiciones distintas
    Tiempo estimado: ~15 min

  Carlos (importar ficheros):
    Pídele que te envíe 30 clips WAV (16kHz, mono)
    Puedes compartirle la guía: docs/recording-guide.md

  María (pendiente):
    Sin definir — puedes asignarla más adelante

¿Empezar con Alfonso ahora? [S/n]
```

### Fase 4 — Importación de ficheros externos

```
─── Importar voz de "Carlos" ── ok_jota ────────────
Arrastra la carpeta con los WAVs o escribe la ruta:
> ~/Downloads/carlos_audios/

Escaneando...
  ✅ 28 ficheros WAV válidos  (16kHz · mono · 16-bit)
  ⚠️   2 ficheros ignorados   (formato incorrecto: .m4a)

¿Importar los 28 clips? [S/n]
  ⚠️  Faltan 2 clips para los 30 recomendados.
     Puedes continuar igualmente o añadir más ficheros después.
```

### Fase 5 — Configuración y selección de voces TTS

Para cada fuente TTS configurada:

1. El sistema llama a `GET /v1/voices` en el endpoint
2. Si responde: muestra lista de voces con género e idioma para seleccionar
3. Si no responde: permite entrada manual de IDs de voz

```
─── Fuente TTS: ElevenLabs ─────────────────────────
Consultando voces disponibles...  ✓ 32 voces encontradas

Selecciona las voces a usar (espacio para marcar, ↵ para confirmar):

  [✓] F  Español (ES)   Sofía
  [✓] M  Español (ES)   Diego
  [ ] F  Español (MX)   Valentina
  [✓] M  Español (MX)   Andrés
  [✓] F  Inglés (US)    Rachel
  [ ] M  Inglés (US)    Josh
  ...

Voces seleccionadas: 4  (mínimo recomendado: 6)
⚠️  Poca variedad. ¿Añadir más voces? [s/N]
```

El sistema valida que haya variación suficiente de género y región antes de continuar.

### Fase 6 — Síntesis

```
─── Síntesis de muestras ────────────────────────────
Generando clips sintéticos...

  Piper — voz es_ES_female    [████████████░░] 12/15
  ElevenLabs — Sofía          [██████░░░░░░░░]  6/15

Total: 18/50 clips sintéticos
```

### Fase 7 — Entrenamiento

```
─── Entrenamiento ───────────────────────────────────
Dataset cargado: 1.400 muestras (900 reales + 500 sintéticas)

  Epoch  47/100  loss 0.038  [████████░░░░]  47%
  Tiempo restante: ~22 min
```

### Fase 8 — Evaluación

```
─── Evaluación del modelo ───────────────────────────
Modelo: models/ok_jota.tflite  (4.8 KB)

  Precisión:              94.2%
  Recall:                 91.7%
  Falsos positivos:       0  en 10 min de audio ambiente
  Threshold recomendado:  0.3

¿Desplegar en jota-voice? [s/N]
```

---

## Estado persistente

Cada proyecto guarda su estado en `projects/<modelo>/session.json`:

```json
{
  "wake_word": "ok jota",
  "model_name": "ok_jota",
  "created_at": "2026-06-11T10:00:00",
  "voices": [
    { "name": "Alfonso", "mode": "record", "clips": 30, "status": "done" },
    { "name": "Carlos",  "mode": "import", "clips": 28, "status": "done" },
    { "name": "María",   "mode": null,      "clips": 0,  "status": "pending" }
  ],
  "synthesis": {
    "status": "done",
    "clips": 50,
    "sources": [
      {
        "type": "piper",
        "binary": "piper/piper",
        "voices_dir": "piper/voices",
        "selected_voices": ["es_ES_female", "es_ES_male", "es_MX_male"]
      },
      {
        "type": "openai",
        "url": "https://api.elevenlabs.io/v1",
        "token_env": "ELEVENLABS_API_KEY",
        "selected_voices": ["Sofía", "Diego", "Andrés", "Rachel"]
      }
    ]
  },
  "training": { "status": "pending", "epochs_completed": 0 },
  "model_path": null
}
```

Los tokens de API nunca se guardan en el JSON — se leen de variables de entorno o se piden en el momento.

---

## Arquitectura del código

```
trainer/
├── cli.py              ← entry point Typer, define subcomandos
├── wizard.py           ← flujo guiado completo (modo sin argumentos)
├── state.py            ← lectura/escritura de session.json
├── recorder.py         ← captura de micrófono, validación, countdown
├── importer.py         ← importación y validación de WAVs externos
├── synthesizer.py      ← Piper (subprocess) + endpoints OpenAI-compatible (httpx)
├── trainer.py          ← llamada a openWakeWord API, augmentación, export TFLite
├── evaluator.py        ← métricas del modelo entrenado
└── ui/
    ├── panels.py       ← componentes Rich reutilizables
    └── prompts.py      ← prompts enriquecidos con explicaciones contextuales
```

### Stack de dependencias

| Librería | Uso |
|----------|-----|
| `rich` | UI: paneles, tablas, barras de progreso, colores |
| `typer` | Subcomandos CLI con autocompletado |
| `sounddevice` | Captura de audio desde micrófono |
| `soundfile` | Lectura/escritura WAV, validación de formato |
| `httpx` | Llamadas a endpoints TTS OpenAI-compatible |
| `openwakeword[train]` | Augmentación, entrenamiento, export TFLite |

---

## Fuentes TTS soportadas

### Piper (offline)
- Binario local, sin coste, sin internet
- Voces listadas desde `piper/voices/` (ficheros `.onnx`)
- Velocidades: [0.8, 0.9, 1.0, 1.1, 1.2]

### Endpoint OpenAI-compatible
Cualquier servicio con API compatible con OpenAI TTS:
- ElevenLabs
- OpenAI
- Servidor TTS local (Kokoro, AllTalk, etc.)

Requisitos del endpoint:
- `POST /v1/audio/speech` — generación de audio
- `GET /v1/voices` — listado de voces (opcional; si no existe, entrada manual)

---

## Flujo de datos del entrenamiento

```
data/positivos/<persona>/*.wav   ──┐
                                    ├─→ augmentación automática (openWakeWord)
data/sintetizados/*.wav          ──┘   · Room Impulse Responses (RIR)
                                        · Background noise injection
                                        · Volume variation
                                        · Factor ×10 por clip
                                            │
                                            ▼
                                 AudioSet embedding (fijo, ~30 MB)
                                            │
                                            ▼
                                 Clasificador binario
                                 (100 epochs, batch 32, lr 0.001)
                                            │
                                            ▼
                                 models/<nombre>.tflite  (~5 KB)
```

La augmentación la ejecuta openWakeWord internamente. El trainer solo necesita pasar los clips base y el factor de augmentación.

---

## Nota de migración

El repositorio tiene actualmente una estructura plana (`data/positivos/`, `data/sintetizados/`, `configs/ok_jota.yaml`) pensada para un único modelo. Con el nuevo diseño multi-proyecto, cada wake word vive en `projects/<model_name>/`. Durante la implementación se migrará la estructura existente de ok_jota a `projects/ok_jota/` manteniendo compatibilidad con el YAML de configuración actual.

---

## Fuera de alcance (v1)

- Speaker identification / diarización de locutor — roadmap futuro, se implementa en el pipeline de jota-voice (post-trigger), no en el trainer
- Interfaz web o gráfica
- Entrenamiento distribuido o en la nube
- Soporte para modelos distintos de openWakeWord
