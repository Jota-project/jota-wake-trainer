# jota-wake-trainer

Entrena tu propia wake word personalizada y despliégala en Home Assistant en menos de 2 horas.

Genera un clasificador `.tflite` compatible con [openWakeWord](https://github.com/dscripka/openWakeWord) y el ecosistema [wyoming](https://github.com/rhasspy/wyoming) a partir de grabaciones reales y muestras sintéticas de TTS. Parte del ecosistema **J** — asistente de voz local, privado y personalizable.

---

## Requisitos

- macOS Apple Silicon (M1/M2/M3) o Linux con CUDA
- Python 3.11 o superior
- ~5-8 GB de espacio libre (datos de entrenamiento + modelos Piper)

---

## Instalación

```bash
git clone https://github.com/SitoSt/jota-wake-trainer.git
cd jota-wake-trainer

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[train]"
```

Verifica que el entorno está listo:

```bash
python3 -c "import openwakeword; print('openWakeWord OK')"
wake-trainer --help
```

Para usar síntesis Piper local instala el extra correspondiente (no hace
falta descargar ningún binario — `rhasspy/piper` está archivado y nunca tuvo
releases para macOS; el paquete Python `piper-tts` es la vía actual):

```bash
pip install -e ".[piper]"
```

Ver [docs/providers.md](docs/providers.md) para configurar providers TTS.

---

## Quickstart — wizard interactivo

El modo más sencillo es el wizard, que guía todo el proceso paso a paso:

```bash
wake-trainer
```

El wizard te pedirá:

1. **Wake word** — la frase que quieres entrenar (ej: `"ok jota"`)
2. **Nombre del modelo** — identificador interno (ej: `ok_jota`)
3. **Personas que van a grabar** — cuántas y sus nombres
4. **Fuente TTS** — provider de síntesis para muestras sintéticas (Piper, ElevenLabs, etc.)

A continuación lanza el proceso de grabación para cada persona (30 clips en 10 condiciones), genera las muestras sintéticas y entrena el modelo.

El archivo resultante queda en `models/<nombre_modelo>.tflite`.

---

## Comandos

Además del wizard, cada etapa se puede ejecutar individualmente:

| Comando | Descripción |
|---------|-------------|
| `wake-trainer` | Lanza el wizard interactivo completo |
| `wake-trainer status [modelo]` | Estado de uno o todos los proyectos |
| `wake-trainer record <modelo>` | Graba muestras de voz para un proyecto |
| `wake-trainer import <modelo>` | Importa clips WAV externos |
| `wake-trainer synthesize <modelo>` | Genera muestras sintéticas con TTS |
| `wake-trainer train <modelo>` | Entrena el modelo |
| `wake-trainer evaluate <modelo>` | Evalúa el modelo entrenado |
| `wake-trainer providers list` | Lista los providers TTS configurados |
| `wake-trainer providers add` | Añade o actualiza un provider TTS |
| `wake-trainer providers remove <nombre>` | Elimina un provider TTS |

Ver [docs/cli-reference.md](docs/cli-reference.md) para la referencia completa con flags y ejemplos.

---

## Providers TTS

La síntesis de muestras sintéticas usa providers TTS configurables: Piper (local, sin red), ElevenLabs, jspeaker u cualquier endpoint compatible con la API de OpenAI. Los providers se configuran una vez y están disponibles para todos los proyectos.

```bash
wake-trainer providers add
```

Ver [docs/providers.md](docs/providers.md) para la guía completa de configuración.

---

## Cómo funciona

openWakeWord separa el problema en dos capas:

```
Audio del micrófono
        ↓
Feature extractor (AudioSet embedding, ~30 MB)   ← compartido, preinstalado
        ↓
Clasificador binario (<modelo>.tflite, ~5 KB)    ← este es el archivo que generamos
        ↓
Puntuación 0-1  →  threshold  →  wake word detectada
```

Lo que entrena esta herramienta es únicamente el clasificador binario — un fichero diminuto y sin latencia adicional.

Al ser un clasificador **binario**, necesita ejemplos de las dos clases, no solo de la wake word:

- **Positivos**: tus grabaciones + síntesis TTS de la propia frase.
- **Negativos**: se generan/descargan solos, no hay que grabarlos —
  variaciones fonéticas cercanas sintetizadas con TTS ("ok jose", "ok rosa"...)
  más un dataset general precalculado (voz, ruido, música). En modo `quick`
  (por defecto) se descargan ~200 MB; en modo `full` ~17 GB para un modelo
  más robusto frente a falsos positivos. Ver [docs/entrenamiento.md](docs/entrenamiento.md).

---

## Documentación

| Doc | Contenido |
|-----|-----------|
| [docs/getting-started.md](docs/getting-started.md) | Instalación detallada del entorno |
| [docs/recording-guide.md](docs/recording-guide.md) | Cómo grabar muestras de calidad |
| [docs/entrenamiento.md](docs/entrenamiento.md) | Cómo funciona el pipeline de entrenamiento (positivos, negativos, quick/full) |
| [docs/cli-reference.md](docs/cli-reference.md) | Referencia completa de comandos |
| [docs/providers.md](docs/providers.md) | Configuración de providers TTS |
| [docs/integracion-jota-voice.md](docs/integracion-jota-voice.md) | Despliegue en Home Assistant |

---

## Ecosistema J

[jota-voice](https://github.com/alfonsogarre/jota-voice) es el repositorio principal: configura el satélite Wyoming en Android, el pipeline de Home Assistant y todos los servicios asociados. `jota-wake-trainer` genera el modelo `.tflite` que jota-voice consume.

---

## Licencia

MIT
