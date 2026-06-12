# Referencia de comandos — wake-trainer

## Índice

- [wake-trainer](#wake-trainer-wizard)
- [status](#status)
- [record](#record)
- [import](#import)
- [synthesize](#synthesize)
- [train](#train)
- [evaluate](#evaluate)
- [providers list](#providers-list)
- [providers add](#providers-add)
- [providers remove](#providers-remove)

---

## wake-trainer (wizard)

```
wake-trainer
```

Lanza el wizard interactivo completo. Si no hay proyectos, crea uno nuevo. Si ya existen, muestra la lista y permite continuar desde el punto en que se quedó.

El wizard es el modo recomendado para empezar: guía la configuración del proyecto, la grabación de voces, la síntesis TTS y el entrenamiento en un único flujo conversacional.

---

## status

```
wake-trainer status [MODELO]
```

Muestra el estado de uno o todos los proyectos.

| Argumento | Descripción |
|-----------|-------------|
| `MODELO` | Nombre del modelo. Si se omite, lista todos los proyectos. |

**Ejemplos:**

```bash
# Ver todos los proyectos
wake-trainer status

# Ver el estado de un proyecto concreto
wake-trainer status ok_jota
```

---

## record

```
wake-trainer record MODELO [--voice NOMBRE]
```

Lanza el flujo de grabación o importación de clips de voz para un proyecto existente.

| Argumento / Flag | Descripción |
|-----------------|-------------|
| `MODELO` | Nombre del modelo (requerido). |
| `--voice`, `-v` | Nombre de la voz. Si se omite, procesa todas las voces pendientes. |

**Ejemplos:**

```bash
# Grabar todas las voces pendientes
wake-trainer record ok_jota

# Grabar solo la voz de una persona concreta
wake-trainer record ok_jota --voice Alfonso
```

El comando muestra el menú de grabación para cada voz pendiente con tres opciones: grabar ahora con el micrófono, importar WAVs externos, o dejar para más tarde.

---

## import

```
wake-trainer import MODELO [--voice NOMBRE] [--dir RUTA]
```

Importa clips WAV externos para una voz. Útil cuando otra persona te envía sus grabaciones.

| Argumento / Flag | Descripción |
|-----------------|-------------|
| `MODELO` | Nombre del modelo (requerido). |
| `--voice`, `-v` | Nombre de la voz. |
| `--dir`, `-d` | Ruta a la carpeta que contiene los WAVs. Si se omite, lo pregunta interactivamente. |

Los clips deben ser WAV 16kHz mono 16-bit. Los ficheros que no cumplan el formato se ignoran y se informa del conteo.

**Ejemplo:**

```bash
wake-trainer import ok_jota --voice Maria --dir ~/Downloads/grabaciones_maria
```

---

## synthesize

```
wake-trainer synthesize MODELO
```

Genera muestras sintéticas de voz usando los providers TTS configurados para el proyecto. Si el proyecto no tiene sources TTS asignadas, lanza el asistente de configuración de síntesis.

| Argumento | Descripción |
|-----------|-------------|
| `MODELO` | Nombre del modelo (requerido). |

**Ejemplo:**

```bash
wake-trainer synthesize ok_jota
```

El número de clips generados depende de las voces configuradas, las velocidades y el número de variaciones por voz.

---

## train

```
wake-trainer train MODELO
```

Entrena el clasificador binario a partir de los datos disponibles (clips reales + sintéticos). Muestra un estimado del dataset antes de confirmar.

| Argumento | Descripción |
|-----------|-------------|
| `MODELO` | Nombre del modelo (requerido). |

El modelo entrenado se guarda en `models/<nombre>.tflite`. En un MacBook Air M2 con ~1.000 muestras el proceso tarda entre 20 y 40 minutos.

**Ejemplo:**

```bash
wake-trainer train ok_jota
```

---

## evaluate

```
wake-trainer evaluate MODELO
```

Evalúa el modelo entrenado e informa de precisión, recall, falsos positivos y threshold recomendado.

| Argumento | Descripción |
|-----------|-------------|
| `MODELO` | Nombre del modelo (requerido). |

**Ejemplo:**

```bash
wake-trainer evaluate ok_jota
```

Salida de ejemplo:

```
  Modelo:            models/ok_jota.tflite
  Precisión:         94.2%
  Recall:            91.7%
  Falsos positivos:  1 en 10 s de silencio
  Threshold:         0.3
```

---

## providers list

```
wake-trainer providers list
```

Lista todos los providers TTS configurados globalmente con sus parámetros: nombre, tipo, URL o directorio, token, voces y velocidades.

---

## providers add

```
wake-trainer providers add [--name NOMBRE] [--type TIPO] [--url URL]
                           [--token-env VAR] [--voice VOZ] [--speed VEL]
```

Añade o actualiza un provider TTS global. Si se omiten `--name` o `--type`, lanza el wizard interactivo de configuración.

| Flag | Descripción |
|------|-------------|
| `--name`, `-n` | Nombre del provider (ej: `elevenlabs`). |
| `--type`, `-t` | Tipo: `openai` o `piper`. |
| `--url` | URL del endpoint (solo tipo `openai`). |
| `--token-env` | Variable de entorno que contiene el token de autenticación. |
| `--voice` | Nombre de una voz (repetible para añadir varias). |
| `--speed` | Velocidad de síntesis (repetible, ej: `0.8 1.0 1.2`). |

**Ejemplos:**

```bash
# Wizard interactivo (recomendado para la primera vez)
wake-trainer providers add

# Añadir un provider programáticamente
wake-trainer providers add \
  --name elevenlabs \
  --type openai \
  --url https://api.elevenlabs.io/v1 \
  --token-env ELEVENLABS_API_KEY \
  --voice Rachel \
  --voice Bella \
  --speed 0.9 --speed 1.0 --speed 1.1
```

Los providers se guardan en `configs/providers.local.json` (ignorado por git).

---

## providers remove

```
wake-trainer providers remove NOMBRE
```

Elimina un provider TTS global.

| Argumento | Descripción |
|-----------|-------------|
| `NOMBRE` | Nombre del provider a eliminar (requerido). |

**Ejemplo:**

```bash
wake-trainer providers remove elevenlabs
```
