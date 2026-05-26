# Integración con jota-voice

## Requisitos previos

- [jota-voice](https://github.com/alfonsogarre/jota-voice) instalado y funcionando
- `wyoming_openwakeword` corriendo en el dispositivo satélite (ver jota-voice docs)

## Instalar el modelo ok_jota

### 1. Copiar el modelo al dispositivo

Desde el Mac, transfiere `models/ok_jota.tflite` al Huawei P8 Lite:

```bash
# Via SSH (Termux)
scp models/ok_jota.tflite \
  -P 8022 u0_a161@192.168.1.129:/data/data/com.termux/files/home/oww-venv/lib/python3.13/site-packages/wyoming_openwakeword/models/
```

### 2. Actualizar el comando de arranque de openwakeword

En el teléfono, editar el script de arranque para cambiar `ok_nabu` → `ok_jota`:

```bash
# En el teléfono (Termux):
nohup ~/oww-venv/bin/python3 -m wyoming_openwakeword \
  --uri tcp://0.0.0.0:10401 \
  --preload-model ok_jota \
  --threshold 0.3 \
  > ~/oww.log 2>&1 &
```

### 3. Actualizar start-satellite.sh

```bash
# En ~/start-satellite.sh, cambiar:
--wake-word-name ok_jota
```

### 4. Reiniciar los servicios

```bash
# Primero openwakeword (esperar ~15s al modelo TFLite)
# Luego el satélite:
nohup sh ~/start-satellite.sh </dev/null >/dev/null 2>&1 &
```

## Ajuste del threshold

El threshold por defecto es **0.3** (igual que `ok_nabu`). Si experimentas:

- **Muchos falsos positivos** → sube a 0.4-0.5
- **Detección poco fiable** → baja a 0.2-0.25

Ajusta `--threshold` en el comando de arranque de openwakeword.
