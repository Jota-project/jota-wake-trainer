# TTS Providers — Preconfiguración global de fuentes de síntesis

**Fecha:** 2026-06-11
**Estado:** Aprobado

---

## Objetivo

Permitir preconfigurar providers TTS (ElevenLabs, Jspeaker u otros endpoints OpenAI-compatible o binarios Piper) una sola vez a nivel global, para reutilizarlos en cualquier proyecto sin tener que reintroducir URL, token ni voces cada vez. Las API keys se guardan solo como referencias a variables de entorno, nunca como valores en ficheros.

---

## Contexto

El sistema actual (`trainer/synthesizer.py`, `trainer/state.py`) ya soporta síntesis via Piper y endpoints OpenAI-compatible mediante `TtsSource` (configurada por proyecto en `session.json`). El problema es que cada vez que se crea un proyecto hay que volver a introducir URL, token y voces manualmente.

La solución añade una capa de **config global de providers** persistida en `configs/providers.local.json` (ignorado por git) que actúa como catálogo reutilizable.

---

## Modelo de datos

### `ProviderConfig` (nuevo, global)

Definición permanente de un provider TTS. Se guarda en `configs/providers.local.json`.

```python
@dataclass
class ProviderConfig:
    name: str                                      # identificador único: "elevenlabs", "jspeaker"
    type: Literal["piper", "openai"]
    # campos openai-compatible
    url: Optional[str] = None
    token_env: Optional[str] = None               # nombre de la var de entorno, nunca el valor
    voices: list[str] = field(default_factory=list)   # voces preconfiguradas manualmente
    speeds: list[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    # campos piper
    binary: Optional[str] = None
    voices_dir: Optional[str] = None
```

### `TtsSource` (existente, por proyecto)

Sin cambios estructurales. Representa la selección específica de voces y velocidades para un proyecto concreto. Se genera a partir de `ProviderConfig` cuando el usuario añade un provider a un proyecto.

### Relación `ProviderConfig → TtsSource`

Al añadir un provider global a un proyecto:
- Se copian `url`, `token_env`, `binary`, `voices_dir`.
- Si el provider tiene `voices` preconfiguradas (ej: Jspeaker con `["Kaia"]`) → se usan directamente sin preguntar.
- Si `voices` está vacío (ej: ElevenLabs) → se intenta `GET /voices` en el endpoint; si falla o el usuario prefiere, entrada manual.
- El usuario puede ajustar las velocidades por proyecto.

---

## Fichero de configuración

**Ruta:** `configs/providers.local.json`  
**Versionado:** No (añadir a `.gitignore`)  
**Formato:**

```json
{
  "providers": [
    {
      "name": "elevenlabs",
      "type": "openai",
      "url": "https://api.elevenlabs.io/v1",
      "token_env": "ELEVENLABS_API_KEY",
      "voices": [],
      "speeds": [0.8, 0.9, 1.0, 1.1, 1.2],
      "binary": null,
      "voices_dir": null
    },
    {
      "name": "jspeaker",
      "type": "openai",
      "url": "http://localhost:5500",
      "token_env": null,
      "voices": ["Kaia"],
      "speeds": [1.0],
      "binary": null,
      "voices_dir": null
    }
  ]
}
```

**Seguridad:** El fichero contiene solo referencias a variables de entorno (`token_env`), nunca valores de API keys. Se excluye de git mediante `.gitignore`.

---

## Nuevo módulo: `trainer/providers.py`

Responsabilidad única: persistencia y gestión CRUD de providers globales.

```python
PROVIDERS_FILE = Path("configs/providers.local.json")  # parcheable via monkeypatch

def load_providers() -> list[ProviderConfig]:
    """Devuelve lista vacía si el fichero no existe."""

def save_providers(providers: list[ProviderConfig]) -> None:
    """Crea el fichero (y configs/) si no existe."""

def add_or_update_provider(provider: ProviderConfig) -> None:
    """Añade si no existe; actualiza si el nombre ya está."""

def remove_provider(name: str) -> bool:
    """Elimina por nombre. Devuelve False si no existe."""

def get_provider(name: str) -> ProviderConfig | None:
    """Busca por nombre."""
```

---

## CLI — subcomando `providers`

Se añade un grupo `providers` a `trainer/cli.py`.

### `wake-trainer providers list`

Tabla Rich con columnas: nombre, tipo, URL, token configurado (sí/no), voces, velocidades.
Si no hay providers: mensaje orientativo con el comando para añadir.

### `wake-trainer providers add`

```
wake-trainer providers add [--name NAME] [--type {piper,openai}] [--url URL]
                           [--token-env VAR] [--voice VOZ] [--speed VEL]
```

- `--voice` y `--speed` son repetibles (múltiples valores).
- Si faltan flags obligatorios (name, type), lanza el wizard interactivo de providers.
- Si el nombre ya existe, pregunta si actualizar.
- Wizard interactivo — rama openai: URL → token env → intentar descubrir voces → selección/manual → velocidades.
- Wizard interactivo — rama piper: directorio de voces → escaneo `.onnx` → selección → velocidades.

### `wake-trainer providers remove <name>`

Elimina el provider con confirmación. Si no existe, mensaje de error y exit 1.

---

## Integración con el wizard principal

En `_wizard_configure_synthesis` (dentro de `trainer/wizard.py`):

1. **Sin providers configurados** (fichero inexistente o lista vacía): ofrecer configurar uno antes de continuar. Si el usuario declina, flujo manual existente sin cambios.
2. **Con providers configurados**: mostrar lista y preguntar cuáles añadir al proyecto (selección múltiple). Opción adicional "añadir nuevo provider ahora".
3. **Conversión al proyecto**: `ProviderConfig → TtsSource` con lógica de voces descrita en el modelo de datos.

El wizard principal no requiere providers para funcionar — la integración es una mejora progresiva.

---

## Testing

### `tests/test_providers.py` (TDD)

- `load_providers()` con fichero inexistente → `[]`
- `add_or_update_provider()` crea el fichero si no existe
- Añadir provider con nombre ya existente → actualiza, no duplica
- `remove_provider()` devuelve `True` si existía, `False` si no
- `get_provider()` devuelve el correcto o `None`
- Roundtrip: guardar y cargar conserva todos los campos (incluidos `None` y listas)

### `tests/test_providers_cli.py` (CliRunner de Typer)

- `providers list` sin fichero → mensaje orientativo
- `providers add` con todos los flags → provider creado, verificado con `load_providers()`
- `providers remove <name>` existente → eliminado
- `providers remove <name>` inexistente → exit 1

---

## Cambios en ficheros existentes

| Fichero | Cambio |
|---------|--------|
| `trainer/cli.py` | Añadir grupo `providers` con subcomandos `list`, `add`, `remove` |
| `trainer/wizard.py` | Modificar `_wizard_configure_synthesis` para integrar providers globales; añadir `_wizard_add_provider` |
| `.gitignore` | Añadir `configs/providers.local.json` |

---

## Ficheros nuevos

| Fichero | Contenido |
|---------|-----------|
| `trainer/providers.py` | Módulo CRUD de providers globales |
| `tests/test_providers.py` | Tests unitarios TDD |
| `tests/test_providers_cli.py` | Smoke tests CLI con CliRunner |

---

## Fuera de alcance

- Cifrado del fichero de providers (las keys van en vars de entorno, no en el fichero).
- Sincronización de providers entre máquinas.
- Soporte de providers de tipo distinto a `piper` y `openai` (extensible en el futuro).
- Validación de conectividad al añadir un provider (el wizard intenta descubrir voces pero no bloquea si falla).
