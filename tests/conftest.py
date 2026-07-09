# tests/conftest.py
"""
tests/test_recorder.py y tests/test_workflows_recording.py importan
trainer.recorder, que hace `import sounddevice as sd` a nivel de módulo.
sounddevice es un paquete puro Python, pero en Linux necesita la librería
nativa PortAudio instalada en el sistema (`libportaudio2` en Debian/Ubuntu).
Si no está presente, el propio `import sounddevice` lanza
`OSError: PortAudio library not found` (si el paquete de Python sí está
instalado) — no un ImportError, así que `pytest.importorskip` no lo
detecta por sí solo — y si sounddevice ni siquiera está instalado como
paquete, es un ModuleNotFoundError normal. En cualquiera de los dos casos,
sin este guard pytest aborta TODA la recolección ("N errors during
collection", exit code 2), tumbando la suite entera aunque el resto de
tests no tenga nada que ver con audio.

El CI (.github/workflows/tests.yml) instala libportaudio2 explícitamente
para que estos dos ficheros se ejecuten de verdad ahí, con cobertura real.
Este conftest solo cubre el caso de un entorno local sin esa librería (por
ejemplo, un sandbox de desarrollo o una instalación nueva sin las
dependencias del sistema): en vez de tirar abajo toda la suite con un
error, se degrada a ignorar esos dos ficheros con un aviso claro.
"""
collect_ignore = []

try:
    import sounddevice  # noqa: F401
except (OSError, ImportError):
    print(
        "\n[tests/conftest.py] PortAudio no está instalado en este sistema "
        "(falta libportaudio2) — se omiten test_recorder.py y "
        "test_workflows_recording.py. En CI sí se ejecutan (ver "
        ".github/workflows/tests.yml)."
    )
    collect_ignore = ["test_recorder.py", "test_workflows_recording.py"]
