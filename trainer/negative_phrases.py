# trainer/negative_phrases.py
"""
Frases negativas/adversarias para sintetizar clips que NO son la wake word.

openWakeWord necesita datos negativos "duros" (parecidos a la wake word, pero
distintos) además de negativos genéricos. La función generate_adversarial_texts
de openwakeword solo funciona con fonemas en inglés (CMUdict), así que para una
wake word en español construimos la lista a mano: variaciones fonéticas cercanas
más un conjunto de frases genéricas de uso habitual con un asistente de voz.
"""
from __future__ import annotations

# Frases genéricas en español que conviene que el modelo aprenda a ignorar:
# conversación normal, otras wake words conocidas, comandos típicos.
GENERIC_NEGATIVES = [
    "hola", "buenas", "oye", "vale", "perdona", "una pregunta",
    "qué tal", "gracias", "de nada", "hasta luego",
    "ok google", "hey siri", "alexa", "ok nabu", "hey jarvis",
    "enciende la luz", "apaga la luz", "qué hora es", "pon música",
    "sube el volumen", "baja el volumen", "para la música",
    "qué tiempo hace", "pon una alarma", "llama a mamá",
    "la cocina está sucia", "voy a salir un momento", "no encuentro las llaves",
    "el partido empieza a las ocho", "me duele la cabeza", "qué película vemos",
    # Conversación cotidiana / saludos
    "buenos días", "buenas tardes", "buenas noches", "cómo estás", "qué haces",
    "nos vemos luego", "hasta mañana", "adiós", "un segundo", "espera un momento",
    "no te preocupes", "vale perfecto", "genial gracias", "de acuerdo", "claro que sí",
    "cómo va todo", "qué dices", "en serio", "no me digas", "qué raro",
    # Otras wake words / asistentes (para que no confunda ninguna con la suya)
    "ok siri", "hola google", "oye google", "hola cortana", "hey cortana",
    "hola alexa", "oye alexa", "computadora", "ordenador escucha",
    # Domótica / comandos típicos de casa inteligente
    "sube la persiana", "baja la persiana", "abre la puerta", "cierra la puerta",
    "enciende la tele", "apaga la tele", "pon la radio", "para el temporizador",
    "pon un temporizador", "qué día es hoy", "qué fecha es hoy", "recuérdame algo",
    "añade a la lista", "pon una nota", "manda un mensaje", "llama a papá",
    "llama a mi hermano", "dónde están mis llaves", "se me ha olvidado algo",
    "sube la calefacción", "baja la calefacción", "cierra la ventana", "abre la ventana",
    "pon las noticias", "cambia de canal", "sube el brillo", "baja el brillo",
    "conecta el wifi", "no tengo cobertura", "se ha ido la luz", "revisa el correo",
    # Vida diaria / frases sueltas de casa
    "tengo hambre", "tengo sueño", "estoy cansado", "qué aburrimiento",
    "vamos al cine", "vamos a cenar", "prepara la cena", "friega los platos",
    "saca la basura", "pon la lavadora", "tiende la ropa", "riega las plantas",
    "hace mucho frío", "hace mucho calor", "está lloviendo", "va a nevar",
    "qué hambre tengo", "voy a la ducha", "me voy a dormir", "ya he llegado",
]


def _near_miss_variants(wake_word: str) -> list[str]:
    """
    Genera variaciones fonéticamente cercanas a la wake word sustituyendo
    palabras clave por otras que suenan parecido en español. Pensado para
    frases cortas tipo "ok jota", pero funciona igual con cualquier frase
    de 2-3 palabras.
    """
    words = wake_word.lower().split()
    variants: list[str] = []

    # Sustituciones fonéticas genéricas para la primera palabra (si es "ok"/"oye"/"hey")
    first_word_subs = {
        "ok": ["oh que", "hoy", "co", "og", "eh", "ah que", "bueno", "oki"],
        "oye": ["ok", "hoy", "eh"],
        "hey": ["ok", "ey"],
    }
    # Sustituciones fonéticas genéricas para nombres/palabras cortas típicas
    generic_subs = [
        "rosa", "nota", "bota", "marta", "jose", "cocina", "gaviota",
        "pelota", "rota", "chota", "mota", "cuota", "derrota", "remota",
        "picota", "carlota", "dakota", "azota", "brota",
    ]

    if words:
        head, *rest = words
        for sub in first_word_subs.get(head, []):
            variants.append(" ".join([sub, *rest]))

        if rest:
            tail = rest[-1]
            for sub in generic_subs:
                if sub != tail:
                    variants.append(" ".join([head, *rest[:-1], sub]))

        # frase completa con orden invertido / partida (a veces suena parecido)
        if len(words) > 1:
            variants.append(" ".join(reversed(words)))
            variants.append(words[0])
            variants.append(" ".join(words[1:]))

    # limpia duplicados manteniendo orden
    seen = set()
    unique = []
    for v in variants:
        v = v.strip()
        if v and v not in seen and v != wake_word.lower():
            seen.add(v)
            unique.append(v)
    return unique


def build_negative_phrases(wake_word: str, n: int = 40) -> list[str]:
    """
    Devuelve una lista de hasta `n` frases negativas para sintetizar:
    mitad variaciones cercanas a la wake word, mitad frases genéricas.
    """
    near_miss = _near_miss_variants(wake_word)
    half = max(1, n // 2)

    phrases: list[str] = []
    phrases.extend(near_miss[:half])
    phrases.extend(GENERIC_NEGATIVES[: n - len(phrases)])

    # si aún falta (wake word muy corta / pocas variantes), rellena con genéricas restantes
    i = 0
    while len(phrases) < n and i < len(GENERIC_NEGATIVES):
        if GENERIC_NEGATIVES[i] not in phrases:
            phrases.append(GENERIC_NEGATIVES[i])
        i += 1

    return phrases[:n]
