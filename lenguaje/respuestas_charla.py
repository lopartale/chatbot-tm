"""
Respuestas de charla

Cuando el clasificador de intenciones decide que el mensaje es charla (saludo, despedida,
agradecimiento, ayuda o una pregunta sobre el asistente), nltk.chat.util.Chat redacta la
respuesta en dos intentos:

    1. Patrón exacto: el mensaje completo es "gracias", "¿quién eres?", "hola"...
    2. Palabra clave: el mensaje contiene una ("hola, ¿me ayudas?", "gracias, muy amable");
       si no contiene ninguna, se muestra la ayuda.

Este módulo no decide la intención, sólo redacta la respuesta.
"""
from __future__ import annotations

from nltk.chat.util import Chat

from lenguaje.preprocesamiento import MensajePreprocesado

# Charla básica: sólo si el mensaje completo es un saludo, despedida, etc.
# Una sola respuesta por regla: Chat elige al azar entre varias y las pruebas dejarían de ser reproducibles.
REGLAS_CHARLA = [
    [r"^(hola|buen[oa]s? (dias|tardes|noches)|que tal|hey|saludos)[\s!.,]*$",
     ["¡Hola! Soy el asistente de servicios REST. Descríbeme la funcionalidad o "
      "la pantalla que necesitas y te digo qué servicios del inventario la cubren."]],
    [r"^(adios|hasta luego|bye|nos vemos|salir|terminar)[\s!.,]*$",
     ["¡Hasta luego! Recuerda reutilizar antes de construir."]],
    # Sólo si no sigue una petición: "Perfecto, ahora necesito..." no es charla
    [r"^((muchas )?gracias( por (todo|tu ayuda|la ayuda))?|excelente|perfecto|listo)[\s!.,]*$",
     ["Con gusto. ¿Quieres analizar otro requerimiento?"]],
    [r"^(ayuda|help|que puedes hacer|como funcionas|que haces|menu)[\s?!.]*$",
     ["Puedo ayudarte a:\n"
      "  1) Buscar un servicio por funcionalidad (ej. 'servicio para consultar saldo').\n"
      "  2) Analizar un requerimiento o pantalla con varias funciones y mostrar la cobertura.\n"
      "  3) Ver el detalle o contrato de un servicio (ej. 'detalle de SRV-007').\n"
      "  4) Listar los dominios y servicios del inventario.\n"
      "  5) Orientarte cuando exista una brecha (funcionalidad sin servicio)."]],
    [r"^(quien eres|como te llamas|que eres)[\s?!.]*$",
     ["Soy un asistente (versión 2) que identifica tu intención con un clasificador entrenado "
      "y relaciona requerimientos funcionales con el inventario de servicios REST de la organización."]],
]

# Cuando el SVM decide "charla" pero el mensaje no coincide completo con
# REGLAS_CHARLA ("hola, ¿me ayudas?", "gracias, muy amable"), la respuesta se elige por la
# palabra clave que contenga, en este orden; si no tiene ninguna, se muestra la ayuda.
RESPUESTA_SALUDO, RESPUESTA_DESPEDIDA, RESPUESTA_GRACIAS, RESPUESTA_AYUDA, RESPUESTA_IDENTIDAD = (
    respuestas[0] for _, respuestas in REGLAS_CHARLA)
REGLAS_CHARLA_POR_PALABRA_CLAVE = [
    [r".*\b(ayud\w*|puedes|opciones|usar|uso|funcion\w*|apoyar|menu)\b.*", [RESPUESTA_AYUDA]],
    [r".*\b(graci\w*|amable|excelente|perfecto)\b.*", [RESPUESTA_GRACIAS]],
    [r".*\b(adios|bye|chao|hasta|nos vemos|leemos)\b.*", [RESPUESTA_DESPEDIDA]],
    [r".*\b(quien|eres|llamas|humano|bot)\b.*", [RESPUESTA_IDENTIDAD]],
    [r".*\b(hola|ola|buen[oa]s?|tal|saludos|hey)\b.*", [RESPUESTA_SALUDO]],
    [r".*", [RESPUESTA_AYUDA]],
]


class RespuestasCharla:
    """Redacta la respuesta de un mensaje de charla con nltk.chat.util.Chat."""

    def __init__(self):
        self.chat = Chat(REGLAS_CHARLA)                                      # 1. patrón exacto
        self.chat_por_palabra_clave = Chat(REGLAS_CHARLA_POR_PALABRA_CLAVE)  # 2. palabra clave

    def responder(self, mensaje: MensajePreprocesado) -> str:
        """Respuesta por patrón exacto ("gracias", "¿quién eres?") y, si no coincide, por
        palabra clave ("hola, ¿me ayudas?"); la última regla garantiza siempre una respuesta."""
        return self.chat.respond(mensaje.texto) or self.chat_por_palabra_clave.respond(mensaje.texto)
