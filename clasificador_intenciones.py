"""
2. Clasificador de intenciones

Decide qué pide el usuario a partir del mensaje preprocesado, sin consultar el inventario.
Aporta dos nodos del flujo de cada turno:

    ¿Charla?                     nltk.chat.util.Chat: saludo, ayuda, gracias, despedida
    Reglas de intención en orden imagen -> ID SRV -> brecha -> dominios -> requerimiento
                                 -> detalle -> búsqueda

Además extrae los datos que cada intención necesita (dominio, ID o posición del
servicio, necesidades de un requerimiento) para que el gestor de respuestas no
tenga que volver a interpretar el texto.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from nltk.chat.util import Chat

from preprocesamiento import MensajePreprocesado, normalizar, preprocesar

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
     ["Soy un asistente basado en reglas (versión 1) que relaciona requerimientos "
      "funcionales con el inventario de servicios REST de la organización."]],
]

TERMINOS_REQUERIMIENTO = r"pantalla|requerimiento|formulario|vista|modulo|historia de usuario|caso de uso"
PATRON_ID_SERVICIO = r"\bsrv-?(\d{1,3})\b"
PATRON_PEDIDO_DETALLE = r"\b(detalle|contrato|parametros|entrada|salida|como se (usa|consume))\b"
# "servicios de pagos", "servicios del dominio de seguros" -> captura el posible dominio
PATRON_DOMINIO = r"\bservicios (?:(?:para )?(?:el |del )?dominio de |del |de )(\w+)"

# Se evalúan en orden y gana la primera que coincide; las reglas más amplias van al final.
# Las frases deben ser específicas: "no existe" o "inventario" sueltos aparecen en búsquedas normales.
REGLAS_INTENCION = [
    ("analizar_imagen", r"\b(imagen|imagenes|captura|mockup|screenshot|figma|foto|png|jpg|boceto|wireframe)\b"),
    ("detalle_servicio", PATRON_ID_SERVICIO),
    ("reportar_brecha", r"\b(brecha|que hago si|si no (existe|hay)|no (existe|hay) (el |un |ese |ningun )?servicio"
                        r"|(crear|solicitar) un (nuevo )?servicio|nuevo servicio)\b"),
    ("listar_dominios", r"\b(dominios|categorias|catalogo de servicios|que servicios (hay|tienes|existen)"
                        r"|lista(r|do)? (de )?(todos )?los servicios|(ver|mostrar|muestrame) el inventario)\b"),
    ("listar_por_dominio", PATRON_DOMINIO),
    ("analizar_requerimiento", rf"\b({TERMINOS_REQUERIMIENTO})\b"),
    ("detalle_servicio", PATRON_PEDIDO_DETALLE),
    ("buscar_servicio", r"\b(servicio|api|endpoint|consultar|buscar|necesito|existe|hay)\b"),
]

# "detalle del segundo" -> posición en la última lista de servicios recomendados
POSICION_POR_ORDINAL = {"primero": 0, "primer": 0, "1": 0, "segundo": 1, "2": 1, "tercero": 2, "3": 2}

# Preámbulo de un requerimiento ("pantalla para ...") y separadores entre sus necesidades
PATRON_PREAMBULO = rf"^.*?\b({TERMINOS_REQUERIMIENTO})\b\s*(de|para|que|donde|en (la|el) que|con)?\s*"
PATRON_SEPARADOR = r",|;|\n|\.\s| y | ademas | tambien | luego "


@dataclass
class Clasificacion:
    intencion: str
    dominio: str | None = None                      # listar_por_dominio
    id_servicio: str | None = None                  # detalle_servicio: "SRV-007"
    posiciones: list[int] = field(default_factory=list)    # detalle_servicio: "del segundo"
    referencia: MensajePreprocesado | None = None   # detalle_servicio: el texto sin "detalle", "contrato"...
    necesidades: list[MensajePreprocesado] = field(default_factory=list)  # analizar_requerimiento


class ClasificadorIntenciones:
    def __init__(self, dominios: Iterable[str]):
        self.chat = Chat(REGLAS_CHARLA)
        self.dominios = list(dominios)  # vocabulario de dominios válidos, no el inventario

    def charla(self, mensaje: MensajePreprocesado) -> str | None:
        """¿Charla? Respuesta de nltk Chat si el mensaje es un saludo, ayuda, gracias o despedida."""
        return self.chat.respond(mensaje.texto)

    def aplicar_reglas(self, mensaje: MensajePreprocesado) -> Clasificacion | None:
        """Reglas de intención en orden; None si ninguna aplica."""
        texto = mensaje.texto
        for intencion, patron in REGLAS_INTENCION:
            if not re.search(patron, texto):
                continue
            if intencion == "listar_por_dominio":
                dominio = self._dominio_mencionado(texto)
                if dominio is None:
                    continue
                return Clasificacion(intencion, dominio=dominio)
            if intencion == "detalle_servicio":
                return self._detalle(texto)
            if intencion == "analizar_requerimiento":
                return Clasificacion(intencion, necesidades=_separar_necesidades(texto))
            return Clasificacion(intencion)
        return None

    @staticmethod
    def _detalle(texto: str) -> Clasificacion:
        """Cómo se refiere el usuario al servicio: ID, posición o nombre."""
        if m := re.search(PATRON_ID_SERVICIO, texto):
            return Clasificacion("detalle_servicio", id_servicio=f"SRV-{int(m.group(1)):03d}")
        posiciones = [posicion for ordinal, posicion in POSICION_POR_ORDINAL.items()
                      if re.search(rf"\b{ordinal}\b", texto)]
        # "contrato del servicio de cancelación de seguros" -> "del servicio de cancelación de seguros"
        referencia = preprocesar(re.sub(PATRON_PEDIDO_DETALLE, " ", texto))
        return Clasificacion("detalle_servicio", posiciones=posiciones, referencia=referencia)

    def _dominio_mencionado(self, texto: str) -> str | None:
        """Dominio nombrado tras "servicios de ...", en plural o singular."""
        m = re.search(PATRON_DOMINIO, texto)
        if not m:
            return None
        nombrado = m.group(1).rstrip("s")
        for dominio in self.dominios:
            if normalizar(dominio).rstrip("s") == nombrado:
                return dominio
        return None


def _separar_necesidades(texto: str) -> list[MensajePreprocesado]:
    """Divide un requerimiento en necesidades funcionales (una por frase)."""
    texto = re.sub(PATRON_PREAMBULO, "", texto)
    fragmentos = (preprocesar(f.strip(" .:")) for f in re.split(PATRON_SEPARADOR, texto))
    return [f for f in fragmentos if f.palabras]
