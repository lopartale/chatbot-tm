"""
Extracción de los datos de la intención (Avance 2)

El SVM (clasificador_svm.py) decide QUÉ pide el usuario; este módulo extrae los datos que
esa intención necesita, para que el gestor de respuestas no tenga que volver a interpretar
el texto:

    listar_por_dominio      el dominio nombrado ("servicios de seguros" -> seguros)
    detalle_servicio        el servicio: por ID ("SRV-012"), posición ("del segundo") o nombre
    analizar_requerimiento  las necesidades funcionales ("consultar saldo y transferir")

Son reglas regex sobre el texto normalizado; no deciden la intención.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from lenguaje.preprocesamiento import PATRON_ID_SERVICIO, MensajePreprocesado, normalizar, preprocesar

# Patrones de extracción de los datos de cada intención
TERMINOS_REQUERIMIENTO = r"pantalla|requerimiento|formulario|vista|modulo|historia de usuario|caso de uso"
PATRON_PEDIDO_DETALLE = r"\b(detalle|contrato|parametros|entrada|salida|como se (usa|consume))\b"
# "servicios de pagos", "servicios del dominio de seguros" -> captura el posible dominio
PATRON_DOMINIO = r"\bservicios (?:(?:para )?(?:el |del )?dominio de |del |de )(\w+)"

# "detalle del segundo" -> posición en la última lista de servicios recomendados
POSICION_POR_ORDINAL = {"primero": 0, "primer": 0, "1": 0, "segundo": 1, "2": 1, "tercero": 2, "tercer": 2, "3": 2,
                        "cuarto": 3, "quinto": 4, "sexto": 5, "septimo": 6, "octavo": 7, "noveno": 8, "decimo": 9}

# Preámbulo de un requerimiento ("pantalla para ...") y separadores entre sus necesidades
PATRON_PREAMBULO = rf"^.*?\b({TERMINOS_REQUERIMIENTO})\b\s*(de|para|que|donde|en (la|el) que|con)?\s*"
PATRON_SEPARADOR = r",|;|\n|\.\s| y | ademas | tambien | luego "


@dataclass
class Clasificacion:
    """Resultado del clasificador: la intención y los datos que el gestor de respuestas necesita."""
    intencion: str
    dominio: str | None = None                      # listar_por_dominio
    id_servicio: str | None = None                  # detalle_servicio: "SRV-007"
    posiciones: list[int] = field(default_factory=list)    # detalle_servicio: "del segundo"
    referencia: MensajePreprocesado | None = None   # detalle_servicio: el texto sin "detalle", "contrato"...
    necesidades: list[MensajePreprocesado] = field(default_factory=list)  # analizar_requerimiento


class ExtractorDatos:
    """Reglas que convierten la intención del SVM en una Clasificacion con sus datos."""

    def __init__(self, dominios: Iterable[str]):
        self.dominios = list(dominios)  # vocabulario de dominios válidos, no el inventario

    def extraer(self, intencion: str, texto: str) -> Clasificacion:
        """Clasificacion con la intención que decidió el SVM y los datos que necesita."""
        if intencion == "listar_por_dominio":
            # dominio=None si no se reconoce ninguno ("servicio para pagar el recibo"): el asistente
            # decide con el inventario si es una búsqueda o si muestra la lista de dominios
            return Clasificacion(intencion, dominio=self._dominio_mencionado(texto) or self._dominio_nombrado(texto))
        if intencion == "detalle_servicio":
            return self._detalle(texto)  # qué servicio: por ID, posición ("el segundo") o nombre
        if intencion == "analizar_requerimiento":
            # "pantalla para consultar saldo y transferir" -> [consultar saldo, transferir]
            return Clasificacion(intencion, necesidades=_separar_necesidades(texto))
        return Clasificacion(intencion)  # el resto de las intenciones no necesita datos extra

    def _dominio_nombrado(self, texto: str) -> str | None:
        """Primer dominio del inventario nombrado en cualquier parte del texto ("¿qué hay de seguros?")."""
        palabras = [palabra.rstrip("s") for palabra in re.findall(r"\w+", texto)]  # "seguros" -> "seguro"
        for dominio in self.dominios:
            if normalizar(dominio).rstrip("s") in palabras:
                return dominio
        return None

    @staticmethod
    def _detalle(texto: str) -> Clasificacion:
        """Cómo se refiere el usuario al servicio: ID, posición o nombre."""
        if id_mencionado := re.search(PATRON_ID_SERVICIO, texto):  # "srv 7" -> "SRV-007"
            return Clasificacion("detalle_servicio", id_servicio=f"SRV-{int(id_mencionado.group(1)):03d}")
        # "detalle del segundo" -> posición 1 de la última lista de recomendados
        posiciones = [posicion for ordinal, posicion in POSICION_POR_ORDINAL.items()
                      if re.search(rf"\b{ordinal}\b", texto)]
        # "contrato del servicio de cancelación de seguros" -> "del servicio de cancelación de seguros"
        referencia = preprocesar(re.sub(PATRON_PEDIDO_DETALLE, " ", texto))
        return Clasificacion("detalle_servicio", posiciones=posiciones, referencia=referencia)

    def _dominio_mencionado(self, texto: str) -> str | None:
        """Dominio nombrado tras "servicios de ...", en plural o singular."""
        mencion = re.search(PATRON_DOMINIO, texto)
        if not mencion:
            return None
        nombrado = mencion.group(1).rstrip("s")  # plural o singular: "pagos" y "pago" valen igual
        for dominio in self.dominios:
            if normalizar(dominio).rstrip("s") == nombrado:
                return dominio
        return None


def _separar_necesidades(texto: str) -> list[MensajePreprocesado]:
    """Divide un requerimiento en necesidades funcionales (una por frase)."""
    texto = re.sub(PATRON_PREAMBULO, "", texto)  # quita "pantalla para ..." del inicio
    # Una necesidad por fragmento separado por comas, "y", "además"...; se descartan los vacíos
    fragmentos = (preprocesar(fragmento.strip(" .:")) for fragmento in re.split(PATRON_SEPARADOR, texto))
    return [fragmento for fragmento in fragmentos if fragmento.palabras]
