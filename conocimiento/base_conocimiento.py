"""
Base de conocimiento: inventario de servicios REST y su búsqueda

Carga el inventario desde el RAML (inventario_raml.py) y permite buscar en él los servicios
que cubren una consulta. Es el único módulo que conoce la estructura del inventario.

Búsqueda ponderada (Avance 1):

    consulta preprocesada -> raíces de cada palabra
                          -> puntaje por servicio: suma del peso TF-IDF de las raíces compartidas
                          -> filtros: PUNTAJE_MINIMO y COBERTURA_MINIMA de las palabras
                          -> bonos: el verbo coincide con el método HTTP; la consulta nombra el recurso
                          -> los MAX_RESULTADOS servicios con mayor puntaje
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from conocimiento.inventario_raml import cargar_inventario_raml
from lenguaje.preprocesamiento import MensajePreprocesado, extraer_raices

PESO_DESCRIPCION = 1.0     # raíz presente en el nombre, la descripción, el recurso o un valor de enum
PESO_PALABRA_CLAVE = 2.0   # raíz presente en el dominio o las palabras clave
PUNTAJE_MINIMO = 2.0       # suma mínima de pesos para considerar un servicio
COBERTURA_MINIMA = 0.4     # fracción mínima de palabras de la consulta presentes en el servicio
BONO_METODO = 1.0          # se suma si el verbo de la consulta corresponde al método HTTP
BONO_RECURSO = 2.0         # se suma si la consulta nombra el recurso del endpoint
MAX_RESULTADOS = 3

# Verbo de la consulta -> métodos HTTP que suelen implementarlo
PATRONES_DE_VERBO_POR_METODO = {
    "GET": r"\b(consultar|ver|saber|cuanto|cuantos|listar|lista|obtener|revisar|buscar|historial)\b",
    "POST": r"\b(agregar|registrar|crear|alta|pagar|mandar|enviar|cancelar|generar|solicitar|contratar|pedir)\b",
    "PATCH": r"\b(cambiar|actualizar|modificar|editar|bloquear|desbloquear)\b",
    "PUT": r"\b(cambiar|actualizar|modificar|editar)\b",
    "DELETE": r"\b(eliminar|borrar|quitar)\b",
}


class Coincidencia(NamedTuple):
    """Un servicio encontrado y su puntaje de relevancia."""
    puntaje: float
    servicio: dict


def metodos_sugeridos(texto: str) -> set[str]:
    """Métodos HTTP que corresponden a los verbos de un texto normalizado ("pagar" -> POST)."""
    return {metodo for metodo, patron in PATRONES_DE_VERBO_POR_METODO.items() if re.search(patron, texto)}


def raices_del_recurso(servicio: dict) -> set[str]:
    """Raíces del último segmento del endpoint que no es parámetro (/clientes/{idCliente}/cuentas
    -> 'cuentas'), en plural y singular: el stemmer da 'prest' para 'prestamos' y 'prestam' para 'prestamo'."""
    segmentos = [segmento for segmento in servicio["endpoint"].split("/") if segmento and not segmento.startswith("{")]
    plural = segmentos[-1].replace("-", " ")
    singular = " ".join(palabra.removesuffix("s") for palabra in plural.split())
    return set(extraer_raices(plural)) | set(extraer_raices(singular))


def pesos_del_servicio(servicio: dict) -> dict[str, float]:
    """Peso de cada raíz del servicio, antes de aplicar el IDF.

    En el RAML el nombre es un resumen de la descripción, por eso pesa igual que ella; el
    dominio y las palabras clave describen la operación y pesan el doble.
    """
    pesos = {}
    for texto in (servicio["nombre"], servicio["descripcion"], servicio.get("contexto", ""),
                  *servicio.get("valores_enum", [])):
        pesos.update(dict.fromkeys(extraer_raices(texto), PESO_DESCRIPCION))
    for texto in (servicio["dominio"], *servicio["palabras_clave"]):
        pesos.update(dict.fromkeys(extraer_raices(texto), PESO_PALABRA_CLAVE))
    return pesos


class InventarioServicios:
    """Base de conocimiento: inventario de servicios del RAML con su índice de búsqueda."""

    def __init__(self, ruta_raml: str | Path):
        inventario = cargar_inventario_raml(ruta_raml)
        self.titulo = inventario["titulo"]
        self.version = inventario["version"]
        self.base_uri = inventario["base_uri"]
        self.servicios: list[dict] = inventario["servicios"]
        # Accesos rápidos: por ID ("SRV-012") y por dominio ("pagos" -> sus servicios)
        self.servicio_por_id = {servicio["id"]: servicio for servicio in self.servicios}
        self.servicios_por_dominio: dict[str, list[dict]] = {}
        for servicio in self.servicios:
            self.servicios_por_dominio.setdefault(servicio["dominio"], []).append(servicio)
        # Índice de búsqueda: pesos TF-IDF por servicio y raíces del recurso de cada endpoint
        self._pesos_por_servicio = self._construir_indice()
        self._raices_recurso = {servicio["id"]: raices_del_recurso(servicio) for servicio in self.servicios}

    def _construir_indice(self) -> dict[str, dict[str, float]]:
        """Asigna a cada raíz de cada servicio un peso TF-IDF simplificado."""
        indice = {servicio["id"]: pesos_del_servicio(servicio) for servicio in self.servicios}
        # IDF: una raíz presente en muchos servicios distingue poco (en todos, peso 0)
        servicios_con_raiz = Counter(raiz for pesos in indice.values() for raiz in pesos)
        total = len(self.servicios)
        for pesos in indice.values():
            for raiz in pesos:
                pesos[raiz] *= math.log(total / servicios_con_raiz[raiz])
        return indice

    def buscar(self, consulta: MensajePreprocesado, limite: int = MAX_RESULTADOS) -> list[Coincidencia]:
        """Servicios que comparten suficientes raíces con la consulta, del más al menos relevante."""
        grupos_de_raices = consulta.raices  # un grupo por palabra: su raíz y la de su sinónimo
        if not grupos_de_raices:
            return []
        metodos = metodos_sugeridos(consulta.texto)
        coincidencias = []
        for id_servicio, pesos in self._pesos_por_servicio.items():
            # Cada palabra de la consulta aporta el mayor peso entre sus raíces alternativas
            aportes = [max(pesos.get(raiz, 0.0) for raiz in grupo)
                       for grupo in grupos_de_raices if grupo & pesos.keys()]
            puntaje = sum(aportes)
            cobertura = len(aportes) / len(grupos_de_raices)
            if puntaje < PUNTAJE_MINIMO or cobertura < COBERTURA_MINIMA:
                continue
            servicio = self.servicio_por_id[id_servicio]
            if servicio["metodo"] in metodos:                     # el verbo favorece al método HTTP
                puntaje += BONO_METODO
            if any(grupo & self._raices_recurso[id_servicio] for grupo in grupos_de_raices):
                puntaje += BONO_RECURSO                           # el sustantivo favorece al recurso
            coincidencias.append(Coincidencia(round(puntaje, 2), servicio))
        coincidencias.sort(key=lambda coincidencia: -coincidencia.puntaje)
        return coincidencias[:limite]

    def mejor_coincidencia(self, consulta: MensajePreprocesado) -> dict | None:
        """El servicio más relevante para la consulta, o None si ninguno califica."""
        coincidencias = self.buscar(consulta, limite=1)
        return coincidencias[0].servicio if coincidencias else None

    def tiene_servicio_para(self, consulta: MensajePreprocesado) -> bool:
        """¿Coincide con el inventario? Al menos un servicio califica para la consulta."""
        return bool(self.buscar(consulta, limite=1))
