"""
3. Gestor de respuestas

Consulta la base de conocimiento (inventario de servicios leído del RAML) y aporta
dos nodos del flujo de cada turno:

    ¿Coincide con el inventario?  búsqueda ponderada para mensajes sin palabra disparadora
    Manejador de la intención     buscar, analizar requerimiento, detalle, listar dominios,
                                  listar por dominio, brecha, imagen (aviso Avance 2)

Es el único componente que accede al inventario. Los manejadores no guardan estado:
devuelven qué servicios recordar y el asistente los guarda como contexto de la sesión.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from clasificador_intenciones import Clasificacion
from inventario_raml import cargar_inventario_raml
from preprocesamiento import MensajePreprocesado, extraer_raices

# --------------------------------------------------------------------------
# Búsqueda ponderada en el inventario
# --------------------------------------------------------------------------
PESO_DESCRIPCION = 1.0     # raíz presente en el nombre, la descripción, el recurso o un valor de enum
PESO_PALABRA_CLAVE = 2.0   # raíz presente en el dominio o las palabras clave
PUNTAJE_MINIMO = 2.0       # suma mínima de pesos para considerar un servicio
COBERTURA_MINIMA = 0.4     # fracción mínima de palabras de la consulta presentes en el servicio
BONO_METODO = 1.0          # se suma si el verbo de la consulta corresponde al método HTTP
BONO_RECURSO = 2.0         # se suma si la consulta nombra el recurso del endpoint
MAX_RESULTADOS = 3

# Verbo de la consulta -> métodos HTTP que suelen implementarlo
VERBOS_POR_METODO = {
    "GET": r"\b(consultar|ver|saber|cuanto|cuantos|listar|lista|obtener|revisar|buscar|historial)\b",
    "POST": r"\b(agregar|registrar|crear|alta|pagar|mandar|enviar|cancelar|generar|solicitar|contratar|pedir)\b",
    "PATCH": r"\b(cambiar|actualizar|modificar|editar|bloquear|desbloquear)\b",
    "PUT": r"\b(cambiar|actualizar|modificar|editar)\b",
    "DELETE": r"\b(eliminar|borrar|quitar)\b",
}


class Coincidencia(NamedTuple):
    puntaje: float
    servicio: dict


def metodos_sugeridos(texto: str) -> set[str]:
    """Métodos HTTP que corresponden a los verbos de un texto normalizado."""
    return {metodo for metodo, patron in VERBOS_POR_METODO.items() if re.search(patron, texto)}


def _raices_recurso(servicio: dict) -> set[str]:
    """Raíces del último segmento del endpoint que no es parámetro (/clientes/{idCliente}/cuentas
    -> 'cuentas'), en plural y singular: el stemmer da 'prest' para 'prestamos' y 'prestam' para 'prestamo'."""
    segmentos = [s for s in servicio["endpoint"].split("/") if s and not s.startswith("{")]
    plural = segmentos[-1].replace("-", " ")
    singular = " ".join(palabra.removesuffix("s") for palabra in plural.split())
    return set(extraer_raices(plural)) | set(extraer_raices(singular))


class InventarioServicios:
    """Base de conocimiento: inventario de servicios del RAML con su índice de búsqueda."""

    def __init__(self, ruta_raml: str | Path):
        inventario = cargar_inventario_raml(ruta_raml)
        self.titulo = inventario["titulo"]
        self.version = inventario["version"]
        self.base_uri = inventario["base_uri"]
        self.servicios: list[dict] = inventario["servicios"]
        self.servicio_por_id = {s["id"]: s for s in self.servicios}
        self.servicios_por_dominio: dict[str, list[dict]] = {}
        for s in self.servicios:
            self.servicios_por_dominio.setdefault(s["dominio"], []).append(s)
        self._pesos_por_servicio = self._construir_indice()
        self._raices_recurso = {s["id"]: _raices_recurso(s) for s in self.servicios}

    def _construir_indice(self) -> dict[str, dict[str, float]]:
        """Asigna a cada raíz de cada servicio un peso TF-IDF simplificado."""
        indice = {s["id"]: self._pesos_de(s) for s in self.servicios}
        # IDF: una raíz presente en muchos servicios distingue poco (en todos, peso 0)
        servicios_con_raiz = Counter(raiz for pesos in indice.values() for raiz in pesos)
        total = len(self.servicios)
        for pesos in indice.values():
            for raiz in pesos:
                pesos[raiz] *= math.log(total / servicios_con_raiz[raiz])
        return indice

    @staticmethod
    def _pesos_de(servicio: dict) -> dict[str, float]:
        # En el RAML el nombre es un resumen de la descripción, por eso pesa igual que ella
        pesos = {}
        for texto in (servicio["nombre"], servicio["descripcion"], servicio.get("contexto", ""),
                      *servicio.get("valores_enum", [])):
            pesos.update(dict.fromkeys(extraer_raices(texto), PESO_DESCRIPCION))
        for texto in (servicio["dominio"], *servicio["palabras_clave"]):
            pesos.update(dict.fromkeys(extraer_raices(texto), PESO_PALABRA_CLAVE))
        return pesos

    def buscar(self, consulta: MensajePreprocesado, limite: int = MAX_RESULTADOS) -> list[Coincidencia]:
        """Servicios que comparten suficientes raíces con la consulta, del más al menos relevante.

        Cada palabra de la consulta aporta el mayor peso entre sus raíces alternativas.
        Un servicio califica si la suma alcanza PUNTAJE_MINIMO y cubre al menos
        COBERTURA_MINIMA de las palabras; entre los que califican, el verbo favorece
        al método HTTP y el sustantivo al recurso del endpoint.
        """
        grupos = consulta.raices
        if not grupos:
            return []
        metodos = metodos_sugeridos(consulta.texto)
        coincidencias = []
        for sid, pesos in self._pesos_por_servicio.items():
            aportes = [max(pesos.get(raiz, 0.0) for raiz in grupo)
                       for grupo in grupos if grupo & pesos.keys()]
            puntaje = sum(aportes)
            if puntaje < PUNTAJE_MINIMO or len(aportes) / len(grupos) < COBERTURA_MINIMA:
                continue
            servicio = self.servicio_por_id[sid]
            if servicio["metodo"] in metodos:
                puntaje += BONO_METODO
            if any(grupo & self._raices_recurso[sid] for grupo in grupos):
                puntaje += BONO_RECURSO
            coincidencias.append(Coincidencia(round(puntaje, 2), servicio))
        coincidencias.sort(key=lambda c: -c.puntaje)
        return coincidencias[:limite]

    def mejor_coincidencia(self, consulta: MensajePreprocesado) -> dict | None:
        coincidencias = self.buscar(consulta, limite=1)
        return coincidencias[0].servicio if coincidencias else None


# --------------------------------------------------------------------------
# Manejador de la intención
# --------------------------------------------------------------------------
class Respuesta(NamedTuple):
    texto: str
    # Servicios que el paso "Respuesta + contexto" guarda para "detalle del primero";
    # None conserva el contexto anterior (p. ej. al mostrar un detalle)
    recomendados: list[str] | None = None


class GestorRespuestas:
    def __init__(self, inventario: InventarioServicios):
        self.inventario = inventario

    def coincide_con_inventario(self, mensaje: MensajePreprocesado) -> bool:
        """¿Coincide con el inventario? Para mensajes sin palabra disparadora."""
        return bool(self.inventario.buscar(mensaje, limite=1))

    def manejar(self, mensaje: MensajePreprocesado, clasificacion: Clasificacion,
                ultimos_recomendados: list[str]) -> Respuesta:
        """Manejador de la intención: consulta la base de conocimiento y arma la respuesta."""
        # Cada intención tiene un manejador _<intencion>
        manejador = getattr(self, f"_{clasificacion.intencion}")
        return manejador(mensaje, clasificacion, ultimos_recomendados)

    def _buscar_servicio(self, mensaje: MensajePreprocesado, clasificacion: Clasificacion,
                         ultimos_recomendados: list[str]) -> Respuesta:
        coincidencias = self.inventario.buscar(mensaje)
        if not coincidencias:
            return Respuesta("No encontré un servicio que cubra esa funcionalidad. " + PASOS_ANTE_BRECHA)
        lineas = ["Encontré estos servicios que pueden cubrir la funcionalidad:"]
        lineas += [f"  {i}) {_resumen(c.servicio)} - puntaje {c.puntaje}"
                   for i, c in enumerate(coincidencias, 1)]
        lineas.append("Escribe 'detalle del primero' o el ID para ver su contrato.")
        return Respuesta("\n".join(lineas), [c.servicio["id"] for c in coincidencias])

    def _analizar_requerimiento(self, mensaje: MensajePreprocesado, clasificacion: Clasificacion,
                                ultimos_recomendados: list[str]) -> Respuesta:
        necesidades = clasificacion.necesidades
        if not necesidades:
            return Respuesta("Describe las funciones de la pantalla separadas por comas. "
                             "Ej.: 'pantalla para buscar cliente, ver su saldo y registrar un pago'.")
        cubiertas, brechas, recomendados = [], [], []
        for necesidad in necesidades:
            servicio = self.inventario.mejor_coincidencia(necesidad)
            if servicio is None:
                brechas.append(f"  [BRECHA] '{necesidad.original}' -> sin servicio en el inventario")
                continue
            cubiertas.append(f"  [OK] '{necesidad.original}' -> {servicio['id']} {servicio['nombre']} "
                             f"({_operacion(servicio)})")
            if servicio["id"] not in recomendados:
                recomendados.append(servicio["id"])
        cobertura = round(100 * len(cubiertas) / len(necesidades))
        lineas = [f"Analicé {len(necesidades)} necesidades funcionales. Cobertura: {cobertura}%.",
                  *cubiertas, *brechas]
        if brechas:
            lineas.append("Sugerencia: valida las brechas con el equipo de arquitectura "
                          "antes de crear un servicio nuevo.")
        return Respuesta("\n".join(lineas), recomendados)

    def _detalle_servicio(self, mensaje: MensajePreprocesado, clasificacion: Clasificacion,
                          ultimos_recomendados: list[str]) -> Respuesta:
        servicio = self._servicio_referido(clasificacion, ultimos_recomendados)
        if servicio is None:
            return Respuesta("No identifiqué el servicio. Indica su ID (ej. SRV-007) "
                             "o busca primero una funcionalidad.")
        return Respuesta(_contrato(servicio))

    def _listar_dominios(self, mensaje: MensajePreprocesado, clasificacion: Clasificacion,
                         ultimos_recomendados: list[str]) -> Respuesta:
        lineas = [f"El inventario tiene {len(self.inventario.servicios)} servicios en estos dominios:"]
        lineas += [f"  - {dominio}: {len(servicios)} servicio(s)"
                   for dominio, servicios in sorted(self.inventario.servicios_por_dominio.items())]
        lineas.append("Pide, por ejemplo, 'servicios de pagos' para ver el detalle de un dominio.")
        return Respuesta("\n".join(lineas))

    def _listar_por_dominio(self, mensaje: MensajePreprocesado, clasificacion: Clasificacion,
                            ultimos_recomendados: list[str]) -> Respuesta:
        servicios = self.inventario.servicios_por_dominio[clasificacion.dominio]
        texto = "\n".join([f"Servicios del dominio {clasificacion.dominio}:",
                           *(f"  - {_resumen(s)}" for s in servicios)])
        return Respuesta(texto, [s["id"] for s in servicios])

    def _reportar_brecha(self, mensaje: MensajePreprocesado, clasificacion: Clasificacion,
                         ultimos_recomendados: list[str]) -> Respuesta:
        return Respuesta("Cuando una funcionalidad no tiene servicio (brecha), te sugiero: " + PASOS_ANTE_BRECHA)

    def _analizar_imagen(self, mensaje: MensajePreprocesado, clasificacion: Clasificacion,
                         ultimos_recomendados: list[str]) -> Respuesta:
        return Respuesta("El análisis de imágenes de pantallas aún no está disponible en esta versión. "
                         "Se incorporará en el Avance 2 con clasificación de imágenes. Por ahora, "
                         "describe en texto los elementos de la pantalla (campos, botones, acciones).")

    def _servicio_referido(self, clasificacion: Clasificacion, ultimos_recomendados: list[str]) -> dict | None:
        """Resuelve el servicio por ID explícito, posición, servicio nombrado o última recomendación."""
        if clasificacion.id_servicio:
            return self.inventario.servicio_por_id.get(clasificacion.id_servicio)
        for posicion in clasificacion.posiciones:
            if posicion < len(ultimos_recomendados):
                return self.inventario.servicio_por_id[ultimos_recomendados[posicion]]
        if clasificacion.referencia and (nombrado := self.inventario.mejor_coincidencia(clasificacion.referencia)):
            return nombrado
        if ultimos_recomendados:
            return self.inventario.servicio_por_id[ultimos_recomendados[0]]
        return None


# --------------------------------------------------------------------------
# Formato de respuestas
# --------------------------------------------------------------------------
PASOS_ANTE_BRECHA = ("\n  1) Reformular con otros términos o revisar servicios similares."
                     "\n  2) Verificar si un servicio existente puede extenderse."
                     "\n  3) Registrar la brecha con el equipo de arquitectura para evaluar un servicio nuevo.")


def _operacion(servicio: dict) -> str:
    return f"{servicio['metodo']} {servicio['endpoint']}"


def _resumen(servicio: dict) -> str:
    return f"{servicio['id']} {servicio['nombre']} [{_operacion(servicio)}]"


def _contrato(servicio: dict) -> str:
    lineas = [f"{servicio['id']} - {servicio['nombre']} (dominio {servicio['dominio']})",
              f"  Método/endpoint: {_operacion(servicio)}",
              f"  URL: {servicio['url']}",
              f"  Descripción: {servicio['descripcion']}"]
    if servicio.get("contexto"):
        lineas.append(f"  Recurso: {servicio['contexto']}")
    lineas += [f"  Entrada: {servicio['entrada']}", f"  Salida: {servicio['salida']}"]
    if servicio.get("controles"):
        lineas.append("  Controles: " + "; ".join(servicio["controles"]))
    return "\n".join(lineas)
