"""
Gestor de respuestas

Con la intención y los datos que entregó el clasificador, consulta la base de conocimiento
(base_conocimiento.py) y arma la respuesta. Cada intención tiene su manejador:

    buscar_servicio         los servicios más relevantes para la consulta
    analizar_requerimiento  un servicio (o una brecha) por cada necesidad, con la cobertura
    detalle_servicio        el contrato del servicio referido por ID, posición o nombre
    listar_dominios         los dominios del inventario
    listar_por_dominio      los servicios de un dominio
    reportar_brecha         los pasos a seguir cuando no hay servicio
    analizar_imagen         sin imagen adjunta: cómo enviarla (con imagen responde la integración multimodal)

Los manejadores no guardan estado: devuelven qué servicios recordar y la sesión los guarda
como contexto (sesion_conversacion.py).
"""
from __future__ import annotations

from typing import NamedTuple

from conocimiento.base_conocimiento import InventarioServicios
from lenguaje.extraccion_datos import Clasificacion
from conocimiento.formato_respuestas import PASOS_ANTE_BRECHA, contrato_servicio, operacion_http, resumen_servicio
from lenguaje.preprocesamiento import MensajePreprocesado

MENSAJE_SIN_IMAGEN = ("Para analizar una pantalla necesito la imagen. En la terminal escribe "
                      "'/imagen <ruta del archivo>'; en Colab usa bot.responder(texto, imagen='ruta.png'). "
                      "También puedes describir en texto los campos y acciones de la pantalla.")
MENSAJE_SIN_NECESIDADES = ("Describe las funciones de la pantalla separadas por comas. "
                           "Ej.: 'pantalla para buscar cliente, ver su saldo y registrar un pago'.")
MENSAJE_SERVICIO_NO_IDENTIFICADO = ("No identifiqué el servicio. Indica su ID (ej. SRV-007) "
                                    "o busca primero una funcionalidad.")
MENSAJE_SIN_RECOMENDACIONES = ("Todavía no te he recomendado servicios. Busca primero una funcionalidad "
                               "o indica el ID del servicio (ej. SRV-007).")
MENSAJE_POSICION_INEXISTENTE = ("Sólo te recomendé {cantidad} servicio(s); pide uno del 1 al {cantidad} "
                                "(ej. 'detalle del primero') o indica su ID.")


class Respuesta(NamedTuple):
    """Texto para el usuario y, si los hay, los servicios que la sesión debe recordar."""
    texto: str
    # Servicios que la sesión guarda para "detalle del primero";
    # None conserva el contexto anterior (p. ej. al mostrar un detalle)
    recomendados: list[str] | None = None


class GestorRespuestas:
    """Redacta la respuesta de cada intención consultando el inventario: un manejador por intención."""

    def __init__(self, inventario: InventarioServicios):
        """El gestor sólo lee el inventario; no guarda estado de la conversación."""
        self.inventario = inventario

    def coincide_con_inventario(self, mensaje: MensajePreprocesado) -> bool:
        """Nodo "¿Coincide con el inventario?": algún servicio cubre el mensaje."""
        return self.inventario.tiene_servicio_para(mensaje)

    def manejar(self, mensaje: MensajePreprocesado, clasificacion: Clasificacion,
                ultimos_recomendados: list[str]) -> Respuesta:
        """Llama al manejador de la intención, cada uno sólo con los datos que necesita."""
        intencion = clasificacion.intencion
        if intencion == "buscar_servicio":
            return self.buscar_servicio(mensaje)
        if intencion == "analizar_requerimiento":
            return self.analizar_requerimiento(clasificacion.necesidades)
        if intencion == "detalle_servicio":
            return self.detalle_servicio(clasificacion, ultimos_recomendados)
        if intencion == "listar_dominios":
            return self.listar_dominios()
        if intencion == "listar_por_dominio":
            return self.listar_por_dominio(clasificacion.dominio)
        if intencion == "reportar_brecha":
            return self.reportar_brecha()
        if intencion == "analizar_imagen":
            return self.pedir_imagen()
        raise ValueError(f"Intención sin manejador: {intencion}")

    # ----------------------------------------------------------------------
    # Manejadores de cada intención
    # ----------------------------------------------------------------------
    def buscar_servicio(self, mensaje: MensajePreprocesado) -> Respuesta:
        """Los servicios más relevantes para la consulta, con su puntaje."""
        coincidencias = self.inventario.buscar(mensaje)
        if not coincidencias:
            return Respuesta("No encontré un servicio que cubra esa funcionalidad. " + PASOS_ANTE_BRECHA)
        lineas = ["Encontré estos servicios que pueden cubrir la funcionalidad:"]
        lineas += [f"  {numero}) {resumen_servicio(coincidencia.servicio)} - puntaje {coincidencia.puntaje}"
                   for numero, coincidencia in enumerate(coincidencias, 1)]
        lineas.append("Escribe 'detalle del primero' o el ID para ver su contrato.")
        return Respuesta("\n".join(lineas), [coincidencia.servicio["id"] for coincidencia in coincidencias])

    def analizar_requerimiento(self, necesidades: list[MensajePreprocesado]) -> Respuesta:
        """Busca un servicio por necesidad; las que no tienen servicio son brechas."""
        if not necesidades:
            return Respuesta(MENSAJE_SIN_NECESIDADES)
        cubiertas, brechas, recomendados = [], [], []
        for necesidad in necesidades:
            servicio = self.inventario.mejor_coincidencia(necesidad)
            if servicio is None:
                brechas.append(f"  [BRECHA] '{necesidad.original}' -> sin servicio en el inventario")
                continue
            cubiertas.append(f"  [OK] '{necesidad.original}' -> {servicio['id']} {servicio['nombre']} "
                             f"({operacion_http(servicio)})")
            if servicio["id"] not in recomendados:  # un servicio puede cubrir varias necesidades
                recomendados.append(servicio["id"])
        cobertura = round(100 * len(cubiertas) / len(necesidades))
        lineas = [f"Analicé {len(necesidades)} necesidades funcionales. Cobertura: {cobertura}%.",
                  *cubiertas, *brechas]
        if brechas:
            lineas.append("Sugerencia: valida las brechas con el equipo de arquitectura "
                          "antes de crear un servicio nuevo.")
        return Respuesta("\n".join(lineas), recomendados)

    def detalle_servicio(self, clasificacion: Clasificacion, ultimos_recomendados: list[str]) -> Respuesta:
        """Contrato del servicio al que se refiere el usuario; no cambia el contexto."""
        if clasificacion.posiciones and not clasificacion.id_servicio:
            return self._detalle_por_posicion(clasificacion.posiciones, ultimos_recomendados)
        servicio = self._servicio_referido(clasificacion, ultimos_recomendados)
        if servicio is None:
            return Respuesta(MENSAJE_SERVICIO_NO_IDENTIFICADO)
        return Respuesta(contrato_servicio(servicio))

    def listar_dominios(self) -> Respuesta:
        """Dominios del inventario con su número de servicios."""
        lineas = [f"El inventario tiene {len(self.inventario.servicios)} servicios en estos dominios:"]
        lineas += [f"  - {dominio}: {len(servicios)} servicio(s)"
                   for dominio, servicios in sorted(self.inventario.servicios_por_dominio.items())]
        lineas.append("Pide, por ejemplo, 'servicios de pagos' para ver el detalle de un dominio.")
        return Respuesta("\n".join(lineas))

    def listar_por_dominio(self, dominio: str) -> Respuesta:
        """Servicios de un dominio; quedan como contexto para pedir su detalle."""
        servicios = self.inventario.servicios_por_dominio[dominio]
        texto = "\n".join([f"Servicios del dominio {dominio}:",
                           *(f"  - {resumen_servicio(servicio)}" for servicio in servicios)])
        return Respuesta(texto, [servicio["id"] for servicio in servicios])

    @staticmethod
    def reportar_brecha() -> Respuesta:
        """Pasos a seguir cuando una funcionalidad no tiene servicio."""
        return Respuesta("Cuando una funcionalidad no tiene servicio (brecha), te sugiero: " + PASOS_ANTE_BRECHA)

    @staticmethod
    def pedir_imagen() -> Respuesta:
        """El usuario anuncia una imagen sin adjuntarla ("te comparto el mockup"): se le explica cómo enviarla.

        Los turnos con imagen no pasan por aquí: los resuelve la integración multimodal
        (asistente/chatbot_servicios.py -> multimodal/integracion_multimodal.py).
        """
        return Respuesta(MENSAJE_SIN_IMAGEN)

    # ----------------------------------------------------------------------
    # Apoyo
    # ----------------------------------------------------------------------
    def _detalle_por_posicion(self, posiciones: list[int], ultimos_recomendados: list[str]) -> Respuesta:
        """"detalle del segundo": el servicio en esa posición de la última recomendación.

        Si la posición no existe se avisa; no se busca otro servicio, porque mostraría un
        contrato distinto del que el usuario pidió.
        """
        if not ultimos_recomendados:
            return Respuesta(MENSAJE_SIN_RECOMENDACIONES)
        for posicion in posiciones:
            if posicion < len(ultimos_recomendados):
                return Respuesta(contrato_servicio(self.inventario.servicio_por_id[ultimos_recomendados[posicion]]))
        return Respuesta(MENSAJE_POSICION_INEXISTENTE.format(cantidad=len(ultimos_recomendados)))

    def _servicio_referido(self, clasificacion: Clasificacion, ultimos_recomendados: list[str]) -> dict | None:
        """Resuelve el servicio en este orden: ID explícito, servicio nombrado
        ("contrato del servicio de cancelación de seguros") o última recomendación."""
        if clasificacion.id_servicio:
            return self.inventario.servicio_por_id.get(clasificacion.id_servicio)
        if clasificacion.referencia and (nombrado := self.inventario.mejor_coincidencia(clasificacion.referencia)):
            return nombrado
        if ultimos_recomendados:
            return self.inventario.servicio_por_id[ultimos_recomendados[0]]
        return None
