"""
Sesión de conversación

Guarda el estado que se conserva entre turnos de una misma conversación:

    contexto   IDs de los últimos servicios recomendados, para que el usuario pueda pedir
               "dame el contrato del tercero" sin repetir el ID
    historial  un registro por turno (mensaje, intención, respuesta y datos de la imagen);
               es la evidencia que revisan las pruebas

No interpreta mensajes ni consulta el inventario: sólo recuerda.
"""
from __future__ import annotations


class SesionConversacion:
    """Estado de una conversación: el historial de turnos y los últimos servicios recomendados."""

    def __init__(self):
        """Sesión vacía, sin turnos ni recomendaciones."""
        self.ultimos_recomendados: list[str] = []  # contexto: "detalle del primero"
        self.historial: list[dict] = []

    def recordar_recomendados(self, recomendados: list[str] | None) -> None:
        """Actualiza el contexto; None significa que la respuesta no recomendó servicios y se conserva el anterior."""
        if recomendados is not None:
            self.ultimos_recomendados = recomendados

    def registrar_turno(self, mensaje: str, intencion: str, respuesta: str, **detalles) -> str:
        """Agrega el turno al historial y devuelve la respuesta para que el asistente la entregue."""
        self.historial.append({"usuario": mensaje, "intencion": intencion, "bot": respuesta, **detalles})
        return respuesta
