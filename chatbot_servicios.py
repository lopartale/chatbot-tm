"""
Asistente para la identificación de servicios REST - Avance 1
Chatbot basado en reglas implementado con Python 3.10+ y NLTK.

El asistente recibe consultas o requerimientos funcionales en español,
los relaciona con el inventario de servicios REST (base de conocimiento)
y recomienda los servicios existentes, señalando brechas de cobertura.

Arquitectura: cada mensaje recorre tres componentes, uno por módulo.

    mensaje -> 1. Preprocesamiento        (preprocesamiento.py)
            -> 2. Clasificador de intenciones (clasificador_intenciones.py)
            -> 3. Gestor de respuestas    (gestor_respuestas.py) <- base de conocimiento
            -> respuesta                                            (inventario_raml.py)

Este módulo conecta los componentes, aplica el flujo conversacional de cada turno
(AsistenteServiciosREST.responder) y ofrece el modo interactivo.
"""
from __future__ import annotations

from pathlib import Path

from clasificador_intenciones import Clasificacion, ClasificadorIntenciones
from gestor_respuestas import GestorRespuestas, InventarioServicios
from preprocesamiento import normalizar, preprocesar

RUTA_INVENTARIO = Path(__file__).parent / "data" / "plataforma_servicios_bancarios.raml"

PALABRAS_SALIDA = {"salir", "adios", "terminar"}

MENSAJE_FUERA_DE_ALCANCE = ("No entendí tu consulta o está fuera de mi alcance. Puedo buscar servicios REST "
                            "por funcionalidad, analizar requerimientos de pantallas o mostrar contratos. "
                            "Escribe 'ayuda' para ver ejemplos.")


class AsistenteServiciosREST:
    def __init__(self, ruta_inventario: str | Path = RUTA_INVENTARIO):
        self.inventario = InventarioServicios(ruta_inventario)  # base de conocimiento
        self.clasificador = ClasificadorIntenciones(self.inventario.servicios_por_dominio)
        self.gestor = GestorRespuestas(self.inventario)
        self.ultimos_recomendados: list[str] = []  # contexto de sesión: IDs para "detalle del primero"
        self.historial: list[dict] = []

    def responder(self, mensaje: str) -> str:
        """Un turno de conversación, nodo por nodo según el flujo conversacional (ver README)."""
        # Normalizar (y el resto del preprocesamiento)
        preprocesado = preprocesar(mensaje)

        # ¿Charla? -> sí: respuesta de charla
        if respuesta_charla := self.clasificador.charla(preprocesado):
            return self._cerrar_turno(mensaje, "charla", respuesta_charla)

        # Reglas de intención en orden
        clasificacion = self.clasificador.aplicar_reglas(preprocesado)

        # Ninguna regla -> ¿Coincide con el inventario?
        if clasificacion is None:
            if not self.gestor.coincide_con_inventario(preprocesado):
                return self._cerrar_turno(mensaje, "fuera_de_alcance", MENSAJE_FUERA_DE_ALCANCE)
            clasificacion = Clasificacion("buscar_servicio")

        # Manejador de la intención <-> base de conocimiento
        respuesta = self.gestor.manejar(preprocesado, clasificacion, self.ultimos_recomendados)

        # Respuesta + contexto
        if respuesta.recomendados is not None:
            self.ultimos_recomendados = respuesta.recomendados
        return self._cerrar_turno(mensaje, clasificacion.intencion, respuesta.texto)

    def _cerrar_turno(self, mensaje: str, intencion: str, respuesta: str) -> str:
        self.historial.append({"usuario": mensaje, "intencion": intencion, "bot": respuesta})
        return respuesta

    def recomendar(self, consulta: str) -> dict | None:
        """Mejor servicio del inventario para una consulta, sin pasar por el clasificador."""
        return self.inventario.mejor_coincidencia(preprocesar(consulta))


# --------------------------------------------------------------------------
# Ejecución interactiva (Colab / terminal)
# --------------------------------------------------------------------------
def main() -> None:
    bot = AsistenteServiciosREST(RUTA_INVENTARIO)
    print(f"Asistente de servicios REST - inventario: {bot.inventario.titulo} "
          f"({len(bot.inventario.servicios)} servicios, RAML)")
    print("Escribe 'salir' para terminar")
    while True:
        try:
            entrada = input("Tú: ")
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C o fin de la entrada
            print()
            break
        print("Bot:", bot.responder(entrada))
        if normalizar(entrada).strip(" !.,") in PALABRAS_SALIDA:
            break


if __name__ == "__main__":
    main()
