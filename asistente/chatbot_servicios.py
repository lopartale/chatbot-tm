"""
Asistente para la identificación de servicios REST - Avance 2 (versión multimodal)
Python 3.10+, NLTK, scikit-learn (TF-IDF + SVM) y ResNet50 (Keras).

El asistente recibe consultas o requerimientos funcionales en español, o capturas de
pantallas de frontend, los relaciona con el inventario de servicios REST (base de
conocimiento) y recomienda los servicios existentes, señalando brechas de cobertura.

Arquitectura: cada componente vive en su propio módulo y este archivo sólo los conecta.

    mensaje -> Preprocesamiento             (preprocesamiento.py)
            -> Clasificador de intenciones  (clasificador_intenciones.py: TF-IDF + SVM)
            -> Gestor de respuestas         (gestor_respuestas.py) <- base de conocimiento
            -> respuesta                                              (inventario_raml.py)
              (si la intención es charla, la respuesta la redacta respuestas_charla.py)

    imagen  -> Clasificador visual          (clasificador_visual.py: ResNet50 + softmax)
            -> Integración multimodal       (integracion_multimodal.py: patrón de pantalla)
            -> Gestor de respuestas         -> respuesta

    Estado entre turnos (contexto e historial): sesion_conversacion.py
    Modo interactivo (entrada y salida):        consola.py

Responsabilidad de este módulo: el flujo conversacional de cada turno
(AsistenteServiciosREST.responder).
"""
from __future__ import annotations

from pathlib import Path

from configuracion.rutas import RUTA_INVENTARIO
from conocimiento.base_conocimiento import InventarioServicios
from lenguaje.clasificador_intenciones import ClasificadorIntenciones
from lenguaje.extraccion_datos import Clasificacion
from conocimiento.gestor_respuestas import GestorRespuestas
from multimodal.integracion_multimodal import AnalizadorPantallas, nombre_de_imagen, redactar_respuesta_de_pantalla
from lenguaje.preprocesamiento import MensajePreprocesado, preprocesar
from lenguaje.respuestas_charla import RespuestasCharla
from asistente.sesion_conversacion import SesionConversacion


MENSAJE_FUERA_DE_ALCANCE = ("No entendí tu consulta o está fuera de mi alcance. Puedo buscar servicios REST "
                            "por funcionalidad, analizar requerimientos de pantallas o mostrar contratos. "
                            "Escribe 'ayuda' para ver ejemplos.")


class AsistenteServiciosREST:
    """Orquestador del asistente: decide qué componente atiende cada turno y registra la conversación.

    No clasifica, no busca ni redacta por sí mismo: delega en el clasificador de intenciones, el
    gestor de respuestas, las respuestas de charla y el analizador de pantallas.
    """

    def __init__(self, ruta_inventario: str | Path = RUTA_INVENTARIO):
        """Crea los componentes del asistente a partir del inventario RAML."""
        # Componentes: cada uno con una sola responsabilidad
        self.inventario = InventarioServicios(ruta_inventario)                        # base de conocimiento
        self.clasificador = ClasificadorIntenciones(self.inventario.servicios_por_dominio)  # qué pide el usuario
        self.gestor = GestorRespuestas(self.inventario)                               # respuesta con el inventario
        self.charla = RespuestasCharla()                                              # saludos, ayuda, gracias...
        self.analizador_pantallas = AnalizadorPantallas()                             # tipo de pantalla de una imagen
        self.sesion = SesionConversacion()                                            # contexto e historial

    # Atajos al estado de la sesión (los usan las pruebas y los notebooks)
    @property
    def historial(self) -> list[dict]:
        """Turnos de la sesión: mensaje, intención, respuesta y, si hubo imagen, pantalla y confianza."""
        return self.sesion.historial

    @property
    def ultimos_recomendados(self) -> list[str]:
        """IDs recomendados en la última respuesta que recomendó servicios ("detalle del primero")."""
        return self.sesion.ultimos_recomendados

    # ----------------------------------------------------------------------
    # Flujo conversacional
    # ----------------------------------------------------------------------
    def responder(self, mensaje: str, imagen=None) -> str:
        """Un turno de conversación. `imagen` (ruta o PIL.Image) es opcional."""
        if imagen is not None:
            return self._turno_con_imagen(mensaje, imagen)
        return self._turno_de_texto(mensaje)

    def _turno_de_texto(self, mensaje: str) -> str:
        """Preprocesar -> decidir la intención -> charla, fuera de alcance o base de conocimiento."""
        preprocesado = preprocesar(mensaje)
        clasificacion = self._decidir_intencion(preprocesado)

        if clasificacion.intencion == "fuera_de_alcance":
            return self.sesion.registrar_turno(mensaje, "fuera_de_alcance", MENSAJE_FUERA_DE_ALCANCE)
        if clasificacion.intencion == "charla":
            return self.sesion.registrar_turno(mensaje, "charla", self.charla.responder(preprocesado))
        return self._responder_con_inventario(mensaje, preprocesado, clasificacion)

    def _turno_con_imagen(self, mensaje: str, imagen) -> str:
        """Clasificador visual -> patrón de pantalla -> servicios del inventario.

        Con imagen la intención es siempre analizar_imagen: el texto que la acompaña no se clasifica.
        """
        nombre = nombre_de_imagen(imagen)
        pantalla, aviso = self.analizador_pantallas.analizar(imagen)
        if aviso:  # imagen inexistente, archivo que no es imagen o modelo visual sin entrenar
            return self.sesion.registrar_turno(mensaje, "analizar_imagen", aviso, imagen=nombre)

        # Integración multimodal: patrón de la pantalla -> servicios del inventario, cobertura y brechas
        texto, recomendados = redactar_respuesta_de_pantalla(pantalla, self.inventario.servicios)
        self.sesion.recordar_recomendados(recomendados)
        return self.sesion.registrar_turno(mensaje, "analizar_imagen", texto, imagen=nombre,
                                           pantalla=pantalla["clase"], confianza=round(pantalla["probabilidad"], 3))

    # ----------------------------------------------------------------------
    # Pasos del flujo
    # ----------------------------------------------------------------------
    def _decidir_intencion(self, preprocesado: MensajePreprocesado) -> Clasificacion:
        """Intención del SVM, con la corrección por inventario del Avance 1.

        Nodo "¿Coincide con el inventario?": si el SVM decide fuera_de_alcance pero el inventario
        tiene un servicio que cubre el mensaje, se trata como una búsqueda. Lo mismo con una lista
        por dominio cuyo dominio no se reconoce.
        """
        clasificacion = self.clasificador.clasificar(preprocesado)
        if clasificacion.intencion == "fuera_de_alcance" and self.gestor.coincide_con_inventario(preprocesado):
            return Clasificacion("buscar_servicio")
        # listar_por_dominio sin un dominio reconocible ("servicio para pagar el recibo de luz"):
        # si el inventario tiene un servicio que lo cubre es una búsqueda; si no, se listan los dominios
        if clasificacion.intencion == "listar_por_dominio" and clasificacion.dominio is None:
            if self.gestor.coincide_con_inventario(preprocesado):
                return Clasificacion("buscar_servicio")
            return Clasificacion("listar_dominios")
        return clasificacion

    def _responder_con_inventario(self, mensaje: str, preprocesado: MensajePreprocesado,
                                  clasificacion: Clasificacion) -> str:
        """El gestor arma la respuesta con la base de conocimiento; la sesión guarda contexto e historial."""
        respuesta = self.gestor.manejar(preprocesado, clasificacion, self.sesion.ultimos_recomendados)
        self.sesion.recordar_recomendados(respuesta.recomendados)
        return self.sesion.registrar_turno(mensaje, clasificacion.intencion, respuesta.texto)

    # ----------------------------------------------------------------------
    # Consulta directa (sin conversación)
    # ----------------------------------------------------------------------
    def recomendar(self, consulta: str) -> dict | None:
        """Mejor servicio del inventario para una consulta, sin pasar por el clasificador."""
        return self.inventario.mejor_coincidencia(preprocesar(consulta))


def main() -> None:
    """Modo interactivo: python -m asistente (la entrada y salida están en consola.py)."""
    from asistente.consola import ejecutar_consola

    ejecutar_consola(AsistenteServiciosREST(RUTA_INVENTARIO))


if __name__ == "__main__":
    main()
