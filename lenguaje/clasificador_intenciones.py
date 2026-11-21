"""
Clasificador de intenciones

Decide qué pide el usuario a partir del mensaje preprocesado, sin consultar el inventario.

    mensaje preprocesado -> SVM (TF-IDF + LinearSVC) -> intención
                         -> ExtractorDatos           -> datos de la intención -> Clasificacion

    Clasificador SVM   TF-IDF + LinearSVC entrenado con data/intenciones.csv (Avance 2):
                       decide cualquiera de las 9 intenciones, incluidas charla y detalle_servicio
    Datos              extraccion_datos.py: reglas que extraen lo que cada intención necesita
                       (dominio, ID o posición del servicio, necesidades de un requerimiento)

La respuesta de los mensajes de charla la redacta respuestas_charla.py.
"""
from __future__ import annotations

from collections.abc import Iterable

from lenguaje.clasificador_svm import cargar_modelo_svm
from lenguaje.extraccion_datos import Clasificacion, ExtractorDatos
from lenguaje.preprocesamiento import MensajePreprocesado


class ClasificadorIntenciones:
    """Decide la intención de un mensaje con el SVM y extrae los datos que esa intención necesita."""

    def __init__(self, dominios: Iterable[str], modelo_svm=None):
        """dominios: los del inventario (para reconocerlos en el mensaje); modelo_svm: None carga el entrenado."""
        self.extractor_datos = ExtractorDatos(dominios)  # reglas de extracción (extraccion_datos.py)
        # Modelo TF-IDF + SVM ya entrenado (modelos/clasificador_intenciones.joblib)
        self.modelo_svm = modelo_svm if modelo_svm is not None else cargar_modelo_svm()

    def clasificar(self, mensaje: MensajePreprocesado) -> Clasificacion:
        """Avance 2: el SVM decide la intención y las reglas extraen sus datos.

        "Detalle de SRV-012" -> SVM: detalle_servicio -> Clasificacion(detalle_servicio, id_servicio="SRV-012")
        """
        # El SVM recibe el texto original: su Pipeline aplica el preprocesamiento y TF-IDF
        intencion = self.modelo_svm.predict([mensaje.original])[0]
        # Las reglas de extracción trabajan sobre el texto normalizado: dominio, ID o necesidades
        return self.extractor_datos.extraer(intencion, mensaje.texto)
