"""
Clasificador visual de pantallas con ResNet50, transferencia de aprendizaje y Keras (Avance 2)

    imagen -> ajuste a 224x224 (preparacion_imagenes.py) -> ResNet50 congelada (preparar_resnet50.py)
           -> 2048 características -> capa Dense softmax (capa_softmax.py) -> tipo de pantalla

Transferencia de aprendizaje por extracción de características: las capas convolucionales de
ResNet50, entrenadas con ImageNet, se usan sin modificar y sólo se entrena la capa Dense softmax.
Es la receta habitual de Keras:

    base = keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")
    base.trainable = False
    modelo = keras.Sequential([base, keras.layers.Dense(8, activation="softmax")])

Aquí las dos partes se ejecutan por separado: ResNet50 describe cada imagen una sola vez (sus
características se guardan en evaluar_pantallas.py) y la capa Dense se entrena sobre ellas. El
resultado es el mismo que entrenar el modelo completo con la base congelada, sin pasar las 1,680
imágenes por ResNet50 en cada época.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from configuracion.rutas import CARPETA_MODELOS
from vision.capa_softmax import CapaSoftmax
from vision.generar_pantallas import DOMINIO_POR_PANTALLA
from vision.preparacion_imagenes import TAMANO_ENTRADA, imagen_a_arreglo
from vision.preparar_resnet50 import preparar_resnet50

RUTA_MODELO_PANTALLAS = CARPETA_MODELOS / "clasificador_pantallas.keras"


# --------------------------------------------------------------------------
# Extracción de características con ResNet50 (capas congeladas)
# --------------------------------------------------------------------------
class ExtractorResNet50:
    """ResNet50 preentrenada sin su capa de clasificación: convierte cada imagen en 2048 características."""

    def __init__(self):
        """Carga ResNet50 congelada; descarga los pesos de ImageNet la primera vez."""
        self.resnet50 = preparar_resnet50()

    @staticmethod
    def descripcion() -> str:
        """Lo que determina las características: si cambia, las características guardadas ya no sirven."""
        import keras

        return (f"ResNet50 ImageNet, pooling avg, {TAMANO_ENTRADA}x{TAMANO_ENTRADA}, preprocess_input de Keras | "
                f"Keras {keras.__version__} | motor {keras.backend.backend()}")

    def extraer(self, imagenes: list[Image.Image], tamano_lote: int = 16) -> np.ndarray:
        """Matriz (número de imágenes, 2048). Las imágenes se procesan en lotes para limitar la memoria."""
        from keras.applications.resnet50 import preprocess_input

        lotes = []
        for inicio in range(0, len(imagenes), tamano_lote):
            # Lote de forma (imágenes, 224, 224, 3) con píxeles de 0 a 255
            lote = np.stack([imagen_a_arreglo(imagen) for imagen in imagenes[inicio:inicio + tamano_lote]])
            # preprocess_input: la misma normalización con la que se entrenó ResNet50 en ImageNet
            lotes.append(np.asarray(self.resnet50.predict_on_batch(preprocess_input(lote))))
        return np.concatenate(lotes)


# --------------------------------------------------------------------------
# Clasificador que usa el asistente
# --------------------------------------------------------------------------
class ModeloVisualNoDisponible(Exception):
    """La capa softmax no está entrenada (falta modelos/clasificador_pantallas.keras), Keras no está
    instalado o el modelo no se pudo abrir."""


class ClasificadorPantallas:
    """Clasifica una captura: tipo de pantalla, dominio del inventario y probabilidades."""

    def __init__(self, ruta_modelo: str | Path = RUTA_MODELO_PANTALLAS):
        """Carga la capa softmax entrenada y ResNet50.

        Sin un modelo entrenado no se puede clasificar: se avisa con un error propio para no
        confundirlo con una imagen inválida. A diferencia del SVM, no se reentrena aquí, porque
        hace falta el conjunto de imágenes y varios minutos (evaluar_pantallas.py).
        """
        self.capa_softmax = CapaSoftmax.cargar(Path(ruta_modelo))
        if self.capa_softmax is None:
            raise ModeloVisualNoDisponible(str(ruta_modelo))
        self.extractor = ExtractorResNet50()

    def clasificar(self, imagen: str | Path | Image.Image) -> dict:
        """Imagen (ruta o PIL.Image) -> tipo de pantalla más probable, su dominio y las dos alternativas."""
        if not isinstance(imagen, Image.Image):
            imagen = Image.open(imagen)
        caracteristicas = self.extractor.extraer([imagen])                               # ResNet50: 2048 valores
        probabilidades = self.capa_softmax.predecir_probabilidades(caracteristicas)[0]  # softmax: 8 probabilidades
        tipos = self.capa_softmax.clases
        orden = probabilidades.argsort()[::-1]  # de la más a la menos probable
        tipo_elegido = str(tipos[orden[0]])
        return {
            "clase": tipo_elegido,
            "dominio": DOMINIO_POR_PANTALLA[tipo_elegido],
            "probabilidad": float(probabilidades[orden[0]]),
            "alternativas": [(str(tipos[i]), float(probabilidades[i])) for i in orden[1:3]],
        }
