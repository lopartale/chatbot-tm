"""
Prepara ResNet50 (preentrenada con ImageNet) como extractor de características con Keras (Avance 2)

    python -m vision.preparar_resnet50     # descarga los pesos de ImageNet (~95 MB, sólo la primera vez)

Se usa `keras.applications.ResNet50` con:
  * weights="imagenet":  pesos aprendidos con 1.28 millones de imágenes de ImageNet
  * include_top=False:   sin la capa de clasificación de ImageNet (1000 clases)
  * pooling="avg":       termina en el global average pooling, un vector de 2048 valores por imagen

Ese vector es lo que recibe la capa softmax del proyecto (transferencia de aprendizaje). Keras
guarda los pesos en ~/.keras/models, así que la descarga ocurre una sola vez por equipo.
Keras corre sobre TensorFlow (motor por omisión); con KERAS_BACKEND=torch o jax el código es el mismo.
"""
from __future__ import annotations

from vision.preparacion_imagenes import TAMANO_ENTRADA


class ResNet50NoDisponible(Exception):
    """No se pudieron obtener los pesos de ResNet50 (sin conexión o la descarga falló)."""


def crear_resnet50_congelada():
    """ResNet50 de ImageNet sin su capa final, terminada en el global average pooling y congelada.

    Congelada (trainable=False) significa que sus pesos no cambian al entrenar: sólo se usa para
    describir cada imagen con 2048 características.
    """
    import keras  # importación diferida: Keras/TensorFlow tarda en cargar y sólo se necesita con imágenes

    try:
        resnet50 = keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg",
                                               input_shape=(TAMANO_ENTRADA, TAMANO_ENTRADA, 3))
    except Exception as error:  # Keras informa la descarga fallida con una excepción genérica
        raise ResNet50NoDisponible(f"No se pudieron obtener los pesos de ResNet50: {error}") from error
    resnet50.trainable = False  # transferencia de aprendizaje: las capas convolucionales no se entrenan
    return resnet50


def preparar_resnet50():
    """Descarga los pesos (si hace falta) y devuelve la ResNet50 congelada."""
    return crear_resnet50_congelada()


if __name__ == "__main__":
    import keras
    import numpy as np

    modelo = preparar_resnet50()
    print("Motor de Keras:", keras.backend.backend(), "| Keras", keras.__version__)
    print("Entrada:", modelo.input_shape, "| Salida:", modelo.output_shape)
    print("Pesos entrenables:", len(modelo.trainable_weights), "| congelados:", len(modelo.non_trainable_weights))
    print("Forma con una imagen en blanco:", modelo.predict(np.zeros((1, TAMANO_ENTRADA, TAMANO_ENTRADA, 3)),
                                                             verbose=0).shape)
