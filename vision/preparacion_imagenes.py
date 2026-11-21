"""
Preparación de las imágenes para ResNet50

    captura -> ajustar_a_cuadro (224 x 224 sin deformar) -> imagen_a_arreglo (píxeles RGB de 0 a 255)

La normalización que espera ResNet50 la aplica después keras.applications.resnet50.preprocess_input
(clasificador_visual.py), la misma con la que se entrenó la red en ImageNet.

y, sólo para el entrenamiento, aumentar_imagen (recorte parcial, brillo y contraste).
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance

TAMANO_ENTRADA = 224                                         # ResNet50 recibe imágenes de 224 x 224


def ajustar_a_cuadro(imagen: Image.Image) -> Image.Image:
    """Escala la captura sin deformarla y la centra en un cuadro de 224 x 224 relleno con el color del borde."""
    imagen = imagen.convert("RGB")
    escala = TAMANO_ENTRADA / max(imagen.size)  # el lado mayor queda en 224 px
    nuevo_ancho = max(1, round(imagen.width * escala))
    nuevo_alto = max(1, round(imagen.height * escala))
    reducida = imagen.resize((nuevo_ancho, nuevo_alto), Image.BICUBIC)
    color_del_borde = imagen.getpixel((0, imagen.height - 1))  # esquina inferior izquierda: suele ser fondo
    cuadro = Image.new("RGB", (TAMANO_ENTRADA, TAMANO_ENTRADA), color_del_borde)
    cuadro.paste(reducida, ((TAMANO_ENTRADA - nuevo_ancho) // 2, (TAMANO_ENTRADA - nuevo_alto) // 2))
    return cuadro


def aumentar_imagen(imagen: Image.Image, aleatorio: random.Random) -> Image.Image:
    """Aumento de datos: recorte parcial y cambios de brillo y contraste.
    No se voltea la imagen: una pantalla con el texto en espejo no es realista."""
    ancho, alto = imagen.size
    # Recorte: conserva del 80 al 100 % del ancho y del 70 al 100 % del alto, en una posición al azar
    ancho_recorte = int(ancho * aleatorio.uniform(0.8, 1.0))
    alto_recorte = int(alto * aleatorio.uniform(0.7, 1.0))
    x = aleatorio.randint(0, ancho - ancho_recorte)
    y = aleatorio.randint(0, alto - alto_recorte)
    imagen = imagen.crop((x, y, x + ancho_recorte, y + alto_recorte))
    imagen = ImageEnhance.Brightness(imagen).enhance(aleatorio.uniform(0.85, 1.15))
    return ImageEnhance.Contrast(imagen).enhance(aleatorio.uniform(0.85, 1.15))


def imagen_a_arreglo(imagen: Image.Image) -> np.ndarray:
    """Imagen -> arreglo (224, 224, 3) con los píxeles RGB de 0 a 255, el formato de entrada de Keras."""
    return np.asarray(ajustar_a_cuadro(imagen), np.float32)
