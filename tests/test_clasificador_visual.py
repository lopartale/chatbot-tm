"""Pruebas (pytest) del componente visual: generación, preparación de imágenes y clasificador."""
import importlib.util
import random

import pytest
from PIL import Image

from vision.clasificador_visual import RUTA_MODELO_PANTALLAS
from vision.generar_pantallas import CARPETA_IMAGENES, DOMINIO_POR_PANTALLA, planear_imagenes
from vision.plantillas_pantallas import GeneradorAleatorio, armar_pagina_html
from vision.preparacion_imagenes import TAMANO_ENTRADA, ajustar_a_cuadro, aumentar_imagen, imagen_a_arreglo

MODELO_LISTO = (RUTA_MODELO_PANTALLAS.exists() and (CARPETA_IMAGENES / "imagenes.csv").exists()
                and importlib.util.find_spec("keras") is not None)


def test_plan_balanceado_y_sin_bocetos_en_entrenamiento():
    """1,360 imágenes y los bocetos sólo en el conjunto de estilo nuevo."""
    imagenes = planear_imagenes()
    assert len(imagenes) == 1360
    assert {imagen["estilo"] for imagen in imagenes if imagen["conjunto"] != "estilo_nuevo"} == {"alta", "front"}
    assert {imagen["estilo"] for imagen in imagenes if imagen["conjunto"] == "estilo_nuevo"} == {"boceto"}


def test_pagina_reproducible_con_la_misma_semilla():
    """La misma semilla produce exactamente la misma pantalla."""
    paginas = [armar_pagina_html(GeneradorAleatorio(random.Random(7), "alta"), "login", "movil") for _ in range(2)]
    assert paginas[0] == paginas[1] and "<button" in paginas[0]


@pytest.mark.parametrize("tamano", [(390, 844), (1366, 768), (50, 50)])
def test_ajuste_a_224_sin_deformar(tamano):
    """Cualquier tamaño de captura termina en 224 x 224."""
    assert ajustar_a_cuadro(Image.new("RGB", tamano, "white")).size == (TAMANO_ENTRADA, TAMANO_ENTRADA)


def test_arreglo_y_aumento():
    """Forma de la entrada de ResNet50 en Keras (alto, ancho, canales) y recorte del aumento de datos."""
    imagen = Image.new("RGB", (512, 300), "white")
    arreglo = imagen_a_arreglo(imagen)
    assert arreglo.shape == (TAMANO_ENTRADA, TAMANO_ENTRADA, 3) and arreglo.max() == 255
    assert aumentar_imagen(imagen, random.Random(1)).size[0] <= 512


@pytest.mark.skipif(not MODELO_LISTO,
                    reason="Requiere generar_pantallas.py y evaluar_pantallas.py (y Keras con los pesos de ResNet50)")
def test_clasifica_una_pantalla_de_prueba():
    """El clasificador devuelve tipo, dominio, probabilidad y dos alternativas."""
    from vision.clasificador_visual import ClasificadorPantallas

    resultado = ClasificadorPantallas().clasificar(CARPETA_IMAGENES / "prueba" / "documentos" / "documentos_000.jpg")
    assert resultado["clase"] in DOMINIO_POR_PANTALLA
    assert resultado["dominio"] == DOMINIO_POR_PANTALLA[resultado["clase"]]
    assert 0 < resultado["probabilidad"] <= 1 and len(resultado["alternativas"]) == 2
