"""Pruebas (pytest) del preprocesamiento para el clasificador de intenciones."""
import pytest

from lenguaje.preprocesamiento import (TERMINO_ID_SERVICIO, mostrar_pasos, normalizar_mensaje,
                              preprocesar_intencion)
from lenguaje.representacion_texto import cargar_conjunto_intenciones


@pytest.mark.parametrize("texto, esperado", [
    ("¿Qué TAL?", "que tal?"),                              # minúsculas, sin acentos ni ¿
    ("Detalle de SRV-012", f"detalle de {TERMINO_ID_SERVICIO}"),
    ("srv 30", TERMINO_ID_SERVICIO),
    ("ayudaaa", "ayuda"),                                   # letras repetidas
    ("q puedes hacer", "que puedes hacer"),                 # abreviaturas
])
def test_normalizacion(texto, esperado):
    """Paso 1: minúsculas, IDs de servicio, letras repetidas y abreviaturas."""
    assert normalizar_mensaje(texto) == esperado


def test_conserva_palabras_que_distinguen_intenciones():
    """Paso 3: "si", "no", "hay" y "que" están en la lista de NLTK, pero indican una brecha o una pregunta."""
    pasos = mostrar_pasos("¿Qué hago si no hay servicio?")
    assert pasos["sin_palabras_vacias"] == ["que", "hago", "si", "no", "hay", "servicio"]


def test_quita_palabras_vacias_y_aplica_stemming():
    """Pasos 3 y 4 sobre una frase completa."""
    assert preprocesar_intencion("Te comparto el mockup de la pantalla") == ["compart", "mockup", "pantall"]


def test_ninguna_frase_del_conjunto_queda_vacia():
    """Todas las frases conservan al menos un término para el clasificador."""
    entrenamiento, prueba = cargar_conjunto_intenciones()
    for texto in list(entrenamiento["texto"]) + list(prueba["texto"]):
        assert preprocesar_intencion(texto), texto
