"""Pruebas (pytest) del clasificador de intenciones TF-IDF + SVM."""
import pytest

from lenguaje.clasificador_svm import entrenar_modelo_svm, margen_de_decision, terminos_decisivos
from lenguaje.representacion_texto import INTENCIONES, cargar_conjunto_intenciones, terminos_con_ngramas


@pytest.fixture(scope="module")
def modelo():
    """SVM entrenado una sola vez para todas las pruebas del módulo."""
    entrenamiento, _ = cargar_conjunto_intenciones()
    return entrenar_modelo_svm(entrenamiento)


def test_terminos_con_unigramas_y_bigramas():
    """El analizador del TF-IDF genera raíces y pares de raíces consecutivas."""
    assert terminos_con_ngramas("¿Qué hago si no existe?") == [
        "que", "hag", "si", "no", "exist", "que hag", "hag si", "si no", "no exist"]


def test_conoce_las_nueve_intenciones(modelo):
    """El modelo aprendió las 9 intenciones del diseño conversacional."""
    assert sorted(modelo.classes_) == sorted(INTENCIONES)


def test_predice_intenciones_validas_en_la_prueba(modelo):
    """Toda predicción es una de las 9 intenciones."""
    _, prueba = cargar_conjunto_intenciones()
    assert set(modelo.predict(prueba["texto"])) <= set(INTENCIONES)


@pytest.mark.parametrize("texto, esperada", [
    ("Detalle de SRV-012", "detalle_servicio"),
    ("Te comparto el mockup de la pantalla", "analizar_imagen"),
    ("¿Qué hago si no existe el servicio que necesito?", "reportar_brecha"),
])
def test_frases_de_entrenamiento(modelo, texto, esperada):
    """Frases vistas al entrenar se clasifican en su intención."""
    assert modelo.predict([texto])[0] == esperada


def test_margen_y_terminos_decisivos(modelo):
    """La explicación de una decisión señala los términos esperados."""
    assert margen_de_decision(modelo, ["Detalle de SRV-012"])[0] > 0
    termino_principal = terminos_decisivos(modelo, "Detalle de SRV-012", "detalle_servicio")[0][0]
    assert termino_principal in {"srvid", "detall", "detall srvid"}
