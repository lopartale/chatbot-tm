"""Pruebas automáticas (pytest) sobre las conversaciones de pruebas_conversaciones.py.

Cada conversación es un caso: falla si algún turno no tiene la intención o el servicio esperado.
La prueba de generalización no se incluye: es una métrica, no un criterio de aceptación.
Las pruebas no escriben en evidencias/; la transcripción y la generalización se generan con
python -m evaluacion.pruebas_conversaciones.
"""
import pytest

from evaluacion.pruebas_conversaciones import CONVERSACIONES, evaluar_conversacion
from lenguaje.preprocesamiento import normalizar


# Regresiones conocidas del Avance 2: con reglas (Avance 1) estas conversaciones pasaban; con el
# clasificador SVM fallan. Se documentan como fallos esperados en lugar de ocultarlos con reglas
# ad hoc o de entrenar con estas frases. strict=True: si un cambio las corrige, pytest lo avisa.
REGRESIONES_CONOCIDAS = {
    "C13 Fuera de dominio con palabra disparadora":
        "El SVM clasifica '¿Qué ropa uso para mañana?' como charla (margen 1.14): 'mañana' y 'uso' "
        "provienen de frases de charla del entrenamiento ('hasta mañana', '¿cómo te uso?').",
    "C18 Requerimiento que contiene 'no existe'":
        "El SVM clasifica 'servicio para validar que el cliente no existe...' como reportar_brecha "
        "(margen 0.85): 'no' y 'no exist' son los términos con más peso hacia esa intención.",
}


def _caso(nombre, turnos):
    """Caso de pytest por conversación; las regresiones conocidas se marcan como fallo esperado."""
    marcas = ([pytest.mark.xfail(reason=REGRESIONES_CONOCIDAS[nombre], strict=True)]
              if nombre in REGRESIONES_CONOCIDAS else [])
    # IDs sin acentos: pytest escapa los caracteres no ASCII ("B\xfasqueda")
    return pytest.param(turnos, id=normalizar(nombre), marks=marcas)


@pytest.mark.parametrize("turnos", [_caso(nombre, turnos) for nombre, turnos in CONVERSACIONES.items()])
def test_conversacion(turnos):
    """Cada turno debe tener la intención y el servicio esperados."""
    fallos = [f"{r['mensaje']!r}: intención {r['obtenida']} (esperada {r['esperada']}), "
              f"servicio {r['operacion_obtenida']} (esperado {r['operacion_esperada']})"
              for r in evaluar_conversacion(turnos) if not r["correcto"]]
    assert not fallos, "\n".join(fallos)
