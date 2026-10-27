"""Pruebas automáticas (pytest) sobre las conversaciones de pruebas_conversaciones.py.

Cada conversación es un caso: falla si algún turno no tiene la intención o el servicio esperado.
La prueba de generalización no se incluye: es una métrica, no un criterio de aceptación;
sus resultados se guardan en evidencias/ junto con la transcripción de las conversaciones.
"""
import pytest

from preprocesamiento import normalizar
from pruebas_conversaciones import CONVERSACIONES, ejecutar, evaluar_conversacion, prueba_generalizacion


@pytest.fixture(scope="module", autouse=True)
def evidencias():
    """Al terminar las pruebas, regenera evidencias/: transcripción, resultados y generalización."""
    yield
    ejecutar()
    prueba_generalizacion()


# IDs sin acentos: pytest escapa los caracteres no ASCII ("B\xfasqueda")
@pytest.mark.parametrize("turnos", CONVERSACIONES.values(),
                         ids=[normalizar(nombre) for nombre in CONVERSACIONES])
def test_conversacion(turnos):
    fallos = [f"{r['mensaje']!r}: intención {r['obtenida']} (esperada {r['esperada']}), "
              f"servicio {r['operacion_obtenida']} (esperado {r['operacion_esperada']})"
              for r in evaluar_conversacion(turnos) if not r["correcto"]]
    assert not fallos, "\n".join(fallos)
