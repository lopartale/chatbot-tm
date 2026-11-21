"""
Carga de modelos guardados (.joblib)

Un modelo guardado sólo se puede usar con seguridad si se carga con la misma estructura de
módulos y la misma versión de scikit-learn con la que se entrenó. Esta función lo verifica
para que cada componente decida qué hacer si no es compatible (reentrenar o avisar).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
from sklearn.exceptions import InconsistentVersionWarning


def cargar_modelo_compatible(ruta: Path):
    """El modelo guardado, o None si no existe o no es compatible.

    No es compatible si se guardó con otra estructura de módulos (no se puede cargar) o con
    otra versión de scikit-learn (se puede cargar, pero sus resultados podrían cambiar).
    """
    if not Path(ruta).exists():
        return None
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", InconsistentVersionWarning)
        try:
            modelo = joblib.load(ruta)
        except (ModuleNotFoundError, AttributeError):
            return None
    if any(issubclass(aviso.category, InconsistentVersionWarning) for aviso in avisos):
        return None
    return modelo
