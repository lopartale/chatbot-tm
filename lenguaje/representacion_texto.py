"""
Representación del texto con TF-IDF (Avance 2)

Convierte cada mensaje en un vector TF-IDF, que es lo que recibe el clasificador SVM:

    mensaje -> preprocesar_intencion() -> raíces -> unigramas y bigramas -> vector TF-IDF

Configuración (fijada antes de evaluar, sin usar el conjunto de prueba):
  * Unigramas y bigramas: los bigramas conservan combinaciones que cambian la intención
    aunque cada palabra sola sea ambigua ("si no", "no exist", "del primer").
  * Frecuencia sublineal (1 + log tf): una palabra repetida en un requerimiento largo
    no domina el vector.
  * Normalización L2: mensajes cortos y largos quedan en la misma escala.
  * El vocabulario y el IDF se aprenden sólo con el conjunto de entrenamiento.

También carga el conjunto de datos de intenciones (data/intenciones.csv).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from configuracion.rutas import RUTA_CONJUNTO_INTENCIONES
from lenguaje.preprocesamiento import preprocesar_intencion


# Las 9 intenciones del diseño conversacional (Avance 1), en el orden de las tablas y matrices
INTENCIONES = ["charla", "buscar_servicio", "analizar_requerimiento", "detalle_servicio",
               "listar_dominios", "listar_por_dominio", "reportar_brecha", "analizar_imagen",
               "fuera_de_alcance"]

UNIGRAMAS_Y_BIGRAMAS = (1, 2)  # términos de una palabra y de dos palabras consecutivas


def cargar_conjunto_intenciones(ruta: str | Path = RUTA_CONJUNTO_INTENCIONES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frases de entrenamiento y de prueba (columnas: texto, intencion, conjunto, tipo)."""
    frases = pd.read_csv(ruta, encoding="utf-8")
    # La división 80/20 ya viene marcada en la columna "conjunto" para que siempre sea la misma
    entrenamiento = frases[frases["conjunto"] == "entrenamiento"].reset_index(drop=True)
    prueba = frases[frases["conjunto"] == "prueba"].reset_index(drop=True)
    return entrenamiento, prueba


def terminos_con_ngramas(texto: str, ngramas: tuple[int, int] = UNIGRAMAS_Y_BIGRAMAS) -> list[str]:
    """Raíces del mensaje y sus combinaciones de palabras consecutivas.

    "¿Qué hago si no existe?" -> raíces [que, hag, si, no, exist]
                              -> [que, hag, si, no, exist, que hag, hag si, si no, no exist]
    """
    raices = preprocesar_intencion(texto)            # preprocesamiento para el clasificador
    minimo, maximo = ngramas
    terminos = []
    for tamano in range(minimo, maximo + 1):         # tamaño 1: unigramas; tamaño 2: bigramas
        for inicio in range(len(raices) - tamano + 1):
            terminos.append(" ".join(raices[inicio:inicio + tamano]))
    return terminos


def crear_vectorizador(ngramas: tuple[int, int] = UNIGRAMAS_Y_BIGRAMAS) -> TfidfVectorizer:
    """TfidfVectorizer de scikit-learn que usa el preprocesamiento propio en lugar del suyo.

    Al entrenar (fit) aprende el vocabulario y el IDF de cada término; al transformar,
    convierte cada mensaje en un vector con un peso por término del vocabulario.
    """
    # partial (y no lambda) para que el modelo entrenado se pueda guardar con joblib
    return TfidfVectorizer(analyzer=partial(terminos_con_ngramas, ngramas=ngramas),
                           sublinear_tf=True, norm="l2", smooth_idf=True)
