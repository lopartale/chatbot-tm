"""
Métricas de evaluación compartidas por el componente textual y el visual (Avance 2)

    accuracy, precision, recall y F1 (promedio macro: todas las clases pesan igual)
    tabla de métricas para las evidencias (las guarda registro_evidencias.py)
"""
from __future__ import annotations

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calcular_metricas(esperadas, obtenidas, clases: list[str]) -> dict[str, float]:
    """Accuracy y promedios macro de precision, recall y F1.

    accuracy:  aciertos / total de casos
    precision: de los casos que el modelo asignó a una clase, cuántos eran de esa clase
    recall:    de los casos de una clase, cuántos reconoció el modelo
    F1:        media armónica de precision y recall
    Macro: se calcula cada métrica por clase y se promedia, así todas las clases pesan igual.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        esperadas, obtenidas, labels=clases, average="macro", zero_division=0)
    return {
        "accuracy": round(accuracy_score(esperadas, obtenidas), 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
    }


def encabezado_de_metricas(ancho: int) -> str:
    """Encabezado de la tabla de métricas de las evidencias (ancho = columna del nombre)."""
    return f"{'modelo / conjunto':{ancho}}{'accuracy':>9}{'precision':>11}{'recall':>9}{'f1':>9}"


def fila_de_metricas(nombre: str, metricas: dict[str, float], ancho: int) -> str:
    """Una fila de la tabla de métricas, alineada con encabezado_de_metricas."""
    return (f"{nombre:{ancho}}{metricas['accuracy']:>9.3f}{metricas['precision_macro']:>11.3f}"
            f"{metricas['recall_macro']:>9.3f}{metricas['f1_macro']:>9.3f}")
