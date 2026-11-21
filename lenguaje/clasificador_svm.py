"""
Clasificador de intenciones con SVM (Avance 2)

Modelo que decide la intención de un mensaje:

    mensaje -> vector TF-IDF (representacion_texto.py) -> LinearSVC -> intención

Se usa un SVM con kernel lineal porque los vectores TF-IDF tienen muchas dimensiones y
pocos valores distintos de cero, y porque cada intención queda descrita por un peso por
término, lo que permite explicar cada decisión (terminos_decisivos). C = 1 es el valor por
defecto y no se ajustó con el conjunto de prueba; el conjunto está casi balanceado (charla tiene
66 frases y las demás entre 55 y 57), así que no se usan pesos por clase.

Cómo decide: el SVM aprende un vector de pesos por intención (uno por término del
vocabulario). Para un mensaje calcula un puntaje por intención (suma de peso TF-IDF del
término x peso aprendido, más un sesgo) y elige la intención con el puntaje más alto.

El entrenamiento y la evaluación completos están en evaluar_intenciones.py.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from configuracion.modelos_guardados import cargar_modelo_compatible
from configuracion.rutas import CARPETA_MODELOS
from lenguaje.representacion_texto import UNIGRAMAS_Y_BIGRAMAS, cargar_conjunto_intenciones, crear_vectorizador

RUTA_MODELO_SVM = CARPETA_MODELOS / "clasificador_intenciones.joblib"
REGULARIZACION_C = 1.0  # C pequeño = margen más amplio y más tolerante a errores de entrenamiento
SEMILLA = 42            # resultados reproducibles


def crear_modelo_svm(ngramas: tuple[int, int] = UNIGRAMAS_Y_BIGRAMAS) -> Pipeline:
    """TF-IDF + SVM lineal en un solo objeto: recibe textos y devuelve intenciones.

    Con el Pipeline, predict(["Detalle de SRV-012"]) aplica el preprocesamiento, el vector
    TF-IDF y el SVM en un solo paso, igual en el entrenamiento que en el asistente.
    """
    return Pipeline([
        ("tfidf", crear_vectorizador(ngramas)),                         # texto -> vector TF-IDF
        ("svm", LinearSVC(C=REGULARIZACION_C, random_state=SEMILLA)),   # vector -> intención
    ])


def entrenar_modelo_svm(frases: pd.DataFrame, ngramas: tuple[int, int] = UNIGRAMAS_Y_BIGRAMAS) -> Pipeline:
    """Aprende el vocabulario, el IDF y los pesos del SVM con las frases de entrenamiento."""
    return crear_modelo_svm(ngramas).fit(frases["texto"], frases["intencion"])


def guardar_modelo_svm(modelo: Pipeline, ruta: Path = RUTA_MODELO_SVM) -> None:
    """Guarda el modelo entrenado para que el asistente no tenga que reentrenarlo al iniciar."""
    ruta.parent.mkdir(exist_ok=True)
    joblib.dump(modelo, ruta)


def cargar_modelo_svm(ruta: Path = RUTA_MODELO_SVM) -> Pipeline:
    """Modelo guardado; si no existe o no es compatible, lo entrena y lo guarda.

    No es compatible si se guardó con otra estructura del código o con otra versión de
    scikit-learn (configuracion/modelos_guardados.py). Entrenar el SVM toma segundos y es
    determinista, así que el modelo resultante es el mismo.
    """
    modelo = cargar_modelo_compatible(ruta)
    if modelo is not None:
        return modelo
    entrenamiento, _ = cargar_conjunto_intenciones()  # el conjunto de prueba nunca se usa para entrenar
    modelo = entrenar_modelo_svm(entrenamiento)
    guardar_modelo_svm(modelo, ruta)
    return modelo


def margen_de_decision(modelo: Pipeline, textos) -> np.ndarray:
    """Diferencia entre el mejor y el segundo mejor puntaje del SVM: qué tan clara fue la decisión."""
    # decision_function: una fila por texto y una columna por intención (el puntaje de cada una)
    puntajes_ordenados = np.sort(modelo.decision_function(textos), axis=1)
    return puntajes_ordenados[:, -1] - puntajes_ordenados[:, -2]


def terminos_decisivos(modelo: Pipeline, texto: str, intencion: str, cantidad: int = 3) -> list[tuple[str, float]]:
    """Términos del mensaje que más empujan hacia una intención (peso TF-IDF x coeficiente del SVM)."""
    vectorizador = modelo.named_steps["tfidf"]
    svm = modelo.named_steps["svm"]
    vector = vectorizador.transform([texto])                          # sólo los términos presentes
    coeficientes = svm.coef_[list(svm.classes_).index(intencion)]     # pesos aprendidos para la intención
    vocabulario = vectorizador.get_feature_names_out()
    # Aporte de cada término al puntaje de la intención
    aportes = [(vocabulario[indice], peso * coeficientes[indice])
               for indice, peso in zip(vector.indices, vector.data)]
    aportes.sort(key=lambda termino_y_aporte: -termino_y_aporte[1])   # del mayor al menor aporte
    return [(termino, round(float(aporte), 3)) for termino, aporte in aportes[:cantidad]]
