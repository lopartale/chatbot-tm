"""
Entrenamiento y evaluación del componente textual (Avance 2)

    python -m evaluacion.evaluar_intenciones

Genera las evidencias del reporte en evidencias/:
  1. preprocesamiento_intenciones.txt  ejemplos paso a paso y efecto de cada paso en el vocabulario
  2. representacion_tfidf.txt          tamaño de la matriz TF-IDF y términos por intención
  3. evaluacion_intenciones.txt/json   métricas del SVM, comparaciones y análisis de errores
     matriz_confusion_intenciones.png

Comparaciones (no cambian el modelo elegido):
  * el mismo SVM con sólo unigramas
  * el asistente completo (SVM + corrección por inventario de los fuera de alcance); se
    diseñó conociendo los errores del SVM, por lo que su cifra es optimista
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from lenguaje.clasificador_svm import (REGULARIZACION_C, SEMILLA, entrenar_modelo_svm, guardar_modelo_svm,
                              margen_de_decision, terminos_decisivos)
from evaluacion.metricas import calcular_metricas, encabezado_de_metricas, fila_de_metricas
from lenguaje.preprocesamiento import (mostrar_pasos, normalizar_mensaje, preprocesar_intencion,
                              quitar_palabras_vacias, tokenizar)
from evaluacion.registro_evidencias import guardar_json, guardar_matriz_confusion, guardar_texto
from lenguaje.representacion_texto import INTENCIONES, cargar_conjunto_intenciones, crear_vectorizador, terminos_con_ngramas


EJEMPLOS_PREPROCESAMIENTO = [
    "¿Qué hago si no existe el servicio que necesito?",
    "Pantalla de captura de datos del cliente con domicilio y correo",
    "Detalle de SRV-012",
    "ayudaaa, q puedes hacer",
    "Te comparto el mockup de la pantalla",
    "¿Hay partido de futbol hoy?",
]


def guardar_evidencia(nombre_archivo: str, lineas: list[str]) -> None:
    """Guarda la evidencia (registro_evidencias.py) y la muestra en pantalla, separada de la siguiente."""
    guardar_texto(nombre_archivo, lineas, mostrar=True)
    print()


def metricas_de(esperadas, obtenidas) -> dict[str, float]:
    """Accuracy, precision, recall y F1 macro sobre las 9 intenciones."""
    return calcular_metricas(esperadas, obtenidas, INTENCIONES)


# --------------------------------------------------------------------------
# Preprocesamiento
# --------------------------------------------------------------------------
def vocabulario_por_paso(textos: list[str]) -> list[tuple[str, int, int]]:
    """(paso, términos distintos, términos totales), agregando un paso a la vez."""
    pasos = {
        "tokenización (sin normalizar)": lambda texto: tokenizar(texto.lower()),
        "+ normalización": lambda texto: tokenizar(normalizar_mensaje(texto)),
        "+ palabras vacías": lambda texto: quitar_palabras_vacias(tokenizar(normalizar_mensaje(texto))),
        "+ stemming": preprocesar_intencion,
    }
    filas = []
    for nombre_paso, aplicar_paso in pasos.items():
        terminos = [termino for texto in textos for termino in aplicar_paso(texto)]
        filas.append((nombre_paso, len(set(terminos)), len(terminos)))
    return filas


def evidencia_preprocesamiento(entrenamiento: pd.DataFrame, todas_las_frases: pd.Series) -> None:
    """Ejemplos paso a paso, tamaño del vocabulario tras cada paso y frases que quedan vacías."""
    lineas = ["===== Ejemplos paso a paso ====="]
    for ejemplo in EJEMPLOS_PREPROCESAMIENTO:
        for paso, resultado in mostrar_pasos(ejemplo).items():
            lineas.append(f"{paso:20}: {resultado}")
        lineas.append("")
    lineas.append(f"===== Vocabulario del conjunto de entrenamiento ({len(entrenamiento)} frases) =====")
    lineas.append(f"{'paso':32}{'términos distintos':>20}{'términos totales':>18}")
    for nombre_paso, distintos, totales in vocabulario_por_paso(entrenamiento["texto"].tolist()):
        lineas.append(f"{nombre_paso:32}{distintos:>20}{totales:>18}")
    # Una frase sin términos no se podría clasificar: debe haber cero
    frases_vacias = [texto for texto in todas_las_frases if not preprocesar_intencion(texto)]
    lineas += ["", f"Frases que quedan sin términos: {len(frases_vacias)} {frases_vacias}"]
    guardar_evidencia("preprocesamiento_intenciones.txt", lineas)


# --------------------------------------------------------------------------
# Representación TF-IDF
# --------------------------------------------------------------------------
def terminos_representativos(vectorizador, matriz, intenciones: pd.Series, cantidad: int = 6) -> dict:
    """Términos con mayor TF-IDF promedio en las frases de cada intención."""
    vocabulario = vectorizador.get_feature_names_out()
    resultado = {}
    for intencion in intenciones.unique():
        filas_de_la_intencion = (intenciones == intencion).to_numpy()
        # Promedio de cada columna (término) en las filas (frases) de la intención
        promedio = np.asarray(matriz[filas_de_la_intencion].mean(axis=0)).ravel()
        mejores = promedio.argsort()[::-1][:cantidad]
        resultado[intencion] = [(vocabulario[i], round(float(promedio[i]), 3)) for i in mejores]
    return resultado


def porcentaje_desconocidos(textos, vectorizador, ngramas) -> tuple[int, int]:
    """(términos de los textos que no están en el vocabulario, términos totales)."""
    terminos = [termino for texto in textos for termino in terminos_con_ngramas(texto, ngramas)]
    desconocidos = sum(termino not in vectorizador.vocabulary_ for termino in terminos)
    return desconocidos, len(terminos)


def evidencia_tfidf(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> None:
    """Tamaño y densidad de la matriz TF-IDF, términos desconocidos y términos por intención."""
    vectorizador = crear_vectorizador()
    matriz_entrenamiento = vectorizador.fit_transform(entrenamiento["texto"])  # aprende vocabulario e IDF
    matriz_prueba = vectorizador.transform(prueba["texto"])                    # usa ese mismo vocabulario
    vocabulario = vectorizador.get_feature_names_out()
    bigramas = sum(" " in termino for termino in vocabulario)        # un bigrama tiene dos palabras
    desconocidos, total = porcentaje_desconocidos(prueba["texto"], vectorizador, (1, 2))
    unigramas_desconocidos, total_unigramas = porcentaje_desconocidos(prueba["texto"], vectorizador, (1, 1))
    frases_sin_terminos = int((matriz_prueba.getnnz(axis=1) == 0).sum())
    densidad = matriz_entrenamiento.nnz / np.prod(matriz_entrenamiento.shape)  # nnz: valores distintos de cero

    lineas = [
        "===== Matriz TF-IDF =====",
        f"Entrenamiento: {matriz_entrenamiento.shape[0]} frases x {matriz_entrenamiento.shape[1]} términos "
        f"({len(vocabulario) - bigramas} unigramas, {bigramas} bigramas)",
        f"Prueba: {matriz_prueba.shape[0]} frases x {matriz_prueba.shape[1]} términos (mismo vocabulario)",
        f"Valores distintos de cero: {matriz_entrenamiento.nnz} ({100 * densidad:.2f} % de la matriz; "
        f"{matriz_entrenamiento.getnnz(axis=1).mean():.1f} términos por frase en promedio)",
        f"Términos de prueba fuera del vocabulario: {desconocidos}/{total} ({100 * desconocidos / total:.1f} %); "
        f"sólo unigramas: {unigramas_desconocidos}/{total_unigramas} "
        f"({100 * unigramas_desconocidos / total_unigramas:.1f} %)",
        f"Frases de prueba sin ningún término conocido: {frases_sin_terminos}",
        "",
        "===== Términos con mayor TF-IDF promedio por intención (entrenamiento) =====",
    ]
    representativos = terminos_representativos(vectorizador, matriz_entrenamiento, entrenamiento["intencion"])
    for intencion, terminos in representativos.items():
        lineas.append(f"{intencion:24}: " + ", ".join(f"{termino} ({peso})" for termino, peso in terminos))

    ejemplo = "¿Qué hago si no existe el servicio que necesito?"
    vector = vectorizador.transform([ejemplo])
    lineas += ["", f"===== Vector de ejemplo (frase de entrenamiento): {ejemplo!r} =====",
               f"Términos: {terminos_con_ngramas(ejemplo)}"]
    for indice, peso in sorted(zip(vector.indices, vector.data), key=lambda par: -par[1]):
        lineas.append(f"  {vocabulario[indice]:22} tf-idf={peso:.3f}  idf={vectorizador.idf_[indice]:.3f}")
    guardar_evidencia("representacion_tfidf.txt", lineas)


# --------------------------------------------------------------------------
# Clasificador SVM y comparaciones
# --------------------------------------------------------------------------
def intenciones_del_asistente(textos) -> list[str]:
    """Intención final que registra el asistente completo, cada frase en una sesión nueva."""
    from asistente.chatbot_servicios import AsistenteServiciosREST

    intenciones = []
    for texto in textos:
        asistente = AsistenteServiciosREST()  # sesión nueva: sin contexto de turnos anteriores
        asistente.responder(texto)
        intenciones.append(asistente.historial[-1]["intencion"])
    return intenciones


def metricas_por_tipo(prueba: pd.DataFrame, obtenidas: list[str]) -> dict[str, dict]:
    """Métricas separadas para los casos normales y los difíciles."""
    return {tipo: metricas_de(grupo["intencion"], [obtenidas[i] for i in grupo.index])
            for tipo, grupo in prueba.groupby("tipo")}


def errores_del_svm(modelo, prueba: pd.DataFrame, predichas: list[str]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Frases de prueba mal clasificadas, con su margen de decisión y los términos que pesaron más."""
    margenes = margen_de_decision(modelo, prueba["texto"])
    es_correcta = (np.array(predichas) == prueba["intencion"]).to_numpy()
    errores = []
    for i in np.flatnonzero(~es_correcta):  # índices de las frases con predicción incorrecta
        frase = prueba.loc[i]
        errores.append({"texto": frase["texto"], "esperada": frase["intencion"], "obtenida": predichas[i],
                        "tipo": frase["tipo"], "margen": round(float(margenes[i]), 3),
                        "terminos_decisivos": terminos_decisivos(modelo, frase["texto"], predichas[i])})
    return errores, margenes, es_correcta


def regresion_avance1(modelo) -> tuple[int, list]:
    """Clasifica con el SVM los turnos de las conversaciones del Avance 1 (en el Avance 1 todos eran correctos)."""
    from evaluacion.pruebas_conversaciones import CONVERSACIONES

    turnos = [(mensaje, esperada) for turnos_conversacion in CONVERSACIONES.values()
              for mensaje, esperada, *_ in turnos_conversacion]
    obtenidas = modelo.predict([mensaje for mensaje, _ in turnos])
    fallas = [(mensaje, esperada, obtenida)
              for (mensaje, esperada), obtenida in zip(turnos, obtenidas) if esperada != obtenida]
    return len(turnos), fallas


def redactar_evaluacion(entrenamiento, prueba, resultados, reporte_por_intencion, matriz,
                        total_regresion, fallas_regresion, margenes, es_correcta) -> list[str]:
    """Texto de evidencias/evaluacion_intenciones.txt."""
    ancho = 44
    errores = resultados["errores"]
    aciertos_regresion = total_regresion - len(fallas_regresion)
    return [
        f"Entrenamiento: {len(entrenamiento)} frases | Prueba: {len(prueba)} frases "
        f"({(prueba['tipo'] == 'dificil').sum()} difíciles) | LinearSVC C={REGULARIZACION_C}",
        "", "===== Resumen (promedios macro sobre las 9 intenciones) =====", encabezado_de_metricas(ancho),
        fila_de_metricas("SVM uni+bigramas - entrenamiento", resultados["entrenamiento"], ancho),
        fila_de_metricas("SVM uni+bigramas - prueba", resultados["prueba"], ancho),
        fila_de_metricas("  prueba, casos normales", resultados["prueba_por_tipo"]["normal"], ancho),
        fila_de_metricas("  prueba, casos difíciles", resultados["prueba_por_tipo"]["dificil"], ancho),
        fila_de_metricas("SVM sólo unigramas - prueba (comparación)", resultados["svm_unigramas"], ancho),
        fila_de_metricas("Asistente integrado - prueba (optimista)", resultados["asistente_integrado"], ancho),
        fila_de_metricas("  prueba, casos normales", resultados["asistente_integrado_por_tipo"]["normal"], ancho),
        fila_de_metricas("  prueba, casos difíciles", resultados["asistente_integrado_por_tipo"]["dificil"], ancho),
        "", "===== SVM uni+bigramas: métricas por intención (prueba) =====", reporte_por_intencion,
        "===== Matriz de confusión SVM (filas: real, columnas: predicha) =====",
        pd.DataFrame(matriz, index=INTENCIONES, columns=[intencion[:8] for intencion in INTENCIONES]).to_string(),
        "", "===== Regresión: turnos de las conversaciones del Avance 1 =====",
        f"SVM: {aciertos_regresion}/{total_regresion} (Avance 1: {total_regresion}/{total_regresion})",
        *[f"  FALLA | {mensaje!r} | esperada={esperada} | obtenida={obtenida}"
          for mensaje, esperada, obtenida in fallas_regresion],
        "", "===== Margen de decisión (mejor puntaje - segundo mejor) =====",
        f"Aciertos: mediana {np.median(margenes[es_correcta]):.3f}, mínimo {margenes[es_correcta].min():.3f}",
        f"Errores:  mediana {np.median(margenes[~es_correcta]):.3f}, máximo {margenes[~es_correcta].max():.3f}",
        "", f"===== Errores del SVM en la prueba ({len(errores)}) =====",
        *[f"  [{error['tipo']}] {error['texto']!r} | esperada={error['esperada']} | obtenida={error['obtenida']} | "
          f"margen={error['margen']} | términos: {error['terminos_decisivos']}" for error in errores],
    ]


def evaluar_svm(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> None:
    """Entrena el SVM, lo evalúa con el conjunto de prueba y lo compara con las alternativas."""
    # 1. Entrenamiento (sólo con el 80 % de entrenamiento) y modelo que usará el asistente
    modelo = entrenar_modelo_svm(entrenamiento)
    guardar_modelo_svm(modelo)

    # 2. Evaluación con el 20 % de prueba: métricas, reporte por intención y matriz de confusión
    predichas = list(modelo.predict(prueba["texto"]))
    reporte_por_intencion = classification_report(prueba["intencion"], predichas, labels=INTENCIONES,
                                                  digits=3, zero_division=0)
    matriz = confusion_matrix(prueba["intencion"], predichas, labels=INTENCIONES)
    etiquetas = [intencion.replace("_", "\n", 1) for intencion in INTENCIONES]  # nombres en dos renglones
    guardar_matriz_confusion("matriz_confusion_intenciones.png", matriz, etiquetas,
                             "Matriz de confusión – SVM (TF-IDF uni+bigramas), conjunto de prueba",
                             nombre_eje="Intención", tamano=(9, 7.5), tamano_titulo=12)

    # 3. Comparaciones con las mismas frases de prueba
    del_asistente = intenciones_del_asistente(prueba["texto"])                      # SVM + inventario
    solo_unigramas = entrenar_modelo_svm(entrenamiento, (1, 1)).predict(prueba["texto"])

    # 4. Regresión con el Avance 1 y análisis de errores
    total_regresion, fallas_regresion = regresion_avance1(modelo)
    errores, margenes, es_correcta = errores_del_svm(modelo, prueba, predichas)

    # 5. Evidencias (texto para el reporte y JSON con todos los valores)
    resultados = {
        "parametros": {"modelo": "LinearSVC", "C": REGULARIZACION_C, "ngramas": [1, 2], "random_state": SEMILLA},
        "entrenamiento": metricas_de(entrenamiento["intencion"], modelo.predict(entrenamiento["texto"])),
        "prueba": metricas_de(prueba["intencion"], predichas),
        "prueba_por_tipo": metricas_por_tipo(prueba, predichas),
        "svm_unigramas": metricas_de(prueba["intencion"], solo_unigramas),
        "asistente_integrado": metricas_de(prueba["intencion"], del_asistente),
        "asistente_integrado_por_tipo": metricas_por_tipo(prueba, del_asistente),
        "regresion_avance1": {"aciertos": total_regresion - len(fallas_regresion), "total": total_regresion},
        "matriz_confusion": {"etiquetas": INTENCIONES, "valores": matriz.tolist()},
        "errores": errores,
    }
    guardar_evidencia("evaluacion_intenciones.txt",
                      redactar_evaluacion(entrenamiento, prueba, resultados, reporte_por_intencion, matriz,
                                          total_regresion, fallas_regresion, margenes, es_correcta))
    guardar_json("evaluacion_intenciones.json", resultados)


def main() -> None:
    """Genera las tres evidencias del componente textual, en el orden del reporte."""
    entrenamiento, prueba = cargar_conjunto_intenciones()
    todas_las_frases = pd.concat([entrenamiento["texto"], prueba["texto"]])
    evidencia_preprocesamiento(entrenamiento, todas_las_frases)
    evidencia_tfidf(entrenamiento, prueba)
    evaluar_svm(entrenamiento, prueba)


if __name__ == "__main__":
    main()
