"""
Entrenamiento y evaluación del clasificador visual con Keras (Avance 2)

    python -m vision.preparar_resnet50     # una vez: pesos de ResNet50 (ImageNet) para Keras
    python -m vision.generar_pantallas     # una vez: conjunto de imágenes
    python -m evaluacion.evaluar_pantallas     # extrae características, entrena la capa Dense softmax y evalúa

Conjuntos (data/imagenes/imagenes.csv):
  * entrenamiento (840): ajusta la capa softmax (más una copia aumentada de cada imagen)
  * validacion (176):    elige la regularización L2 y el umbral de confianza de la integración
  * prueba (184):        métricas finales, mismo estilo que el entrenamiento
  * estilo_nuevo (160):  bocetos, estilo nunca visto: mide la generalización

Genera en evidencias/: evaluacion_pantallas.txt/json, las matrices de confusión de prueba y
estilo nuevo y la curva de entrenamiento de la capa softmax. Compara además con la misma capa
softmax entrenada sobre píxeles (sin transferencia de aprendizaje).

    main: características -> entrenamiento (L2 elegido con validación) -> evaluación -> evidencias
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from evaluacion.metricas import calcular_metricas, encabezado_de_metricas, fila_de_metricas
from evaluacion.registro_evidencias import (guardar_curva_entrenamiento, guardar_json, guardar_matriz_confusion,
                                           guardar_texto, ruta_de_evidencia)
from vision.capa_softmax import EPOCAS, SEMILLA, TAMANO_LOTE, TASA_APRENDIZAJE, CapaSoftmax, crear_capa_softmax
from vision.clasificador_visual import RUTA_MODELO_PANTALLAS, ExtractorResNet50
from vision.generar_pantallas import CARPETA_IMAGENES, TIPOS_DE_PANTALLA
from vision.preparacion_imagenes import ajustar_a_cuadro, aumentar_imagen

NOMBRE_CARACTERISTICAS_GUARDADAS = "caracteristicas_resnet50_keras.npz"  # caché de ResNet50 (no se versiona)
# Regularización L2 de la capa Dense: valores altos = pesos más pequeños = modelo más simple.
# Equivale a 1 / (2 · C · n) de la regresión logística: 0.03 corresponde a C ≈ 0.01 con 1,680 ejemplos
VALORES_DE_L2 = [0.1, 0.03, 0.01, 0.001, 0.0001]
UMBRALES_DE_CONFIANZA = [0.5, 0.6, 0.7, 0.8]
CONJUNTOS = ["entrenamiento", "validacion", "prueba", "estilo_nuevo"]
ANCHO_DEL_NOMBRE = 46  # columna de nombres en las tablas de texto de la evidencia


def metricas_de(esperadas, obtenidas) -> dict[str, float]:
    """Accuracy, precision, recall y F1 macro sobre los 8 tipos de pantalla."""
    return calcular_metricas(esperadas, obtenidas, TIPOS_DE_PANTALLA)


# --------------------------------------------------------------------------
# Características de ResNet50 (se guardan para no recalcularlas)
# --------------------------------------------------------------------------
def extraer_caracteristicas(extractor: ExtractorResNet50, rutas: list[str], con_aumento: bool = False) -> np.ndarray:
    """Abre cada imagen (aumentada si se pide) y la pasa por ResNet50: matriz (imágenes, 2048)."""
    imagenes = []
    for indice, ruta in enumerate(rutas):
        imagen = Image.open(CARPETA_IMAGENES / ruta)
        if con_aumento:
            imagen = aumentar_imagen(imagen, random.Random(SEMILLA + indice))  # semilla por imagen
        imagenes.append(imagen)
    return extractor.extraer(imagenes)


def firma_de_imagenes(imagenes: pd.DataFrame) -> str:
    """Huella de las imágenes y del extractor: ruta, tamaño y fecha de cada archivo, más la versión
    de Keras, el motor y la preparación de las imágenes.

    Cambia si las imágenes se regeneran (aunque imagenes.csv quede igual) o si cambia el extractor,
    por ejemplo al actualizar Keras o al pasar de TensorFlow a PyTorch.
    """
    huella = hashlib.sha256(ExtractorResNet50.descripcion().encode())
    for ruta in imagenes["ruta"]:
        datos = (CARPETA_IMAGENES / ruta).stat()
        huella.update(f"{ruta}|{datos.st_size}|{datos.st_mtime_ns}\n".encode())
    return huella.hexdigest()


def cargar_caracteristicas(imagenes: pd.DataFrame) -> dict[str, np.ndarray]:
    """Características de cada conjunto, y de la copia aumentada del entrenamiento.

    Se guardan porque ResNet50 tarda varios minutos con 1,360 imágenes; sólo se reutilizan si
    corresponden a las imágenes y al extractor actuales (misma firma), si no, se vuelven a extraer.
    """
    firma = firma_de_imagenes(imagenes)
    ruta_guardadas = ruta_de_evidencia(NOMBRE_CARACTERISTICAS_GUARDADAS)
    if ruta_guardadas.exists():
        guardadas = dict(np.load(ruta_guardadas))
        if str(guardadas.pop("firma", "")) == firma:
            return guardadas
        print("Las imágenes o el extractor cambiaron desde la última extracción: se recalculan las características.")
    extractor = ExtractorResNet50()
    caracteristicas = {}
    for conjunto, grupo in imagenes.groupby("conjunto"):
        rutas = list(grupo["ruta"])
        caracteristicas[conjunto] = extraer_caracteristicas(extractor, rutas)
        if conjunto == "entrenamiento":
            caracteristicas["entrenamiento_aumentado"] = extraer_caracteristicas(extractor, rutas, con_aumento=True)
    np.savez_compressed(ruta_guardadas, firma=np.array(firma), **caracteristicas)
    return caracteristicas


def pixeles_en_gris(rutas) -> np.ndarray:
    """Imágenes de 32 x 32 en gris como vectores: la entrada de la comparación sin transferencia."""
    vectores = []
    for ruta in rutas:
        gris = ajustar_a_cuadro(Image.open(CARPETA_IMAGENES / ruta)).convert("L").resize((32, 32))
        vectores.append(np.asarray(gris, np.float32).ravel() / 255)
    return np.stack(vectores)


# --------------------------------------------------------------------------
# Entrenamiento y evaluación
# --------------------------------------------------------------------------
def elegir_regularizacion(x_entrenamiento, y_entrenamiento, x_validacion, y_validacion) -> tuple[float, dict]:
    """L2 con el mejor F1 en validación; en un empate, el más alto (modelo más simple)."""
    resultados = {}
    for regularizacion in VALORES_DE_L2:
        capa = crear_capa_softmax(regularizacion).entrenar(x_entrenamiento, y_entrenamiento)
        resultados[regularizacion] = metricas_de(y_validacion, capa.predecir(x_validacion))
    # El conjunto de prueba no interviene: sólo entrenamiento y validación
    mejor_l2 = max(VALORES_DE_L2, key=lambda regularizacion: (resultados[regularizacion]["f1_macro"], regularizacion))
    return mejor_l2, resultados


def revisar_umbrales(capa: CapaSoftmax, x_validacion, y_validacion) -> list[str]:
    """Cuántos errores y cuántos aciertos de la validación quedarían por debajo de cada umbral."""
    probabilidades = capa.predecir_probabilidades(x_validacion)
    predichas = capa.clases[probabilidades.argmax(axis=1)]
    confianza = probabilidades.max(axis=1)
    es_correcta = predichas == y_validacion
    lineas = []
    for umbral in UMBRALES_DE_CONFIANZA:
        debajo = confianza < umbral  # turnos en los que el asistente pediría confirmación
        lineas.append(f"  umbral {umbral}: marca {(debajo & ~es_correcta).sum()} de {(~es_correcta).sum()} errores "
                      f"y {(debajo & es_correcta).sum()} de {es_correcta.sum()} aciertos")
    return lineas


def entrenar_capa_softmax(caracteristicas: dict, tipos_reales: dict):
    """Elige L2 con la validación y entrena la capa softmax final con las imágenes originales y aumentadas."""
    # Entrenamiento: imágenes originales + una copia aumentada de cada una (el doble de ejemplos)
    x_entrenamiento = np.vstack([caracteristicas["entrenamiento"], caracteristicas["entrenamiento_aumentado"]])
    y_entrenamiento = np.concatenate([tipos_reales["entrenamiento"]] * 2)
    mejor_l2, seleccion_l2 = elegir_regularizacion(x_entrenamiento, y_entrenamiento,
                                                   caracteristicas["validacion"], tipos_reales["validacion"])
    capa_softmax = crear_capa_softmax(mejor_l2).entrenar(x_entrenamiento, y_entrenamiento)
    capa_softmax.guardar(RUTA_MODELO_PANTALLAS)  # modelo que usa el asistente
    return capa_softmax, mejor_l2, seleccion_l2, len(y_entrenamiento)


def evaluar_por_conjunto(capa_softmax: CapaSoftmax, caracteristicas: dict, tipos_reales: dict) -> tuple[dict, dict]:
    """Métricas y reporte por clase en cada conjunto; matrices de confusión de prueba y estilo nuevo."""
    resultados, reportes_por_tipo = {}, {}
    for conjunto in CONJUNTOS:
        predichas = capa_softmax.predecir(caracteristicas[conjunto])
        resultados[conjunto] = metricas_de(tipos_reales[conjunto], predichas)
        reportes_por_tipo[conjunto] = classification_report(tipos_reales[conjunto], predichas,
                                                            labels=TIPOS_DE_PANTALLA, digits=3, zero_division=0)
        if conjunto in ("prueba", "estilo_nuevo"):
            matriz = confusion_matrix(tipos_reales[conjunto], predichas, labels=TIPOS_DE_PANTALLA)
            guardar_matriz_confusion(f"matriz_confusion_pantallas_{conjunto}.png", matriz, TIPOS_DE_PANTALLA,
                                     f"Matriz de confusión – ResNet50 + softmax, conjunto de {conjunto.replace('_', ' ')}")
            resultados[f"matriz_{conjunto}"] = matriz.tolist()
    return resultados, reportes_por_tipo


def comparar_sin_transferencia(imagenes: pd.DataFrame, tipos_reales: dict, mejor_l2: float) -> dict:
    """La misma capa softmax entrenada con píxeles en lugar de características de ResNet50."""
    rutas = {conjunto: grupo["ruta"] for conjunto, grupo in imagenes.groupby("conjunto")}
    sin_transferencia = crear_capa_softmax(mejor_l2).entrenar(pixeles_en_gris(rutas["entrenamiento"]),
                                                             tipos_reales["entrenamiento"])
    resultados = {}
    for conjunto in ("prueba", "estilo_nuevo"):
        predichas = sin_transferencia.predecir(pixeles_en_gris(rutas[conjunto]))
        resultados[f"sin_transferencia_{conjunto}"] = metricas_de(tipos_reales[conjunto], predichas)
    return resultados


def analizar_prueba(capa_softmax: CapaSoftmax, imagenes: pd.DataFrame, caracteristicas: dict) -> dict:
    """Confianza, métricas por estilo y por dispositivo, y lista de errores en el conjunto de prueba."""
    prueba = imagenes[imagenes["conjunto"] == "prueba"].reset_index(drop=True)
    probabilidades = capa_softmax.predecir_probabilidades(caracteristicas["prueba"])
    predichas = capa_softmax.clases[probabilidades.argmax(axis=1)]
    confianza = probabilidades.max(axis=1)  # probabilidad de la clase elegida
    es_correcta = predichas == prueba["clase"].to_numpy()
    return {
        "confianza": confianza, "es_correcta": es_correcta,
        "por_estilo": {estilo: metricas_de(grupo["clase"], predichas[grupo.index])
                       for estilo, grupo in prueba.groupby("estilo")},
        "por_dispositivo": {dispositivo: metricas_de(grupo["clase"], predichas[grupo.index])
                            for dispositivo, grupo in prueba.groupby("dispositivo")},
        "errores": [f"  {ruta} | real={real} | predicha={predicha} | prob={probabilidad:.2f}"
                    for ruta, real, predicha, probabilidad, correcta
                    in zip(prueba["ruta"], prueba["clase"], predichas, confianza, es_correcta) if not correcta],
    }


def crear_imagenes_que_no_son_pantallas() -> dict[str, Image.Image]:
    """Tres imágenes que no son pantallas: ruido aleatorio, una imagen negra y un degradado."""
    aleatorio = np.random.default_rng(SEMILLA)
    degradado = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (600, 1))
    return {
        "ruido aleatorio": Image.fromarray(aleatorio.integers(0, 255, (600, 400, 3), dtype=np.uint8)),
        "imagen negra": Image.new("RGB", (400, 600), "black"),
        "degradado": Image.fromarray(np.stack([degradado] * 3, axis=-1)),
    }


def probar_imagenes_que_no_son_pantallas(capa_softmax: CapaSoftmax) -> list[str]:
    """Clasifica las imágenes que no son pantallas para medir la limitación: siempre se elige un tipo."""
    imagenes = crear_imagenes_que_no_son_pantallas()
    probabilidades = capa_softmax.predecir_probabilidades(ExtractorResNet50().extraer(list(imagenes.values())))
    return [f"  {nombre:<16} -> {capa_softmax.clases[fila.argmax()]} (confianza {fila.max():.2f})"
            for nombre, fila in zip(imagenes, probabilidades)]


# --------------------------------------------------------------------------
# Evidencias
# --------------------------------------------------------------------------
def redactar_evidencia(imagenes: pd.DataFrame, evaluacion: dict) -> list[str]:
    """Líneas de evidencias/evaluacion_pantallas.txt: conjunto, selección de L2, umbrales, métricas y errores."""
    resultados, prueba = evaluacion["resultados"], evaluacion["prueba"]
    confianza, es_correcta = prueba["confianza"], prueba["es_correcta"]
    conteo = pd.crosstab(imagenes["clase"], imagenes["conjunto"]).reindex(TIPOS_DE_PANTALLA)
    ancho = ANCHO_DEL_NOMBRE
    return [
        "===== Conjunto de imágenes =====", conteo.to_string(),
        f"Entrenamiento con aumento: {evaluacion['ejemplos_entrenamiento']} ejemplos (original + 1 copia aumentada)",
        f"Capa Dense softmax: {EPOCAS} épocas, lotes de {TAMANO_LOTE}, Adam con tasa de aprendizaje {TASA_APRENDIZAJE}",
        "", "===== Selección de la regularización L2 (validación, F1 macro) =====",
        *[f"  L2={l2:<7} accuracy={m['accuracy']:.3f} f1={m['f1_macro']:.3f}"
          for l2, m in evaluacion["seleccion_l2"].items()],
        f"  Elegido: L2={evaluacion['mejor_l2']}",
        "", "===== Umbral de confianza para la integración (validación) =====", *evaluacion["umbrales"],
        "", "===== Resumen (promedios macro sobre las 8 clases) =====", encabezado_de_metricas(ancho),
        *[fila_de_metricas(f"ResNet50 + softmax - {conjunto}", resultados[conjunto], ancho) for conjunto in CONJUNTOS],
        *[fila_de_metricas(f"  prueba, estilo {estilo}", m, ancho) for estilo, m in prueba["por_estilo"].items()],
        *[fila_de_metricas(f"  prueba, {dispositivo}", m, ancho) for dispositivo, m in prueba["por_dispositivo"].items()],
        fila_de_metricas("Sin transferencia (píxeles) - prueba", resultados["sin_transferencia_prueba"], ancho),
        fila_de_metricas("Sin transferencia (píxeles) - estilo nuevo", resultados["sin_transferencia_estilo_nuevo"], ancho),
        "", "===== Métricas por clase: prueba =====", evaluacion["reportes_por_tipo"]["prueba"],
        "===== Métricas por clase: estilo nuevo (bocetos) =====", evaluacion["reportes_por_tipo"]["estilo_nuevo"],
        "===== Confianza (probabilidad de la clase elegida) en la prueba =====",
        f"Aciertos: mediana {np.median(confianza[es_correcta]):.3f}, mínimo {confianza[es_correcta].min():.3f}",
        f"Errores:  mediana {np.median(confianza[~es_correcta]):.3f}, máximo {confianza[~es_correcta].max():.3f}",
        "", f"===== Errores en la prueba ({len(prueba['errores'])}) =====", *prueba["errores"],
        "", "===== Imágenes que no son pantallas (limitación: siempre se elige uno de los 8 tipos) =====",
        *evaluacion["no_pantallas"],
    ]


def guardar_evidencias(imagenes: pd.DataFrame, evaluacion: dict) -> None:
    """Curva de entrenamiento, resultados en JSON y resumen en texto (las matrices se guardan al evaluar)."""
    import keras

    capa_softmax, mejor_l2, prueba = evaluacion["capa_softmax"], evaluacion["mejor_l2"], evaluacion["prueba"]
    guardar_curva_entrenamiento("curva_entrenamiento_pantallas.png", capa_softmax.historial,
                                f"Capa Dense softmax sobre ResNet50 (L2 = {mejor_l2})")
    guardar_json("evaluacion_pantallas.json",
                 {"implementacion": f"Keras {keras.__version__} ({keras.backend.backend()})",
                  "L2": mejor_l2, "seleccion_L2": {str(l2): m for l2, m in evaluacion["seleccion_l2"].items()},
                  "epocas": EPOCAS, "historial": capa_softmax.historial, "clases": TIPOS_DE_PANTALLA,
                  "prueba_por_estilo": prueba["por_estilo"], "prueba_por_dispositivo": prueba["por_dispositivo"],
                  **evaluacion["resultados"]})
    guardar_texto("evaluacion_pantallas.txt", redactar_evidencia(imagenes, evaluacion), mostrar=True)


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def evaluar(imagenes: pd.DataFrame) -> dict:
    """Entrena la capa softmax y reúne todos los resultados de la evaluación."""
    caracteristicas = cargar_caracteristicas(imagenes)  # ResNet50: 2048 valores por imagen
    tipos_reales = {conjunto: grupo["clase"].to_numpy() for conjunto, grupo in imagenes.groupby("conjunto")}

    # 1. Entrenamiento (L2 elegido con validación) y umbral de confianza para la integración
    capa_softmax, mejor_l2, seleccion_l2, ejemplos_entrenamiento = entrenar_capa_softmax(caracteristicas, tipos_reales)
    umbrales = revisar_umbrales(capa_softmax, caracteristicas["validacion"], tipos_reales["validacion"])

    # 2. Evaluación por conjunto, comparación sin transferencia, análisis de la prueba y limitación
    resultados, reportes_por_tipo = evaluar_por_conjunto(capa_softmax, caracteristicas, tipos_reales)
    resultados.update(comparar_sin_transferencia(imagenes, tipos_reales, mejor_l2))
    return {
        "capa_softmax": capa_softmax, "mejor_l2": mejor_l2, "seleccion_l2": seleccion_l2,
        "ejemplos_entrenamiento": ejemplos_entrenamiento, "umbrales": umbrales,
        "resultados": resultados, "reportes_por_tipo": reportes_por_tipo,
        "prueba": analizar_prueba(capa_softmax, imagenes, caracteristicas),
        "no_pantallas": probar_imagenes_que_no_son_pantallas(capa_softmax),
    }


def main() -> None:
    """Entrena la capa softmax, la evalúa en los cuatro conjuntos y guarda las evidencias."""
    imagenes = pd.read_csv(CARPETA_IMAGENES / "imagenes.csv")
    guardar_evidencias(imagenes, evaluar(imagenes))


if __name__ == "__main__":
    main()
