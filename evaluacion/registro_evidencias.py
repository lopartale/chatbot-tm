"""
Registro de evidencias

Único componente que escribe en la carpeta evidencias/. Los demás módulos calculan
resultados (métricas, transcripciones, matrices) y se los entregan aquí para guardarlos:

    guardar_texto             transcripciones y reportes de métricas (.txt)
    guardar_json              resultados completos para revisarlos o reutilizarlos (.json)
    guardar_matriz_confusion  matrices de confusión dibujadas con matplotlib (.png)
    guardar_imagen            figuras ya armadas, como la muestra de pantallas (.png)

Así, el formato de los archivos y la ubicación de la carpeta se definen en un solo lugar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from configuracion.rutas import CARPETA_EVIDENCIAS



def ruta_de_evidencia(nombre_archivo: str) -> Path:
    """Ruta del archivo dentro de evidencias/; crea la carpeta si todavía no existe."""
    CARPETA_EVIDENCIAS.mkdir(exist_ok=True)
    return CARPETA_EVIDENCIAS / nombre_archivo


def guardar_texto(nombre_archivo: str, lineas: list[str], mostrar: bool = False) -> Path:
    """Escribe las líneas en un .txt y, si se pide, también las muestra (salida del notebook)."""
    texto = "\n".join(lineas)
    ruta = ruta_de_evidencia(nombre_archivo)
    ruta.write_text(texto, encoding="utf-8")
    if mostrar:
        print(texto)
    return ruta


def guardar_json(nombre_archivo: str, datos) -> Path:
    """Escribe los datos como JSON legible (con acentos y sangría)."""
    ruta = ruta_de_evidencia(nombre_archivo)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def guardar_imagen(nombre_archivo: str, imagen) -> Path:
    """Guarda una imagen de PIL ya armada."""
    ruta = ruta_de_evidencia(nombre_archivo)
    imagen.save(ruta)
    return ruta


def guardar_matriz_confusion(nombre_archivo: str, matriz: np.ndarray, etiquetas: list[str], titulo: str,
                             nombre_eje: str = "Clase", tamano: tuple[float, float] = (8, 6.8),
                             tamano_titulo: int = 10) -> Path:
    """Matriz de confusión con el número de casos en cada celda (filas: real, columnas: predicha)."""
    import matplotlib
    matplotlib.use("Agg")  # dibuja en archivo, sin abrir ventanas
    import matplotlib.pyplot as plt

    figura, eje = plt.subplots(figsize=tamano)
    imagen = eje.imshow(matriz, cmap="Blues")  # más casos, azul más intenso
    posiciones = range(len(etiquetas))
    eje.set_xticks(posiciones, etiquetas, rotation=45, ha="right", fontsize=8)
    eje.set_yticks(posiciones, etiquetas, fontsize=8)
    eje.set_xlabel(f"{nombre_eje} predicha")
    eje.set_ylabel(f"{nombre_eje} real")
    eje.set_title(titulo, fontsize=tamano_titulo)
    for fila in posiciones:
        for columna in posiciones:
            casos = matriz[fila, columna]
            if casos:  # las celdas en cero se dejan vacías para que resalten los errores
                color = "white" if casos > matriz.max() / 2 else "black"
                eje.text(columna, fila, casos, ha="center", va="center", color=color, fontsize=9)
    figura.colorbar(imagen, ax=eje, fraction=0.046)
    figura.tight_layout()
    ruta = ruta_de_evidencia(nombre_archivo)
    figura.savefig(ruta, dpi=200)
    plt.close(figura)
    return ruta


def guardar_curva_entrenamiento(nombre_archivo: str, historial: dict[str, list[float]], titulo: str) -> Path:
    """Pérdida y exactitud de entrenamiento en cada época (historial de model.fit de Keras)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, (eje_perdida, eje_exactitud) = plt.subplots(1, 2, figsize=(9, 3.4))
    epocas = range(1, len(historial["loss"]) + 1)
    eje_perdida.plot(epocas, historial["loss"], color="#1f4e79")
    eje_perdida.set(xlabel="Época", ylabel="Pérdida (entropía cruzada)", title="Pérdida")
    eje_exactitud.plot(epocas, historial["accuracy"], color="#1f4e79")
    eje_exactitud.set(xlabel="Época", ylabel="Exactitud", title="Exactitud en entrenamiento", ylim=(0, 1.02))
    figura.suptitle(titulo, fontsize=10)
    figura.tight_layout()
    ruta = ruta_de_evidencia(nombre_archivo)
    figura.savefig(ruta, dpi=150)
    plt.close(figura)
    return ruta
