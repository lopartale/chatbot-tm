"""
Conjunto de imágenes: pantallas sintéticas de banca digital (Avance 2)

Genera capturas de 8 tipos de pantalla; cada tipo se relaciona con un dominio del RAML:

    login          -> autenticacion        prestamo       -> prestamos
    cuentas        -> cuentas              seguros        -> seguros
    transferencia  -> transferencias       documentos     -> documentos / hipotecas
    pago_servicios -> pagos                alta_cliente   -> clientes

Cada pantalla se arma en HTML (plantillas_pantallas.py) y se captura con Chromium
(Playwright). Para que el modelo aprenda el tipo de pantalla y no un diseño fijo, varían
banco, colores, modo claro/oscuro, tipografía, dispositivo, navegación, campos y datos.

Estilos: "alta" (diseño final) y "front" (maqueta en grises) para entrenamiento,
validación y prueba; "boceto" sólo forma el conjunto "estilo_nuevo", que el modelo nunca
ve al entrenar y mide si generaliza.

    python -m vision.generar_pantallas     # data/imagenes/<conjunto>/<tipo>/<tipo>_###.jpg + imagenes.csv
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from configuracion.rutas import CARPETA_IMAGENES
from vision.plantillas_pantallas import GeneradorAleatorio, armar_pagina_html
from evaluacion.registro_evidencias import guardar_imagen

SEMILLA = 2026

# Tipo de pantalla -> dominio del inventario RAML
DOMINIO_POR_PANTALLA = {
    "login": "autenticacion", "cuentas": "cuentas", "transferencia": "transferencias",
    "pago_servicios": "pagos", "prestamo": "prestamos", "seguros": "seguros",
    "documentos": "hipotecas", "alta_cliente": "clientes",
}  # documentos: carga de documentos de una solicitud de hipoteca (/hipotecas/{idHipoteca}/documentos)
TIPOS_DE_PANTALLA = list(DOMINIO_POR_PANTALLA)

# Imágenes por tipo de pantalla en cada conjunto (70/15/15 más el estilo nuevo)
IMAGENES_POR_TIPO = {"entrenamiento": 105, "validacion": 22, "prueba": 23, "estilo_nuevo": 20}

# (nombre, ancho, alto) en píxeles
DISPOSITIVOS = [("movil", 390, 844), ("movil", 360, 780), ("tableta", 820, 1180),
                ("escritorio", 1366, 768), ("escritorio", 1280, 800), ("escritorio", 1440, 900)]
LADO_MAXIMO = 512  # las capturas se guardan en JPEG con el lado mayor de 512 px
COLUMNAS_CSV = ["ruta", "conjunto", "clase", "dominio", "estilo", "dispositivo", "ancho", "alto", "semilla"]


def planear_imagenes() -> list[dict]:
    """Lista de imágenes a generar: conjunto, tipo de pantalla, estilo, semilla y ruta de cada una."""
    imagenes = []
    for conjunto, cantidad in IMAGENES_POR_TIPO.items():
        for tipo in TIPOS_DE_PANTALLA:
            for numero in range(cantidad):
                if conjunto == "estilo_nuevo":
                    estilo = "boceto"
                else:
                    estilo = "front" if numero % 3 == 0 else "alta"  # una de cada tres es maqueta
                imagenes.append({
                    "conjunto": conjunto, "clase": tipo, "dominio": DOMINIO_POR_PANTALLA[tipo], "estilo": estilo,
                    "semilla": SEMILLA * 100000 + len(imagenes),
                    "ruta": f"{conjunto}/{tipo}/{tipo}_{numero:03d}.jpg",
                })
    return imagenes


def generar_imagenes(carpeta: Path = CARPETA_IMAGENES) -> list[dict]:
    """Captura cada pantalla planeada y guarda la lista en imagenes.csv."""
    from io import BytesIO

    from PIL import Image
    from playwright.sync_api import sync_playwright

    imagenes = planear_imagenes()
    with sync_playwright() as playwright:
        navegador = playwright.chromium.launch()
        pestana_por_tamano = {}
        for imagen in imagenes:
            aleatorio = random.Random(imagen["semilla"])  # semilla propia: cada imagen es reproducible
            dispositivo, ancho, alto = aleatorio.choice(DISPOSITIVOS)
            # Una pestaña del navegador por tamaño de pantalla, reutilizada entre imágenes
            if (ancho, alto) not in pestana_por_tamano:
                pestana_por_tamano[(ancho, alto)] = navegador.new_page(viewport={"width": ancho, "height": alto})
            pestana = pestana_por_tamano[(ancho, alto)]

            azar = GeneradorAleatorio(aleatorio, imagen["estilo"])
            pestana.set_content(armar_pagina_html(azar, imagen["clase"], dispositivo))  # HTML -> página
            captura = Image.open(BytesIO(pestana.screenshot())).convert("RGB")        # página -> imagen
            captura.thumbnail((LADO_MAXIMO, LADO_MAXIMO))

            ruta = carpeta / imagen["ruta"]
            ruta.parent.mkdir(parents=True, exist_ok=True)
            captura.save(ruta, quality=aleatorio.randint(80, 92))  # calidad JPEG variable, como capturas reales
            imagen.update(dispositivo=dispositivo, ancho=ancho, alto=alto)
        navegador.close()

    with open(carpeta / "imagenes.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CSV)
        escritor.writeheader()
        escritor.writerows(imagenes)
    return imagenes


def crear_hoja_de_muestra(celda: int = 260) -> Path:
    """Arma la muestra de pantallas y la guarda como evidencia (evidencias/muestra_pantallas.png)."""
    return guardar_imagen("muestra_pantallas.png", armar_hoja_de_muestra(celda))


def armar_hoja_de_muestra(celda: int = 260):
    """Una imagen por tipo de pantalla (columnas) y estilo (filas: alta, front, boceto) para el reporte."""
    from PIL import Image, ImageDraw

    filas = [("entrenamiento", "alta"), ("entrenamiento", "front"), ("estilo_nuevo", "boceto")]
    alto_encabezado = 24
    hoja = Image.new("RGB", (celda * len(TIPOS_DE_PANTALLA), alto_encabezado + celda * len(filas)), "white")
    dibujo = ImageDraw.Draw(hoja)
    with open(CARPETA_IMAGENES / "imagenes.csv", encoding="utf-8") as archivo:
        imagenes = list(csv.DictReader(archivo))

    for columna, tipo in enumerate(TIPOS_DE_PANTALLA):
        dibujo.text((columna * celda + 8, 6), tipo, fill="black")
        for fila, (conjunto, estilo) in enumerate(filas):
            primera = next(imagen for imagen in imagenes
                           if imagen["clase"] == tipo and imagen["conjunto"] == conjunto and imagen["estilo"] == estilo)
            miniatura = Image.open(CARPETA_IMAGENES / primera["ruta"])
            miniatura.thumbnail((celda - 8, celda - 8))
            x = columna * celda + (celda - miniatura.width) // 2
            y = alto_encabezado + fila * celda + (celda - miniatura.height) // 2
            hoja.paste(miniatura, (x, y))
            dibujo.rectangle([x, y, x + miniatura.width - 1, y + miniatura.height - 1], outline="#999")
    return hoja


if __name__ == "__main__":
    import time
    from collections import Counter

    inicio = time.time()
    generadas = generar_imagenes()
    crear_hoja_de_muestra()
    print(f"{len(generadas)} imágenes en {time.time() - inicio:.0f} s")
    print(Counter((imagen["conjunto"], imagen["estilo"]) for imagen in generadas))
    print(Counter(imagen["dispositivo"] for imagen in generadas))
