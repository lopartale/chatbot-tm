"""
Integración multimodal: de una captura de pantalla a los servicios REST que la cubren (Avance 2)

    imagen -> clasificador visual (ResNet50 + Dense softmax, Keras) -> tipo de pantalla y confianza
           -> patrón de pantalla (operaciones típicas) -> inventario RAML -> cobertura y brechas

El clasificador visual sólo reconoce el TIPO de pantalla; los patrones de pantalla
(patrones_pantalla.py) traducen ese tipo en las operaciones que necesita, y cada operación
se busca en el inventario por método y endpoint exactos; si no está, se reporta como brecha.

Umbral de confianza: si la probabilidad de la clase elegida es menor que UMBRAL_DE_CONFIANZA,
la respuesta muestra también las alternativas y pide confirmación. El valor 0.8 se eligió
con el conjunto de VALIDACIÓN de imágenes (evaluar_pantallas.py): marca 5 de los 7 errores a
cambio de pedir confirmación en 7 de los 169 aciertos (a 0.7 marcaba sólo 3 errores). Una
confirmación innecesaria cuesta menos que recomendar los servicios de otra pantalla.
"""
from __future__ import annotations

from pathlib import Path

from multimodal.patrones_pantalla import NOMBRE_DE_PANTALLA, buscar_operaciones_en_inventario

UMBRAL_DE_CONFIANZA = 0.8

# Avisos cuando la imagen no se puede analizar
MENSAJE_IMAGEN_NO_ENCONTRADA = "No encontré la imagen ({}). Revisa la ruta."
MENSAJE_NO_ES_IMAGEN = ("El archivo {} no es una imagen que pueda abrir. Envía una captura en PNG o JPG, "
                        "o describe en texto los campos y acciones de la pantalla.")
MENSAJE_RESNET50_NO_DISPONIBLE = ("No pude descargar ResNet50, el modelo que analiza las imágenes (¿hay conexión a "
                                  "internet?). Ejecuta 'python -m vision.preparar_resnet50' con conexión y vuelve a "
                                  "enviar la imagen; mientras tanto puedes describir la pantalla en texto.")
MENSAJE_MODELO_VISUAL_NO_DISPONIBLE = ("El clasificador de pantallas no está entrenado, falta instalar Keras o el "
                                       "modelo no se pudo abrir. Ejecuta "
                                       "'python -m evaluacion.evaluar_pantallas' y vuelve a enviar la imagen; mientras tanto "
                                       "puedes describir la pantalla en texto.")


class AnalizadorPantallas:
    """Componente visual del asistente; carga el clasificador la primera vez que llega una imagen
    (Keras y ResNet50 tardan en cargar y no hacen falta en las conversaciones de sólo texto)."""

    def __init__(self, ruta_modelo: str | Path | None = None):
        """El clasificador no se carga aquí, sino con la primera imagen de la sesión."""
        self.ruta_modelo = ruta_modelo  # None: el modelo del proyecto (modelos/clasificador_pantallas.keras)
        self.clasificador = None

    def clasificar(self, imagen) -> dict:
        """Tipo de pantalla de la imagen (ver ClasificadorPantallas.clasificar)."""
        if self.clasificador is None:  # primera imagen de la sesión: se carga ResNet50
            from vision.clasificador_visual import RUTA_MODELO_PANTALLAS, ClasificadorPantallas
            self.clasificador = ClasificadorPantallas(self.ruta_modelo or RUTA_MODELO_PANTALLAS)
        return self.clasificador.clasificar(imagen)

    def analizar(self, imagen) -> tuple[dict | None, str | None]:
        """(pantalla, None) si la imagen se pudo clasificar, o (None, aviso para el usuario) si no.

        Primero se abre la imagen y después se carga el modelo: una ruta equivocada se avisa al
        momento, sin esperar a que carguen Keras y ResNet50.
        """
        from PIL import Image, UnidentifiedImageError

        from vision.clasificador_visual import ModeloVisualNoDisponible
        from vision.preparar_resnet50 import ResNet50NoDisponible

        nombre = nombre_de_imagen(imagen)
        try:
            if not isinstance(imagen, Image.Image):
                imagen = Image.open(imagen)  # valida que exista y que sea una imagen
            return self.clasificar(imagen), None
        except ModeloVisualNoDisponible:   # falta entrenar la capa softmax
            return None, MENSAJE_MODELO_VISUAL_NO_DISPONIBLE
        except ResNet50NoDisponible:       # primera imagen sin ResNet50 descargado y sin conexión
            return None, MENSAJE_RESNET50_NO_DISPONIBLE
        except FileNotFoundError:          # la ruta no existe
            return None, MENSAJE_IMAGEN_NO_ENCONTRADA.format(nombre)
        except (UnidentifiedImageError, IsADirectoryError, PermissionError):
            # el archivo existe pero no es una imagen, es una carpeta o no se puede leer
            return None, MENSAJE_NO_ES_IMAGEN.format(nombre)


def nombre_de_imagen(imagen) -> str:
    """Ruta de la imagen para el historial y los avisos ("imagen" si llegó como PIL.Image)."""
    return str(imagen) if isinstance(imagen, (str, Path)) else "imagen"


def redactar_respuesta_de_pantalla(resultado: dict, servicios: list[dict]) -> tuple[str, list[str]]:
    """Texto de la respuesta y los IDs recomendados (contexto para "detalle del primero").

    Partes: tipo de pantalla y confianza; alternativas si la confianza es baja; operaciones del
    patrón con el servicio que las cubre o la brecha; aclaración de que la imagen no se lee campo por campo.
    """
    tipo_pantalla = resultado["clase"]
    lineas = [_linea_de_clasificacion(resultado)]
    if resultado["probabilidad"] < UMBRAL_DE_CONFIANZA:
        lineas.append(_linea_de_alternativas(resultado["alternativas"]))
    lineas_de_operaciones, recomendados = _lineas_de_operaciones(tipo_pantalla, servicios)
    lineas += lineas_de_operaciones
    lineas.append("Estas operaciones son las habituales para este tipo de pantalla; la imagen no se lee "
                  "campo por campo. Si tu pantalla tiene funciones adicionales, descríbelas en texto.")
    return "\n".join(lineas), recomendados


def _linea_de_clasificacion(resultado: dict) -> str:
    """"La imagen parece una pantalla de transferencia (confianza 97%; dominio transferencias)."."""
    return (f"La imagen parece una pantalla de {NOMBRE_DE_PANTALLA[resultado['clase']]} "
            f"(confianza {resultado['probabilidad']:.0%}; dominio {resultado['dominio']}).")


def _linea_de_alternativas(alternativas: list[tuple[str, float]]) -> str:
    """Confianza baja: las otras dos clases más probables, para que el usuario confirme."""
    texto = ", ".join(f"{NOMBRE_DE_PANTALLA[tipo]} ({probabilidad:.0%})" for tipo, probabilidad in alternativas)
    return (f"No estoy seguro: también podría ser {texto}. Si es otra pantalla, "
            "descríbeme sus funciones en texto y la analizo.")


def _lineas_de_operaciones(tipo_pantalla: str, servicios: list[dict]) -> tuple[list[str], list[str]]:
    """Patrón de pantalla -> servicios del inventario: cobertura, una línea por operación y los IDs recomendados."""
    operaciones = buscar_operaciones_en_inventario(tipo_pantalla, servicios)
    cubiertas = sum(1 for _, servicio in operaciones if servicio)
    lineas = [f"Operaciones típicas de esta pantalla: {len(operaciones)}. "
              f"Cobertura en el inventario: {round(100 * cubiertas / len(operaciones))}%."]
    recomendados = []
    for descripcion, servicio in operaciones:
        if servicio is None:
            lineas.append(f"  [BRECHA] {descripcion} -> sin servicio en el inventario")
            continue
        lineas.append(f"  [OK] {descripcion} -> {servicio['id']} {servicio['metodo']} {servicio['endpoint']}")
        if servicio["id"] not in recomendados:  # un servicio puede cubrir varias operaciones
            recomendados.append(servicio["id"])
    return lineas, recomendados
