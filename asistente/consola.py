"""
Modo interactivo (terminal o Colab)

Lee los mensajes del usuario, se los pasa al asistente y muestra sus respuestas. Para
analizar una pantalla se usa el comando "/imagen <ruta> [mensaje]".

Sólo se encarga de la entrada y la salida; el flujo de cada turno está en chatbot_servicios.py.
"""
from __future__ import annotations

import re
from pathlib import Path

from lenguaje.preprocesamiento import normalizar

PALABRAS_SALIDA = {"salir", "adios", "terminar"}

INSTRUCCIONES = "Escribe 'salir' para terminar. Para analizar una pantalla: /imagen <ruta del archivo>"

# "/imagen <ruta> [mensaje]". La ruta puede ir entre comillas o, sin comillas, contener espacios
# si termina en una extensión de imagen: /imagen Mis capturas/login 1.png ¿qué servicios usa?
PATRON_COMANDO_IMAGEN = re.compile(
    r'^/imagen\s+(?:"(?P<entre_comillas>[^"]+)"'                              # "ruta con espacios.png"
    r'|(?P<con_extension>.+?\.(?:png|jpe?g|webp|bmp|gif))(?=\s|$)'           # ruta con espacios.png
    r'|(?P<sin_espacios>\S+))'                                                # cualquier ruta sin espacios
    r'\s*(?P<mensaje>.*)$', re.IGNORECASE | re.DOTALL)


def separar_comando_imagen(entrada: str) -> tuple[Path | None, str]:
    """Comando de la terminal "/imagen <ruta> [mensaje]" -> (ruta de la imagen, mensaje)."""
    coincidencia = PATRON_COMANDO_IMAGEN.match(entrada.strip())
    if not coincidencia:  # no es el comando, o es "/imagen" sin ruta
        return None, entrada
    ruta = (coincidencia["entre_comillas"] or coincidencia["con_extension"] or coincidencia["sin_espacios"])
    return Path(ruta).expanduser(), coincidencia["mensaje"]


def es_despedida(entrada: str) -> bool:
    """¿El usuario quiere terminar? ("salir", "Adiós!", "terminar.")"""
    return normalizar(entrada).strip(" !.,") in PALABRAS_SALIDA


def ejecutar_consola(asistente, leer=input, escribir=print) -> None:
    """Ciclo de conversación: leer mensaje -> responder -> mostrar, hasta una despedida o fin de la entrada."""
    escribir(f"Asistente de servicios REST - inventario: {asistente.inventario.titulo} "
             f"({len(asistente.inventario.servicios)} servicios, RAML)")
    escribir(INSTRUCCIONES)
    while True:
        try:
            entrada = leer("Tú: ")
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C o fin de la entrada
            escribir()
            break
        ruta_imagen, texto = separar_comando_imagen(entrada)  # "/imagen captura.png ¿qué servicios usa?"
        escribir("Bot:", asistente.responder(texto, imagen=ruta_imagen))
        if es_despedida(entrada):  # se responde la despedida y después se termina
            break
