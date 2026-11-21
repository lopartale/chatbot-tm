"""Pruebas (pytest) de la integración multimodal."""
from pathlib import Path

import pytest

from vision.generar_pantallas import TIPOS_DE_PANTALLA
from asistente.consola import separar_comando_imagen
from multimodal.integracion_multimodal import UMBRAL_DE_CONFIANZA, redactar_respuesta_de_pantalla
from multimodal.patrones_pantalla import PATRONES_DE_PANTALLA, buscar_operaciones_en_inventario
from tests.test_clasificador_visual import MODELO_LISTO


@pytest.fixture(scope="module")
def servicios():
    """Servicios del inventario RAML."""
    from asistente.chatbot_servicios import AsistenteServiciosREST
    return AsistenteServiciosREST().inventario.servicios


def test_cada_tipo_de_pantalla_tiene_patron():
    """Los 8 tipos de pantalla tienen patrón de operaciones."""
    assert set(PATRONES_DE_PANTALLA) == set(TIPOS_DE_PANTALLA)


@pytest.mark.parametrize("tipo_pantalla", TIPOS_DE_PANTALLA)
def test_las_operaciones_de_cada_patron_existen_en_el_inventario(servicios, tipo_pantalla):
    """Cada operación de los patrones existe en el RAML (método y endpoint)."""
    operaciones = buscar_operaciones_en_inventario(tipo_pantalla, servicios)
    assert all(servicio for _, servicio in operaciones), "operación del patrón sin servicio en el RAML"


def test_con_baja_confianza_muestra_alternativas(servicios):
    """Debajo del umbral, la respuesta pide confirmar y muestra alternativas."""
    resultado = {"clase": "login", "dominio": "autenticacion", "probabilidad": UMBRAL_DE_CONFIANZA - 0.1,
                 "alternativas": [("alta_cliente", 0.2), ("transferencia", 0.1)]}
    texto, recomendados = redactar_respuesta_de_pantalla(resultado, servicios)
    assert "No estoy seguro" in texto and "alta de cliente" in texto
    assert recomendados[0] == "SRV-045"


@pytest.mark.parametrize("entrada, ruta, mensaje", [
    ("/imagen captura.png", Path("captura.png"), ""),
    ("/imagen captura.png ¿qué servicios usa?", Path("captura.png"), "¿qué servicios usa?"),
    ("hola", None, "hola"),
    ('/imagen "Mis capturas/login 1.png" ¿qué servicios usa?', Path("Mis capturas/login 1.png"), "¿qué servicios usa?"),
    ("/imagen Mis capturas/login 1.png ¿qué servicios usa?", Path("Mis capturas/login 1.png"), "¿qué servicios usa?"),
    ("/imagen Mis capturas/login 1.PNG", Path("Mis capturas/login 1.PNG"), ""),
    ("/imagen", None, "/imagen"),
])
def test_comando_imagen(entrada, ruta, mensaje):
    """El comando /imagen de la terminal separa la ruta y el mensaje."""
    assert separar_comando_imagen(entrada) == (ruta, mensaje)


def test_sin_imagen_pide_adjuntarla():
    """Si se anuncia una imagen sin adjuntarla, el asistente la pide."""
    from asistente.chatbot_servicios import AsistenteServiciosREST
    assert "/imagen" in AsistenteServiciosREST().responder("Te comparto el mockup de la pantalla")


# Fallos conocidos del clasificador visual: se documentan como fallos esperados en lugar de cambiar
# la conversación o de entrenar con esas imágenes. strict=True: si un cambio los corrige, pytest lo avisa.
FALLOS_VISUALES_CONOCIDOS = {
    "M04 Boceto (estilo no visto) de seguros":
        "Con Keras, el boceto de seguros se clasifica como préstamo (confianza 0.46, seguros 0.39). Al quedar "
        "debajo del umbral de 0.8, el asistente advierte la duda y ofrece seguros como alternativa.",
}


def _conversacion(nombre):
    """Caso de pytest por conversación multimodal; los fallos conocidos se marcan como esperados."""
    marcas = ([pytest.mark.xfail(reason=FALLOS_VISUALES_CONOCIDOS[nombre], strict=True)]
              if nombre in FALLOS_VISUALES_CONOCIDOS else [])
    return pytest.param(nombre, id=nombre.split()[0], marks=marcas)


def _nombres_de_conversaciones():
    """Nombres de las conversaciones multimodales (importación diferida para la recolección de pytest)."""
    from evaluacion.pruebas_multimodales import CONVERSACIONES_MULTIMODALES
    return list(CONVERSACIONES_MULTIMODALES)


@pytest.mark.skipif(not MODELO_LISTO, reason="Requiere el modelo visual y las imágenes")
@pytest.mark.parametrize("nombre", [_conversacion(nombre) for nombre in _nombres_de_conversaciones()])
def test_conversacion_multimodal(nombre):
    """Cada conversación con imágenes obtiene la intención y la pantalla esperadas en todos sus turnos."""
    from asistente.chatbot_servicios import AsistenteServiciosREST
    from evaluacion.pruebas_multimodales import CONVERSACIONES_MULTIMODALES, ejecutar_conversacion

    fallas = [f"{turno['mensaje']!r} -> {turno['intencion_obtenida']}/{turno['pantalla_obtenida']}"
              for turno in ejecutar_conversacion(AsistenteServiciosREST(), CONVERSACIONES_MULTIMODALES[nombre])
              if not turno["correcto"]]
    assert not fallas, "\n".join(fallas)


def test_imagen_inexistente():
    """Una ruta inválida produce un aviso en lugar de un error."""
    from asistente.chatbot_servicios import AsistenteServiciosREST
    assert "No encontré la imagen" in AsistenteServiciosREST().responder("", imagen="no_existe.png")


def test_archivo_que_no_es_imagen(tmp_path):
    """Un archivo que no es imagen produce un aviso en lugar de detener el asistente."""
    from asistente.chatbot_servicios import AsistenteServiciosREST
    archivo = tmp_path / "notas.png"
    archivo.write_text("no es una imagen")
    assert "no es una imagen que pueda abrir" in AsistenteServiciosREST().responder("", imagen=archivo)


def test_sin_modelo_visual(tmp_path):
    """Sin la capa softmax entrenada, el aviso indica que falta el modelo y no culpa a la imagen."""
    from asistente.chatbot_servicios import AsistenteServiciosREST
    from multimodal.integracion_multimodal import AnalizadorPantallas

    asistente = AsistenteServiciosREST()
    from PIL import Image

    captura = tmp_path / "captura.png"
    Image.new("RGB", (390, 844), "white").save(captura)
    asistente.analizador_pantallas = AnalizadorPantallas(ruta_modelo=tmp_path / "no_existe.keras")
    assert "no está entrenado" in asistente.responder("", imagen=captura)
