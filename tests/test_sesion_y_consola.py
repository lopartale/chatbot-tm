"""Pruebas (pytest) de la sesión de conversación y del modo interactivo."""
from asistente.consola import ejecutar_consola, es_despedida
from asistente.sesion_conversacion import SesionConversacion


def test_sesion_conserva_el_contexto_si_no_hay_recomendaciones():
    """Una respuesta sin servicios (None) no borra los últimos recomendados."""
    sesion = SesionConversacion()
    sesion.recordar_recomendados(["SRV-013", "SRV-025"])
    sesion.recordar_recomendados(None)
    assert sesion.ultimos_recomendados == ["SRV-013", "SRV-025"]
    sesion.recordar_recomendados([])
    assert sesion.ultimos_recomendados == []


def test_sesion_registra_el_turno_y_devuelve_la_respuesta():
    """registrar_turno guarda el turno con sus detalles y devuelve la respuesta."""
    sesion = SesionConversacion()
    assert sesion.registrar_turno("hola", "charla", "¡Hola!", imagen="x.png") == "¡Hola!"
    assert sesion.historial == [{"usuario": "hola", "intencion": "charla", "bot": "¡Hola!", "imagen": "x.png"}]


def test_despedidas():
    """Las palabras de salida se reconocen sin importar acentos, mayúsculas o signos."""
    assert es_despedida("Adiós!") and es_despedida("salir") and es_despedida(" terminar.")
    assert not es_despedida("adiós a los formularios en papel")


def test_consola_responde_y_termina_con_la_despedida():
    """La consola pasa cada mensaje al asistente, responde la despedida y termina."""
    class AsistenteDePrueba:
        """Asistente mínimo que registra lo que recibe la consola."""
        class inventario:
            titulo, servicios = "Inventario de prueba", []

        def __init__(self):
            self.recibidos = []

        def responder(self, mensaje, imagen=None):
            self.recibidos.append((mensaje, imagen))
            return "ok"

    asistente, salida = AsistenteDePrueba(), []
    entradas = iter(["hola", "/imagen captura.png ¿qué servicios usa?", "salir", "no se lee"])
    ejecutar_consola(asistente, leer=lambda _: next(entradas), escribir=lambda *partes: salida.append(partes))
    assert [mensaje for mensaje, _ in asistente.recibidos] == ["hola", "¿qué servicios usa?", "salir"]
    assert str(asistente.recibidos[1][1]) == "captura.png"
    assert salida[-1] == ("Bot:", "ok")
