"""Pruebas (pytest) de casos límite del asistente encontrados en la revisión del código."""
import sys
import types

import pytest

from asistente.chatbot_servicios import AsistenteServiciosREST
from vision import preparar_resnet50


@pytest.fixture
def asistente():
    """Una sesión nueva por prueba."""
    return AsistenteServiciosREST()


@pytest.mark.parametrize("mensaje", ["detalle de SRV-012", "detalle de srv 12", "detalle de SRV 12", "contrato del srv12"])
def test_id_de_servicio_con_o_sin_guion(asistente, mensaje):
    """El ID se reconoce con guion, con espacio o junto, igual que en el preprocesamiento."""
    assert asistente.responder(mensaje).startswith("SRV-012 - ")


def test_posicion_sin_recomendaciones_previas(asistente):
    """Sin una recomendación previa, "el tercero" no se busca por nombre: se avisa."""
    assert asistente.responder("dame el contrato del tercero").startswith("Todavía no te he recomendado")


def test_posicion_fuera_de_la_lista(asistente):
    """Una posición mayor que la lista recomendada se avisa en lugar de mostrar otro servicio."""
    asistente.responder("servicio para consultar saldo")
    cantidad = len(asistente.ultimos_recomendados)
    assert asistente.responder("detalle del quinto").startswith(f"Sólo te recomendé {cantidad} servicio(s)")


def test_posicion_valida(asistente):
    """El ordinal elige el servicio en esa posición de la última recomendación."""
    asistente.responder("servicios de seguros")
    cuarto = asistente.ultimos_recomendados[3]
    assert asistente.responder("detalle del cuarto").startswith(f"{cuarto} - ")


def test_lista_por_dominio_sin_dominio_reconocido(asistente):
    """Si el SVM pide una lista por dominio sin dominio reconocible pero el inventario cubre
    el mensaje, se trata como búsqueda; si no lo cubre, se muestran los dominios."""
    asistente.responder("servicio para pagar")
    assert asistente.historial[-1]["intencion"] == "buscar_servicio"
    asistente.responder("servicio para pagar el recibo de luz")
    assert asistente.historial[-1]["intencion"] == "listar_dominios"


def test_carpeta_en_lugar_de_imagen(asistente, tmp_path):
    """Una carpeta produce un aviso y no detiene el asistente."""
    assert "no es una imagen que pueda abrir" in asistente.responder("", imagen=tmp_path)


def test_resnet50_sin_conexion(monkeypatch):
    """Si Keras no puede descargar los pesos de ResNet50, se informa con un error propio."""
    def descarga_fallida(*argumentos, **opciones):
        """Simula keras.applications.ResNet50 sin conexión."""
        raise Exception("URL fetch failure: sin conexión")   # así informa Keras una descarga fallida

    # Un módulo keras simulado: la prueba no necesita Keras instalado ni conexión
    keras_simulado = types.SimpleNamespace(applications=types.SimpleNamespace(ResNet50=descarga_fallida))
    monkeypatch.setitem(sys.modules, "keras", keras_simulado)
    with pytest.raises(preparar_resnet50.ResNet50NoDisponible):
        preparar_resnet50.crear_resnet50_congelada()


def test_aviso_si_resnet50_no_esta_disponible(asistente, monkeypatch, tmp_path):
    """Si ResNet50 no se pudo descargar, el asistente avisa en lugar de detenerse."""
    from PIL import Image

    from vision import clasificador_visual

    captura = tmp_path / "captura.png"
    Image.new("RGB", (390, 844), "white").save(captura)

    def sin_resnet50(*argumentos, **opciones):
        """Simula un ClasificadorPantallas que no pudo obtener ResNet50."""
        raise preparar_resnet50.ResNet50NoDisponible("sin conexión")

    monkeypatch.setattr(clasificador_visual, "ClasificadorPantallas", sin_resnet50)
    assert "No pude descargar ResNet50" in asistente.responder("", imagen=captura)


def test_modelo_de_otra_version_no_se_usa(tmp_path, monkeypatch):
    """Un modelo guardado con otra versión de scikit-learn se considera no compatible."""
    import joblib
    import sklearn.base
    from sklearn.linear_model import LogisticRegression

    from configuracion.modelos_guardados import cargar_modelo_compatible

    ruta = tmp_path / "modelo.joblib"
    assert cargar_modelo_compatible(ruta) is None                     # no existe
    monkeypatch.setattr(sklearn.base, "__version__", "0.0.1")        # se guarda como si fuera otra versión
    joblib.dump(LogisticRegression().fit([[0], [1]], [0, 1]), ruta)
    monkeypatch.undo()
    assert cargar_modelo_compatible(ruta) is None
    joblib.dump(LogisticRegression().fit([[0], [1]], [0, 1]), ruta)   # misma versión: sí se usa
    assert cargar_modelo_compatible(ruta) is not None
