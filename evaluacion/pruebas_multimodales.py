"""
Pruebas funcionales de la integración multimodal (Avance 2)

Conversaciones que combinan imágenes de pantallas (del conjunto de prueba y del estilo
nuevo, que el clasificador no vio al entrenar) con mensajes de texto. Cada turno indica
la intención esperada y, si lleva imagen, el tipo de pantalla esperado.

    python -m evaluacion.pruebas_multimodales    # transcripción en evidencias/transcripcion_multimodal.txt
"""
from __future__ import annotations

from asistente.chatbot_servicios import AsistenteServiciosREST
from vision.generar_pantallas import CARPETA_IMAGENES
from evaluacion.registro_evidencias import guardar_json, guardar_texto

# (mensaje, imagen relativa a data/imagenes o None, intención esperada, pantalla esperada)
CONVERSACIONES_MULTIMODALES = {
    "M01 Anuncia la imagen y luego la envía": [
        ("Te comparto el mockup de la pantalla", None, "analizar_imagen", None),
        ("Aquí está", "prueba/transferencia/transferencia_003.jpg", "analizar_imagen", "transferencia"),
        ("dame el contrato del tercero", None, "detalle_servicio", None),
    ],
    "M02 Captura de login y pregunta de seguimiento": [
        ("¿Qué servicios necesita esta pantalla?", "prueba/login/login_002.jpg", "analizar_imagen", "login"),
        ("¿Qué hago si necesito recuperar la contraseña y no existe el servicio?", None, "reportar_brecha", None),
    ],
    "M03 Maqueta del front de pagos": [
        ("Revisa el front de pagos", "prueba/pago_servicios/pago_servicios_000.jpg", "analizar_imagen", "pago_servicios"),
        ("detalle del primero", None, "detalle_servicio", None),
    ],
    "M04 Boceto (estilo no visto) de seguros": [
        ("Clasifica este boceto", "estilo_nuevo/seguros/seguros_000.jpg", "analizar_imagen", "seguros"),
        ("Servicios de seguros", None, "listar_por_dominio", None),
    ],
    "M05 Boceto de carga de documentos y requerimiento en texto": [
        ("", "estilo_nuevo/documentos/documentos_001.jpg", "analizar_imagen", "documentos"),
        ("Además necesita consultar el estado de la solicitud de hipoteca y registrar al cliente",
         None, "analizar_requerimiento", None),
    ],
    "M06 Varias pantallas en la misma sesión": [
        ("Pantalla 1", "prueba/cuentas/cuentas_001.jpg", "analizar_imagen", "cuentas"),
        ("Pantalla 2", "prueba/prestamo/prestamo_000.jpg", "analizar_imagen", "prestamo"),
        ("Pantalla 3", "prueba/alta_cliente/alta_cliente_000.jpg", "analizar_imagen", "alta_cliente"),
    ],
}


def ejecutar_conversacion(asistente: AsistenteServiciosREST, turnos) -> list[dict]:
    """Envía cada turno al asistente y compara la intención y la pantalla con las esperadas."""
    resultados = []
    for mensaje, imagen, intencion_esperada, pantalla_esperada in turnos:
        ruta_imagen = CARPETA_IMAGENES / imagen if imagen else None
        respuesta = asistente.responder(mensaje, imagen=ruta_imagen)
        registro = asistente.historial[-1]  # intención, pantalla y confianza que registró el asistente
        resultados.append({
            "mensaje": mensaje, "imagen": imagen, "respuesta": respuesta,
            "intencion_esperada": intencion_esperada, "intencion_obtenida": registro["intencion"],
            "pantalla_esperada": pantalla_esperada, "pantalla_obtenida": registro.get("pantalla"),
            "confianza": registro.get("confianza"),
            "correcto": registro["intencion"] == intencion_esperada and registro.get("pantalla") == pantalla_esperada,
        })
    return resultados


def transcribir_turno(turno: dict) -> list[str]:
    """Líneas de la transcripción de un turno: mensaje, respuesta y resultado (OK o FALLA)."""
    adjunto = f" [imagen: {turno['imagen']}]" if turno["imagen"] else ""
    resultado = f"intención {turno['intencion_obtenida']}"
    if turno["imagen"]:
        resultado += f", pantalla {turno['pantalla_obtenida']} ({turno['confianza']:.0%})"
    estado = "OK" if turno["correcto"] else f"FALLA (esperada {turno['intencion_esperada']}/{turno['pantalla_esperada']})"
    return [f"Tú: {turno['mensaje']}{adjunto}", f"Bot: {turno['respuesta']}", f"   [{resultado}] {estado}", ""]


def ejecutar_conversaciones(mostrar: bool = False, guardar: bool = True) -> dict[str, list[dict]]:
    """Ejecuta todas las conversaciones (cada una en una sesión nueva) y, si se pide, guarda la transcripción.

    Las pruebas usan guardar=False para no modificar las evidencias versionadas.
    """
    resultados, transcripcion = {}, []
    for nombre, turnos in CONVERSACIONES_MULTIMODALES.items():
        resultados[nombre] = ejecutar_conversacion(AsistenteServiciosREST(), turnos)
        transcripcion.append(f"===== {nombre} =====")
        for turno in resultados[nombre]:
            transcripcion += transcribir_turno(turno)
    todos_los_turnos = [turno for turnos in resultados.values() for turno in turnos]
    correctos = sum(turno["correcto"] for turno in todos_los_turnos)
    transcripcion.append(f"Turnos correctos: {correctos}/{len(todos_los_turnos)}")

    if guardar:
        guardar_texto("transcripcion_multimodal.txt", transcripcion, mostrar=mostrar)
        guardar_json("resultados_multimodal.json", resultados)
    return resultados


if __name__ == "__main__":
    ejecutar_conversaciones(mostrar=True)
