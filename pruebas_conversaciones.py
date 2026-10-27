"""Pruebas funcionales: conversaciones con la intención y, cuando aplica, el servicio esperado.

El inventario se lee del archivo RAML (data/plataforma_servicios_bancarios.raml).
Las pruebas se refieren a los servicios por "MÉTODO /endpoint" y no por su ID,
porque los IDs (SRV-001, SRV-002...) se asignan según el orden del RAML y cambian
si se agregan o se reordenan recursos.

    python pruebas_conversaciones.py   # reporte completo; guarda evidencias/
    python -m pytest                   # falla si algún turno no cumple lo esperado
"""
import json
import re
from pathlib import Path

from chatbot_servicios import RUTA_INVENTARIO, AsistenteServiciosREST
from inventario_raml import cargar_inventario_raml

RUTA = RUTA_INVENTARIO

_SERVICIOS = cargar_inventario_raml(RUTA)["servicios"]
_ID_POR_OPERACION = {f"{s['metodo']} {s['endpoint']}": s["id"] for s in _SERVICIOS}
_OPERACION_POR_ID = {sid: operacion for operacion, sid in _ID_POR_OPERACION.items()}


def id_de(operacion: str) -> str:
    """ID asignado a una operación del RAML, p. ej. id_de('GET /cuentas/{idCuenta}')."""
    try:
        return _ID_POR_OPERACION[operacion]
    except KeyError:
        raise KeyError(f"La operación '{operacion}' ya no existe en {RUTA.name}; "
                       f"actualiza las pruebas.") from None


# Cada turno: (mensaje, intención esperada) o (mensaje, intención esperada, operación esperada).
# La operación es el primer servicio de la respuesta: el primero recomendado (buscar_servicio,
# listar_por_dominio) o aquel cuyo contrato se muestra (detalle_servicio). SIN_SERVICIO exige
# que la respuesta no nombre ninguno (p. ej. una búsqueda que reporta brecha).
SIN_SERVICIO = "sin servicio"

CONVERSACIONES = {
    "C01 Saludo y ayuda": [
        ("Hola", "charla"),
        ("¿Qué puedes hacer?", "charla"),
        ("Gracias", "charla"),
    ],
    "C02 Búsqueda simple": [
        ("¿Existe un servicio para consultar el saldo de una cuenta?", "buscar_servicio",
         "GET /cuentas/{idCuenta}"),
        ("Dame el detalle del primero", "detalle_servicio", "GET /cuentas/{idCuenta}"),
    ],
    "C03 Búsqueda con sinónimo": [
        ("Necesito una API de login para que el usuario pueda ingresar", "buscar_servicio",
         "POST /autenticacion/sesiones"),
        ("¿Cuál es el contrato?", "detalle_servicio", "POST /autenticacion/sesiones"),
    ],
    "C04 Requerimiento de pantalla completo": [
        ("Requerimiento: pantalla para buscar un cliente por correo, mostrar sus cuentas, "
         "consultar movimientos y registrar un pago", "analizar_requerimiento"),
        ("detalle del tercero", "detalle_servicio", "GET /cuentas/{idCuenta}/movimientos"),
    ],
    "C05 Requerimiento con brecha": [
        ("Necesito una pantalla de alta de cliente, validar el código postal, "
         "calcular el score crediticio y enviar un correo de bienvenida", "analizar_requerimiento"),
        ("¿Qué hago si no existe el servicio?", "reportar_brecha"),
    ],
    "C06 Exploración del inventario": [
        ("¿Qué servicios hay en el inventario?", "listar_dominios"),
        ("Muéstrame los servicios de pagos", "listar_por_dominio", "GET /pagos"),
        (f"Detalle de {id_de('GET /transferencias/{idTransferencia}')}", "detalle_servicio",
         "GET /transferencias/{idTransferencia}"),
    ],
    "C07 Consulta directa por ID": [
        (id_de("GET /cuentas/{idCuenta}").lower(), "detalle_servicio", "GET /cuentas/{idCuenta}"),
        ("¿Hay servicio para ver el historial de movimientos de la cuenta?", "buscar_servicio",
         "GET /cuentas/{idCuenta}/movimientos"),
    ],
    "C08 Imagen de pantalla (fuera de v1)": [
        ("Te comparto la imagen del mockup de Figma", "analizar_imagen"),
        ("Pantalla con resumen de cuentas, transferencia a un beneficiario y pago de servicios",
         "analizar_requerimiento"),
    ],
    "C09 Sin palabra disparadora": [
        ("movimientos de la cuenta del último mes", "buscar_servicio",
         "GET /cuentas/{idCuenta}/movimientos"),
        ("estado de cuenta en pdf", "buscar_servicio", "POST /cuentas/{idCuenta}/estado-cuenta"),
    ],
    "C10 Fuera de alcance": [
        ("¿Cuál es el clima en Monterrey?", "fuera_de_alcance"),
        ("Cuéntame un chiste", "fuera_de_alcance"),
        ("ayuda", "charla"),
    ],
    "C11 Comprobantes y beneficiarios": [
        ("Necesito generar el comprobante de una transferencia", "buscar_servicio",
         "POST /transferencias/{idTransferencia}/comprobante"),
        ("Formulario para registrar un beneficiario con su límite diario y correo de notificación",
         "analizar_requerimiento"),
    ],
    "C12 Cierre de sesión y despedida": [
        ("¿Existe un endpoint de logout?", "buscar_servicio", "DELETE /autenticacion/sesiones/{idSesion}"),
        ("Adiós", "charla"),
    ],
    # ---- casos detectados en la revisión integral ----
    "C13 Fuera de dominio con palabra disparadora": [
        ("¿Qué ropa uso para mañana?", "fuera_de_alcance"),
    ],
    "C14 Charla seguida de una petición": [
        ("Perfecto, ahora necesito un servicio para consultar movimientos", "buscar_servicio",
         "GET /cuentas/{idCuenta}/movimientos"),
    ],
    "C15 Búsqueda que menciona el inventario": [
        ("¿Existe en el inventario un servicio para consultar el saldo?", "buscar_servicio",
         "GET /cuentas/{idCuenta}"),
    ],
    "C16 Detalle de un servicio nombrado tras una búsqueda": [
        ("servicio para consultar movimientos", "buscar_servicio", "GET /cuentas/{idCuenta}/movimientos"),
        ("dame el contrato del servicio de cancelación de seguros", "detalle_servicio",
         "POST /seguros/{idSeguro}/cancelacion"),
    ],
    "C17 Dominio que aparece junto a otro": [
        ("servicios de seguros para clientes", "listar_por_dominio", "GET /seguros"),
    ],
    "C18 Requerimiento que contiene 'no existe'": [
        ("servicio para validar que el cliente no existe antes de darlo de alta", "buscar_servicio"),
    ],
    "C19 Búsqueda sin resultados tras una recomendación": [
        ("servicio para consultar movimientos", "buscar_servicio", "GET /cuentas/{idCuenta}/movimientos"),
        ("necesito un servicio para invertir en fondos", "buscar_servicio", SIN_SERVICIO),
        ("detalle del primero", "detalle_servicio", "GET /cuentas/{idCuenta}/movimientos"),
    ],
}


def operacion_obtenida(respuesta: str) -> str:
    """Primer servicio que nombra la respuesta, como "MÉTODO /endpoint", o SIN_SERVICIO.

    Se lee del texto y no del contexto de la sesión: una búsqueda sin resultados conserva
    la recomendación del turno anterior y no debe contar como si la hubiera hecho.
    """
    m = re.search(r"\bSRV-\d{3}\b", respuesta)
    return _OPERACION_POR_ID[m.group()] if m else SIN_SERVICIO


def evaluar_conversacion(turnos: list[tuple]) -> list[dict]:
    """Ejecuta una conversación en una sesión nueva y evalúa cada turno."""
    bot = AsistenteServiciosREST(RUTA)
    resultados = []
    for mensaje, intencion_esperada, *operacion in turnos:
        operacion_esperada = operacion[0] if operacion else None
        respuesta = bot.responder(mensaje)
        intencion = bot.historial[-1]["intencion"]
        obtenida = operacion_obtenida(respuesta) if operacion_esperada else None
        intencion_ok = intencion == intencion_esperada
        operacion_ok = obtenida == operacion_esperada
        resultados.append({"mensaje": mensaje, "respuesta": respuesta,
                           "esperada": intencion_esperada, "obtenida": intencion,
                           "operacion_esperada": operacion_esperada, "operacion_obtenida": obtenida,
                           "intencion_ok": intencion_ok, "operacion_ok": operacion_ok,
                           "correcto": intencion_ok and operacion_ok})
    return resultados


def ejecutar():
    resultados, transcript = [], []
    for nombre, turnos in CONVERSACIONES.items():
        transcript.append(f"===== {nombre} =====")
        for r in evaluar_conversacion(turnos):
            resultados.append({"conversacion": nombre, **r})
            transcript.append(f"Tú : {r['mensaje']}")
            transcript.append("Bot: " + r["respuesta"].replace("\n", "\n     "))
            verificacion = f"intención: {r['obtenida']} | esperada: {r['esperada']}"
            if r["operacion_esperada"]:
                verificacion += (f" | servicio: {r['operacion_obtenida']}"
                                 f" | esperado: {r['operacion_esperada']}")
            transcript.append(f"     [{verificacion} | {'OK' if r['correcto'] else 'FALLA'}]")
        transcript.append("")
    total = len(resultados)
    intenciones = sum(r["intencion_ok"] for r in resultados)
    con_servicio = [r for r in resultados if r["operacion_esperada"]]
    servicios = sum(r["operacion_ok"] for r in con_servicio)
    turnos_ok = sum(r["correcto"] for r in resultados)
    transcript += [f"Intenciones correctas: {intenciones}/{total} = {100 * intenciones / total:.1f}%",
                   f"Servicios correctos: {servicios}/{len(con_servicio)}",
                   f"Turnos correctos (intención y servicio): {turnos_ok}/{total}"]
    salida = Path(__file__).parent / "evidencias"
    salida.mkdir(exist_ok=True)
    (salida / "transcripcion_pruebas.txt").write_text("\n".join(transcript), encoding="utf-8")
    (salida / "resultados_pruebas.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print("\n".join(transcript))
    return resultados


# Consultas parafraseadas (exploratorio). Se indica la operación esperada como
# "MÉTODO /endpoint"; None significa que el RAML no tiene servicio y lo correcto es reportar brecha.
GENERALIZACION = [
    ("quiero saber cuánto dinero tengo disponible", "GET /cuentas/{idCuenta}"),
    ("cambiar mi correo electrónico", "PATCH /clientes/{idCliente}"),
    ("mandar dinero a otro banco", "POST /transferencias"),
    ("ver cuánto debo de mi crédito", "GET /prestamos/{idPrestamo}"),
    ("tabla de mensualidades del préstamo", "GET /prestamos/{idPrestamo}/amortizacion"),
    ("agregar un contacto frecuente para enviar dinero", "POST /clientes/{idCliente}/beneficiarios"),
    ("pagar el recibo de luz", "POST /pagos"),
    ("cancelar mi póliza de seguro", "POST /seguros/{idSeguro}/cancelacion"),
    ("bloquear mi tarjeta de crédito", None),
    ("consultar el tipo de cambio del dólar", None),
    ("Servicio para registrar un cliente", "POST /clientes"),
    ("dar de alta a un nuevo cliente en el banco", "POST /clientes"),
]

# Conjunto de validación: escrito antes de ajustar el puntaje y nunca usado para
# elegir sinónimos ni parámetros. Mide la generalización real.
GENERALIZACION_VALIDACION = [
    ("buscar clientes por nombre", "GET /clientes"),
    ("actualizar el teléfono del cliente", "PATCH /clientes/{idCliente}"),
    ("qué cuentas tiene el cliente", "GET /clientes/{idCliente}/cuentas"),
    ("borrar un beneficiario que ya no uso", "DELETE /clientes/{idCliente}/beneficiarios/{idBeneficiario}"),
    ("abrir una cuenta de ahorro", "POST /cuentas"),
    ("congelar una cuenta por robo", "PATCH /cuentas/{idCuenta}"),
    ("pedir el estado de cuenta del mes pasado", "POST /cuentas/{idCuenta}/estado-cuenta"),
    ("solicitar un préstamo personal", "POST /prestamos"),
    ("adelantar un pago a mi préstamo", "POST /prestamos/{idPrestamo}/pagos"),
    ("revisar si mi transferencia ya llegó", "GET /transferencias/{idTransferencia}"),
    ("anular una transferencia que hice por error", "POST /transferencias/{idTransferencia}/cancelacion"),
    ("obtener el comprobante de la transferencia en pdf", "POST /transferencias/{idTransferencia}/comprobante"),
    ("lista de empresas a las que puedo pagar", "GET /pagos/convenios"),
    ("pedir un crédito para comprar casa", "POST /hipotecas"),
    ("subir la escritura de la casa para la hipoteca", "POST /hipotecas/{idHipoteca}/documentos"),
    ("cotizar un seguro de auto", "POST /seguros"),
    ("contratar el seguro que coticé", "POST /seguros/{idSeguro}/contratacion"),
    ("iniciar sesión con usuario y contraseña", "POST /autenticacion/sesiones"),
    ("validar el código que llega por SMS", "POST /autenticacion/verificacion-segundo-factor"),
    ("encriptar el número de tarjeta", "POST /criptografia/cifrados"),
    ("invertir en un fondo de inversión", None),
    ("solicitar una chequera", None),
    ("consultar los puntos de recompensa de mi tarjeta", None),
    ("programar un retiro sin tarjeta en cajero", None),
]


def evaluar_generalizacion(bot, casos):
    """Compara la mejor coincidencia de cada consulta con la operación esperada."""
    lineas, aciertos = [], 0
    for consulta, esperado in casos:
        servicio = bot.recomendar(consulta)
        obtenido = f"{servicio['metodo']} {servicio['endpoint']}" if servicio else None
        ok = obtenido == esperado
        aciertos += ok
        lineas.append(f"{'OK   ' if ok else 'FALLA'} | {consulta:50} | esperado={esperado} | obtenido={obtenido}")
    return lineas, aciertos


def prueba_generalizacion():
    """None = no existe servicio; lo correcto es reportar brecha."""
    bot = AsistenteServiciosREST(RUTA)
    lineas = []
    for titulo, casos in [("Ajuste", GENERALIZACION), ("Validación", GENERALIZACION_VALIDACION)]:
        resultado, aciertos = evaluar_generalizacion(bot, casos)
        lineas += [f"===== {titulo} =====", *resultado,
                   f"Aciertos {titulo.lower()}: {aciertos}/{len(casos)}", ""]
    destino = Path(__file__).parent / "evidencias"
    destino.mkdir(exist_ok=True)
    (destino / "prueba_generalizacion.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    ejecutar()
    prueba_generalizacion()
