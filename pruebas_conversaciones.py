"""Pruebas funcionales: 12 conversaciones con la intención esperada en cada turno."""
import json
from pathlib import Path

from chatbot_servicios import AsistenteServiciosREST

RUTA = Path(__file__).parent / "data" / "inventario_servicios.json"

CONVERSACIONES = {
    "C01 Saludo y ayuda": [
        ("Hola", "charla"),
        ("¿Qué puedes hacer?", "charla"),
        ("Gracias", "charla"),
    ],
    "C02 Búsqueda simple": [
        ("¿Existe un servicio para consultar el saldo de una cuenta?", "buscar_servicio"),
        ("Dame el detalle del primero", "detalle_servicio"),
    ],
    "C03 Búsqueda con sinónimo": [
        ("Necesito una API de login para que el usuario pueda ingresar", "buscar_servicio"),
        ("¿Cuál es el contrato?", "detalle_servicio"),
    ],
    "C04 Requerimiento de pantalla completo": [
        ("Requerimiento: pantalla para buscar un cliente por RFC, mostrar su saldo, "
         "consultar movimientos y registrar un pago", "analizar_requerimiento"),
        ("detalle del tercero", "detalle_servicio"),
    ],
    "C05 Requerimiento con brecha": [
        ("Necesito una pantalla de alta de cliente, validar el código postal, "
         "calcular el score crediticio y enviar un correo de bienvenida", "analizar_requerimiento"),
        ("¿Qué hago si no existe el servicio?", "reportar_brecha"),
    ],
    "C06 Exploración del inventario": [
        ("¿Qué servicios hay en el inventario?", "listar_dominios"),
        ("Muéstrame los servicios de pagos", "listar_por_dominio"),
        ("Detalle de SRV-011", "detalle_servicio"),
    ],
    "C07 Consulta directa por ID": [
        ("srv-13", "detalle_servicio"),
        ("¿Hay servicio para ver el historial de pedidos del cliente?", "buscar_servicio"),
    ],
    "C08 Imagen de pantalla (fuera de v1)": [
        ("Te comparto la imagen del mockup de Figma", "analizar_imagen"),
        ("Pantalla con catálogo de productos, carrito de compra y pago con tarjeta", "analizar_requerimiento"),
    ],
    "C09 Sin palabra disparadora": [
        ("transacciones de la cuenta del último mes", "buscar_servicio"),
        ("estado de cuenta en pdf", "buscar_servicio"),
    ],
    "C10 Fuera de alcance": [
        ("¿Cuál es el clima en Monterrey?", "fuera_de_alcance"),
        ("Cuéntame un chiste", "fuera_de_alcance"),
        ("ayuda", "charla"),
    ],
    "C11 Documentos y notificaciones": [
        ("Necesito subir la identificación del cliente y avisarle por SMS", "buscar_servicio"),
        ("Formulario para adjuntar comprobante de domicilio y actualizar el teléfono del cliente",
         "analizar_requerimiento"),
    ],
    "C12 Cierre de sesión y despedida": [
        ("¿Existe un endpoint de logout?", "buscar_servicio"),
        ("Adiós", "charla"),
    ],
}


def ejecutar():
    resultados, aciertos, total = [], 0, 0
    transcript = []
    for nombre, turnos in CONVERSACIONES.items():
        bot = AsistenteServiciosREST(RUTA)          # sesión nueva por conversación
        transcript.append(f"===== {nombre} =====")
        for mensaje, esperada in turnos:
            respuesta = bot.responder(mensaje)
            obtenida = bot.historial[-1]["intencion"]
            ok = obtenida == esperada
            aciertos += ok
            total += 1
            resultados.append({"conversacion": nombre, "mensaje": mensaje, "esperada": esperada,
                               "obtenida": obtenida, "correcto": ok, "respuesta": respuesta})
            transcript.append(f"Tú : {mensaje}")
            transcript.append("Bot: " + respuesta.replace("\n", "\n     "))
            transcript.append(f"     [intención: {obtenida} | esperada: {esperada} | {'OK' if ok else 'FALLA'}]")
        transcript.append("")
    transcript.append(f"Precisión de intenciones: {aciertos}/{total} = {100*aciertos/total:.1f}%")
    salida = Path(__file__).parent / "evidencias"
    salida.mkdir(exist_ok=True)
    (salida / "transcripcion_pruebas.txt").write_text("\n".join(transcript), encoding="utf-8")
    (salida / "resultados_pruebas.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print("\n".join(transcript))
    return resultados


# Consultas parafraseadas que NO se usaron para ajustar las reglas (exploratorio)
GENERALIZACION = [
    ("quiero saber cuánto dinero tiene el cliente", "SRV-007"),
    ("cambiar el correo electrónico del cliente", "SRV-006"),
    ("cancelar un pedido", None),
    ("solicitar un reembolso", None),
    ("ver las compras que hizo el cliente", "SRV-014"),
    ("dar de baja a un cliente", None),
    ("recuperar contraseña olvidada", None),
    ("enviar una alerta push", "SRV-015"),
    ("precio de un artículo", "SRV-012"),
    ("descargar el comprobante del pago", None),
]


def prueba_generalizacion():
    """None = no existe servicio; lo correcto es reportar brecha."""
    bot = AsistenteServiciosREST(RUTA)
    lineas, aciertos = [], 0
    for consulta, esperado in GENERALIZACION:
        res = bot.kb.buscar(consulta, maximo=1)
        obtenido = res[0][1]["id"] if res else None
        ok = obtenido == esperado
        aciertos += ok
        lineas.append(f"{'OK   ' if ok else 'FALLA'} | {consulta:45} | esperado={esperado} | obtenido={obtenido}")
    lineas.append(f"Aciertos en generalización: {aciertos}/{len(GENERALIZACION)}")
    (Path(__file__).parent / "evidencias" / "prueba_generalizacion.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    ejecutar()
    prueba_generalizacion()
