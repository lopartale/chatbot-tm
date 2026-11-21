"""
Plantillas HTML de las pantallas de banca digital (Avance 2)

Arma el HTML de una pantalla de un tipo dado, eligiendo al azar (con semilla) textos,
campos, colores, tipografía y navegación. generar_pantallas.py la captura como imagen.

    armar_pagina_html(azar, tipo_pantalla, dispositivo)
        -> marco de la aplicación (banco, menú, navegación)
        -> contenido del tipo de pantalla (contenido_login, contenido_cuentas, ...)
        -> estilo visual (estilo "alta", "front" o "boceto") como variables CSS

Importante: cada elección al azar consume la secuencia de números aleatorios, así que el
orden de las elecciones determina las imágenes generadas con cada semilla.
"""
from __future__ import annotations

import random

BANCOS = ["Banco Aurora", "Finanzas Delta", "Banca Norte", "Kuali Bank", "BancoSur Digital",
          "Mi Banca", "Grupo Ceiba", "Banco Arrecife", "Nova Financiera", "Banca Tlaloc",
          "Banco Pradera", "Altavista Bank"]
FUENTES = ["Inter", "DejaVu Sans", "FreeSans", "Carlito", "Caladea", "DejaVu Serif", "Bitstream Charter"]
MODULOS_DEL_MENU = ["Inicio", "Cuentas", "Transferir", "Pagos", "Créditos", "Seguros", "Documentos",
                    "Perfil", "Ayuda", "Inversiones", "Tarjetas", "Mensajes"]
NOMBRES = ["Ana", "Luis", "María", "Jorge", "Sofía", "Carlos", "Valeria", "Diego", "Fernanda", "Raúl"]
APELLIDOS = ["García", "López", "Hernández", "Martínez", "Ramírez", "Torres", "Flores", "Cruz"]


class GeneradorAleatorio:
    """Elecciones al azar reproducibles para una pantalla, según su estilo visual."""

    def __init__(self, aleatorio: random.Random, estilo: str):
        self.aleatorio = aleatorio
        self.estilo = estilo  # "alta" (diseño final), "front" (maqueta) o "boceto"

    def elegir(self, *opciones):
        """Una de las opciones."""
        return self.aleatorio.choice(opciones)

    def elegir_varios(self, opciones, minimo: int, maximo: int) -> list:
        """Entre `minimo` y `maximo` opciones distintas, en orden aleatorio."""
        cantidad = self.aleatorio.randint(minimo, min(maximo, len(opciones)))
        return self.aleatorio.sample(list(opciones), cantidad)

    def con_probabilidad(self, probabilidad: float = 0.5) -> bool:
        """True con la probabilidad dada: decide si un elemento opcional aparece en la pantalla."""
        return self.aleatorio.random() < probabilidad

    def entero(self, minimo: int, maximo: int) -> int:
        """Entero entre minimo y maximo (ambos incluidos)."""
        return self.aleatorio.randint(minimo, maximo)

    def decimal(self, minimo: float, maximo: float) -> float:
        """Número con decimales entre minimo y maximo."""
        return self.aleatorio.uniform(minimo, maximo)

    def posicion(self, cantidad: int) -> int:
        """Índice al azar de una lista de `cantidad` elementos."""
        return self.aleatorio.randrange(cantidad)

    def revolver(self, elementos: list) -> None:
        """Cambia el orden de la lista (por ejemplo, el de los campos de un formulario)."""
        self.aleatorio.shuffle(elementos)

    def monto(self, minimo: float = 50, maximo: float = 90000) -> str:
        """Importe en pesos con formato: "$1,234.56"."""
        return f"${self.decimal(minimo, maximo):,.2f}"

    def fecha(self) -> str:
        """Fecha de 2026 en formato dd/mm/aaaa."""
        dia = self.entero(1, 28)
        mes = self.entero(1, 12)
        return f"{dia:02d}/{mes:02d}/2026"

    def nombre_persona(self) -> str:
        """Nombre y apellido ficticios."""
        return f"{self.elegir(*NOMBRES)} {self.elegir(*APELLIDOS)}"

    def texto(self, contenido: str, secundario: bool = False) -> str:
        """Texto en un <span>; en las maquetas ("front"), la mitad del texto secundario es una barra gris."""
        if secundario and self.estilo == "front" and self.con_probabilidad(0.5):
            return f'<span class="barra" style="width:{self.entero(40, 140)}px"></span>'
        clase = "sec" if secundario else ""
        return f'<span class="{clase}">{contenido}</span>'


# --------------------------------------------------------------------------
# Elementos de interfaz
# --------------------------------------------------------------------------
def campo(azar: GeneradorAleatorio, etiqueta: str, valor: str = "", es_lista: bool = False) -> str:
    """Etiqueta y caja de captura; sin valor, la caja muestra un texto de ejemplo o queda vacía."""
    contenido = valor or azar.elegir("", "", "Escribe aquí", "—")
    flecha = '<span class="flecha">▾</span>' if es_lista else ""
    return (f'<div class="campo"><label>{etiqueta}</label>'
            f'<div class="entrada">{azar.texto(contenido, secundario=True)}{flecha}</div></div>')


def boton(texto: str, principal: bool = True) -> str:
    """Botón principal (relleno) o secundario (sólo contorno)."""
    clase = "prim" if principal else "secu"
    return f'<button class="{clase}">{texto}</button>'


def titulo(azar: GeneradorAleatorio, *opciones: str) -> str:
    """Título de la pantalla elegido entre varias redacciones."""
    return f"<h1>{azar.elegir(*opciones)}</h1>"


def subtitulo(azar: GeneradorAleatorio, *opciones: str) -> str:
    """Texto secundario bajo el título."""
    return f'<p class="sub">{azar.texto(azar.elegir(*opciones), secundario=True)}</p>'


def tarjeta(contenido: str, clase: str = "") -> str:
    """Contenedor con fondo, borde y sombra."""
    return f'<div class="tarjeta {clase}">{contenido}</div>'


def casilla(texto: str) -> str:
    """Casilla de verificación con su texto ("Recordar usuario", "Acepto el aviso...")."""
    return f'<label class="check"><span class="caja"></span>{texto}</label>'


def pasos(azar: GeneradorAleatorio, etiquetas: list[str]) -> str:
    """Indicador de pasos (1 Datos · 2 Confirmar · 3 Comprobante) con un paso actual al azar."""
    paso_actual = azar.posicion(len(etiquetas))
    html = ""
    for numero, etiqueta in enumerate(etiquetas):
        clase = "activo" if numero <= paso_actual else ""
        html += f'<div class="paso {clase}"><b>{numero + 1}</b>{etiqueta}</div>'
    return f'<div class="pasos">{html}</div>'


def tabla(azar: GeneradorAleatorio, encabezados: list[str], filas: list[list[str]]) -> str:
    """Tabla HTML; la primera columna es texto principal y las demás, secundario."""
    encabezado = "".join(f"<th>{titulo_columna}</th>" for titulo_columna in encabezados)
    cuerpo = ""
    for fila in filas:
        celdas = "".join(f"<td>{azar.texto(valor, secundario=columna > 0)}</td>"
                         for columna, valor in enumerate(fila))
        cuerpo += f"<tr>{celdas}</tr>"
    return f"<table><thead><tr>{encabezado}</tr></thead><tbody>{cuerpo}</tbody></table>"


def lista(azar: GeneradorAleatorio, filas: list[tuple[str, str, str, str]]) -> str:
    """Filas con icono, texto principal, texto secundario y un valor a la derecha."""
    html = ""
    for icono, principal, secundario, valor in filas:
        html += (f'<div class="fila"><span class="icono">{icono}</span><div class="crece">'
                 f'<div>{principal}</div><div class="sec">{azar.texto(secundario, secundario=True)}</div></div>'
                 f'<b>{valor}</b></div>')
    return f'<div class="lista">{html}</div>'


# --------------------------------------------------------------------------
# Contenido de cada tipo de pantalla
# --------------------------------------------------------------------------
def contenido_login(azar: GeneradorAleatorio) -> str:
    """Inicio de sesión: usuario, contraseña, a veces código de verificación, botón de entrar."""
    campos = [
        campo(azar, azar.elegir("Usuario", "Correo electrónico", "Número de cliente", "Usuario o correo")),
        campo(azar, azar.elegir("Contraseña", "Clave de acceso", "NIP"), "••••••••"),
    ]
    if azar.con_probabilidad(0.35):
        campos.append(campo(azar, azar.elegir("Código de verificación", "Token", "Código SMS"), "______"))
    casillas = ""
    if azar.con_probabilidad(0.6):
        casillas = casilla(azar.elegir("Recordar usuario", "Mantener sesión iniciada"))
    enlace = f'<a>{azar.elegir("¿Olvidaste tu contraseña?", "Recuperar acceso", "¿Problemas para entrar?")}</a>'
    cuerpo = titulo(azar, "Iniciar sesión", "Bienvenido", "Accede a tu banca", "Ingresa a tu cuenta", "Hola de nuevo")
    cuerpo += subtitulo(azar, "Ingresa tus datos para continuar", "Banca en línea segura", "Tu banco, siempre contigo")
    cuerpo += "".join(campos) + casillas
    cuerpo += boton(azar.elegir("Entrar", "Ingresar", "Iniciar sesión", "Continuar")) + enlace
    if azar.con_probabilidad(0.4):
        pregunta = azar.elegir("¿Aún no tienes cuenta?", "¿Eres nuevo?")
        cuerpo += f'<p class="sub">{pregunta} <a>{azar.elegir("Regístrate", "Abre tu cuenta")}</a></p>'
    formulario = tarjeta(cuerpo, "angosta")
    if azar.con_probabilidad(0.4):  # ilustración a un lado del formulario
        lema = azar.texto(azar.elegir("Seguridad", "Tu dinero protegido", "Banca 24/7"), secundario=True)
        formulario = f'<div class="dividido"><div class="ilustracion">{lema}</div>{formulario}</div>'
    return formulario


def contenido_cuentas(azar: GeneradorAleatorio) -> str:
    """Resumen de cuentas: tarjetas de saldo y movimientos en tabla o en lista."""
    tarjetas_de_saldo = ""
    for _ in range(azar.entero(1, 3)):
        tipo_cuenta = azar.elegir("Cuenta de ahorro", "Cuenta nómina", "Cuenta digital", "Cheques")
        terminacion = azar.entero(1000, 9999)
        saldo = azar.monto(1000, 250000)
        etiqueta_saldo = azar.elegir("Saldo disponible", "Disponible", "Saldo actual")
        tarjetas_de_saldo += tarjeta(f'<div class="sec">{tipo_cuenta} •••{terminacion}</div>'
                                     f'<div class="grande">{saldo}</div><div class="sec">{etiqueta_saldo}</div>', "saldo")
    conceptos = ["Depósito de nómina", "Compra en supermercado", "Retiro en cajero", "Pago de tarjeta",
                 "Transferencia recibida", "Cargo domiciliado", "Comisión", "Compra en línea", "Abono de intereses"]
    movimientos = []
    for _ in range(azar.entero(4, 9)):
        fecha = azar.fecha()
        concepto = azar.elegir(*conceptos)
        signo = "+" if azar.con_probabilidad(0.35) else "-"
        movimientos.append((fecha, concepto, signo + azar.monto(20, 15000)))
    if azar.con_probabilidad(0.5):
        encabezados = ["Fecha", "Concepto", "Importe"]
        if azar.con_probabilidad(0.5):
            encabezados.append("Saldo")  # columna anunciada en el encabezado, sin valores
        detalle = tabla(azar, encabezados, [list(movimiento) for movimiento in movimientos])
    else:
        filas = [(azar.elegir("↓", "↑", "⇄", "●"), concepto, fecha, importe) for fecha, concepto, importe in movimientos]
        detalle = lista(azar, filas)
    filtros = ""
    if azar.con_probabilidad(0.6):
        filtros = f'<div class="filtros">{campo(azar, "Desde", azar.fecha())}{campo(azar, "Hasta", azar.fecha())}</div>'
    texto_boton = azar.elegir("Descargar estado de cuenta", "Exportar PDF", "Ver estado de cuenta")
    acciones = boton(texto_boton, principal=azar.con_probabilidad(0.5))
    encabezado = titulo(azar, "Mis cuentas", "Resumen", "Estado de cuenta", "Movimientos", "Inicio")
    seccion = azar.elegir("Últimos movimientos", "Movimientos recientes", "Historial")
    return (encabezado + f'<div class="fila-tarjetas">{tarjetas_de_saldo}</div>'
            + f"<h2>{seccion}</h2>" + filtros + detalle + acciones)


def contenido_transferencia(azar: GeneradorAleatorio) -> str:
    """Transferencia: cuenta de origen, beneficiario, monto y campos opcionales en orden aleatorio."""
    campos = [
        campo(azar, azar.elegir("Cuenta de origen", "Desde", "Cargar a"), f"Ahorro •••{azar.entero(1000, 9999)}", True),
        campo(azar, azar.elegir("Beneficiario", "Destinatario", "Para"), azar.nombre_persona(), True),
        campo(azar, "Monto", azar.monto(100, 20000)),
    ]
    if azar.con_probabilidad(0.6):
        etiqueta = azar.elegir("CLABE", "Cuenta destino", "Número de cuenta")
        clabe = " ".join(str(azar.entero(1000, 9999)) for _ in range(4))
        campos.append(campo(azar, etiqueta, clabe))
    if azar.con_probabilidad(0.6):
        etiqueta = azar.elegir("Concepto", "Motivo", "Referencia")
        campos.append(campo(azar, etiqueta, azar.elegir("Renta", "Pago", "Préstamo", "Cena", "")))
    if azar.con_probabilidad(0.3):
        etiqueta = azar.elegir("Fecha de aplicación", "Programar para")
        campos.append(campo(azar, etiqueta, azar.fecha()))
    azar.revolver(campos)
    indicador_pasos = pasos(azar, ["Datos", "Confirmar", "Comprobante"]) if azar.con_probabilidad(0.4) else ""
    frecuentes = ""
    if azar.con_probabilidad(0.4):
        seccion = azar.elegir("Beneficiarios frecuentes", "Contactos")
        contactos = [(azar.elegir("👤", "●"), azar.nombre_persona(), f"•••{azar.entero(1000, 9999)}", "")
                     for _ in range(3)]
        frecuentes = f"<h2>{seccion}</h2>" + lista(azar, contactos)
    encabezado = titulo(azar, "Transferir", "Nueva transferencia", "Enviar dinero", "Transferencias", "Transferencia SPEI")
    formulario = tarjeta("".join(campos) + boton(azar.elegir("Continuar", "Transferir", "Enviar", "Siguiente")))
    return encabezado + indicador_pasos + formulario + frecuentes


def contenido_pago_servicios(azar: GeneradorAleatorio) -> str:
    """Pago de servicios: rejilla de convenios (luz, agua...), formulario de referencia y pagos recientes."""
    servicios = azar.elegir_varios([("💡", "Luz"), ("💧", "Agua"), ("📞", "Teléfono"), ("🌐", "Internet"),
                                    ("🔥", "Gas"), ("📺", "TV de paga"), ("🏠", "Predial"), ("🚗", "Tenencia"),
                                    ("📱", "Recarga celular"), ("🎓", "Colegiatura")], 4, 8)
    rejilla = "".join(f'<div class="servicio"><span class="icono">{icono}</span>{nombre}</div>'
                      for icono, nombre in servicios)
    rejilla = f'<div class="rejilla">{rejilla}</div>'
    formulario = ""
    if azar.con_probabilidad(0.7):
        nombres_servicios = [nombre for _, nombre in servicios]
        formulario = campo(azar, azar.elegir("Convenio", "Servicio", "Empresa"), azar.elegir(*nombres_servicios), True)
        formulario += campo(azar, azar.elegir("Referencia", "Número de servicio", "Número de contrato"),
                            str(azar.entero(10**9, 10**10)))
        formulario += campo(azar, azar.elegir("Importe", "Monto a pagar"), azar.monto(100, 3000))
        formulario = tarjeta(formulario + boton(azar.elegir("Pagar", "Pagar ahora", "Continuar")))
    recientes = ""
    if azar.con_probabilidad(0.5):
        seccion = azar.elegir("Pagos recientes", "Historial de pagos")
        pagos = [(icono, nombre, azar.fecha(), azar.monto(100, 3000))
                 for icono, nombre in azar.elegir_varios(servicios, 2, 3)]
        recientes = f"<h2>{seccion}</h2>" + lista(azar, pagos)
    encabezado = titulo(azar, "Pago de servicios", "Pagar", "Pagos", "¿Qué quieres pagar?", "Servicios")
    encabezado += subtitulo(azar, "Selecciona el servicio", "Paga tus recibos sin comisión", "Elige un convenio")
    return encabezado + rejilla + formulario + recientes


def contenido_prestamo(azar: GeneradorAleatorio) -> str:
    """Préstamo: monto, plazo, tasa y mensualidad, y a veces la tabla de amortización."""
    monto = azar.entero(10, 500) * 1000
    plazo = azar.elegir(12, 18, 24, 36, 48)
    tipo_credito = azar.elegir("Monto del crédito", "Préstamo personal", "Crédito de nómina", "Saldo pendiente")
    avance = azar.entero(15, 90)
    tasa = azar.decimal(9, 45)
    etiqueta_pago = azar.elegir("Mensualidad", "Pago mensual")
    mensualidad = azar.monto(500, 20000)
    resumen = tarjeta(
        f'<div class="sec">{tipo_credito}</div><div class="grande">${monto:,.2f}</div>'
        f'<div class="deslizador"><span style="width:{avance}%"></span></div>'
        f'<div class="filtros"><div><div class="sec">Plazo</div><b>{plazo} meses</b></div>'
        f'<div><div class="sec">Tasa anual</div><b>{tasa:.1f}%</b></div>'
        f'<div><div class="sec">{etiqueta_pago}</div><b>{mensualidad}</b></div></div>', "saldo")
    pagos_programados = []
    for numero in range(azar.entero(3, 7)):
        pagos_programados.append([str(numero + 1), azar.fecha(), azar.monto(300, 9000),
                                  azar.monto(50, 3000), azar.monto(1000, monto)])
    amortizacion = ""
    if azar.con_probabilidad(0.7):
        seccion = azar.elegir("Tabla de amortización", "Calendario de pagos", "Próximos pagos")
        amortizacion = f"<h2>{seccion}</h2>" + tabla(azar, ["No.", "Fecha", "Capital", "Interés", "Saldo"],
                                                    pagos_programados)
    encabezado = titulo(azar, "Créditos", "Mi préstamo", "Simulador de crédito", "Solicita tu préstamo", "Préstamo personal")
    accion = boton(azar.elegir("Solicitar crédito", "Pagar mensualidad", "Simular", "Adelantar pago"))
    return encabezado + resumen + amortizacion + accion


def contenido_seguros(azar: GeneradorAleatorio) -> str:
    """Seguros: tarjetas de pólizas con suma asegurada y prima, y a veces un cotizador."""
    seguros = azar.elegir_varios([("🚗", "Seguro de auto"), ("❤", "Seguro de vida"), ("🏠", "Seguro de hogar"),
                                  ("✚", "Gastos médicos"), ("✈", "Seguro de viaje"), ("📱", "Protección celular")], 2, 4)
    tarjetas_de_polizas = ""
    for icono, nombre in seguros:
        cobertura = azar.elegir("Cobertura amplia", "Suma asegurada", "Desde")
        suma_asegurada = azar.monto(100000, 3000000)
        prima = azar.monto(150, 2500)
        periodo = azar.elegir("/mes", "al mes", "prima mensual")
        estado = azar.elegir("<span class=chip>Vigente</span>", "<span class=chip>Póliza activa</span>", "")
        texto_boton = azar.elegir("Cotizar", "Contratar", "Ver póliza", "Más información")
        tarjetas_de_polizas += tarjeta(
            f'<div class="icono grande-icono">{icono}</div><h3>{nombre}</h3>'
            f'<div class="sec">{cobertura} {suma_asegurada}</div><div><b>{prima}</b> {periodo}</div>{estado}'
            + boton(texto_boton, principal=azar.con_probabilidad(0.5)), "seguro")
    cotizador = ""
    if azar.con_probabilidad(0.4):
        cotizador = f'<h2>{azar.elegir("Cotiza tu seguro", "Cotizador")}</h2>'
        cotizador += campo(azar, "Marca", azar.elegir("Nissan", "Toyota", "Mazda", "Kia"), True)
        cotizador += campo(azar, "Modelo", azar.elegir("Versa", "Corolla", "CX-30", "Rio"), True)
        cotizador += campo(azar, "Año", str(azar.entero(2012, 2026)), True)
        cotizador = tarjeta(cotizador + boton("Cotizar"))
    encabezado = titulo(azar, "Seguros", "Mis seguros", "Protege lo que más quieres", "Pólizas", "Seguros para ti")
    encabezado += subtitulo(azar, "Elige la protección ideal", "Tus pólizas contratadas", "Asegura tu patrimonio")
    return encabezado + f'<div class="fila-tarjetas">{tarjetas_de_polizas}</div>' + cotizador


def contenido_documentos(azar: GeneradorAleatorio) -> str:
    """Carga de documentos: zona para arrastrar archivos y lista de documentos con su estado."""
    instruccion = azar.elegir("Arrastra tus archivos aquí", "Selecciona un archivo", "Sube tus documentos")
    formatos = azar.elegir("PDF, JPG o PNG, máximo 10 MB", "Formatos permitidos: PDF, JPG")
    zona_de_carga = (f'<div class="zona"><div class="icono grande-icono">⇪</div><div>{instruccion}</div>'
                     f'<div class="sec">{formatos}</div>'
                     + boton(azar.elegir("Examinar", "Seleccionar archivo", "Subir"), principal=False) + "</div>")
    documentos = azar.elegir_varios(["Identificación oficial (INE)", "Comprobante de domicilio", "Escrituras",
                                     "Estado de cuenta", "Comprobante de ingresos", "Acta de nacimiento",
                                     "Avalúo", "CURP"], 3, 6)
    estados = ["Cargado", "Pendiente", "En revisión", "Rechazado", "Aprobado"]
    filas = ""
    for documento in documentos:
        progreso = azar.elegir(100, 100, azar.entero(10, 95))
        estado = azar.elegir(*estados)
        filas += (f'<div class="fila"><span class="icono">📄</span><div class="crece"><div>{documento}</div>'
                  f'<div class="progreso"><span style="width:{progreso}%"></span></div></div>'
                  f'<span class="chip">{estado}</span></div>')
    encabezado = titulo(azar, "Documentos", "Carga de documentos", "Expediente", "Documentos de tu solicitud",
                        "Mis documentos")
    encabezado += subtitulo(azar, "Solicitud de crédito hipotecario", "Completa tu expediente",
                            "Adjunta los documentos requeridos")
    indicador_pasos = pasos(azar, ["Solicitud", "Documentos", "Revisión"]) if azar.con_probabilidad(0.3) else ""
    zona = zona_de_carga if azar.con_probabilidad(0.8) else ""
    accion = boton(azar.elegir("Enviar documentos", "Guardar", "Continuar"))
    return encabezado + indicador_pasos + zona + f'<div class="lista">{filas}</div>' + accion


def contenido_alta_cliente(azar: GeneradorAleatorio) -> str:
    """Alta de cliente: formulario de datos personales, contacto y domicilio."""
    campos_por_grupo = {
        "Datos personales": [
            campo(azar, "Nombre(s)", azar.elegir(*NOMBRES)),
            campo(azar, "Apellido paterno", azar.elegir(*APELLIDOS)),
            campo(azar, "Apellido materno", azar.elegir(*APELLIDOS)),
            campo(azar, "Fecha de nacimiento", azar.fecha()),
            campo(azar, azar.elegir("CURP", "RFC"), "XXXX000000XXXXXX00"),
            campo(azar, "Género", "", True),
        ],
        "Contacto": [
            campo(azar, "Correo electrónico", "correo@ejemplo.com"),
            campo(azar, "Teléfono celular", "55 " + str(azar.entero(10**7, 10**8))),
        ],
        "Domicilio": [
            campo(azar, "Calle y número"),
            campo(azar, "Código postal", str(azar.entero(10000, 99999))),
            campo(azar, "Colonia", "", True),
            campo(azar, azar.elegir("Ciudad", "Municipio", "Estado"), "", True),
        ],
    }
    formulario = ""
    for grupo in azar.elegir_varios(list(campos_por_grupo), 1, 3):
        campos = azar.elegir_varios(campos_por_grupo[grupo], 2, 6)
        formulario += f'<h2>{grupo}</h2><div class="columnas">{"".join(campos)}</div>'
    encabezado = titulo(azar, "Alta de cliente", "Registro", "Abre tu cuenta", "Nuevo cliente", "Crea tu cuenta")
    if azar.con_probabilidad(0.6):
        encabezado += pasos(azar, ["Datos personales", "Domicilio", "Confirmación"])
    if azar.con_probabilidad(0.5):
        formulario += casilla("Acepto el aviso de privacidad")
    formulario += boton(azar.elegir("Continuar", "Registrar", "Crear cuenta", "Siguiente"))
    return encabezado + tarjeta(formulario)


# Tipo de pantalla -> función que arma su contenido
CONTENIDO_POR_TIPO = {
    "login": contenido_login, "cuentas": contenido_cuentas, "transferencia": contenido_transferencia,
    "pago_servicios": contenido_pago_servicios, "prestamo": contenido_prestamo, "seguros": contenido_seguros,
    "documentos": contenido_documentos, "alta_cliente": contenido_alta_cliente,
}


# --------------------------------------------------------------------------
# Estilo visual: valores al azar en variables CSS y una hoja de estilos fija
# --------------------------------------------------------------------------
def elegir_estilo(azar: GeneradorAleatorio, dispositivo: str) -> dict[str, str]:
    """Colores, tipografía y forma según el estilo (alta, front o boceto)."""
    tono = azar.entero(0, 359)
    modo_oscuro = azar.estilo == "alta" and azar.con_probabilidad(0.25)
    if azar.estilo == "alta":  # diseño final: color principal al azar, modo claro u oscuro
        primario = f"hsl({tono},{azar.entero(45, 85)}%,{azar.entero(35, 50)}%)"
        if modo_oscuro:
            fondo, superficie, texto, secundario, borde = "#121418", "#1d2027", "#eceef2", "#9aa1ad", "#2c313b"
        else:
            fondo = azar.elegir("#f4f6f9", "#ffffff", "#f7f5f2", f"hsl({tono},30%,97%)")
            superficie, texto, secundario, borde = "#ffffff", "#1d2330", "#6b7280", "#e3e6eb"
        sombra = azar.elegir("0 2px 8px rgba(0,0,0,.08)", "0 6px 20px rgba(0,0,0,.10)", "none")
        fuente = azar.elegir(*FUENTES)
        tipo_borde = "solid"
    elif azar.estilo == "front":  # maqueta en grises
        primario, fondo, superficie, texto, secundario, borde = "#7a7a7a", "#f2f2f2", "#ffffff", "#333", "#888", "#c8c8c8"
        sombra = "none"
        fuente = azar.elegir("DejaVu Sans", "FreeSans", "Inter")
        tipo_borde = "solid"
    else:  # boceto: trazo negro, bordes punteados y fuente monoespaciada
        primario, fondo, superficie, texto, secundario, borde = "#222", "#fffef8", "#fffef8", "#111", "#444", "#222"
        sombra = "none"
        fuente = azar.elegir("Courier 10 Pitch", "FreeMono", "DejaVu Sans Mono")
        tipo_borde = "dashed"
    es_boceto = azar.estilo == "boceto"
    radio = azar.elegir(0, 4, 8, 12, 16) if not es_boceto else 2
    tamano_letra = azar.entero(13, 16) if dispositivo != "escritorio" else azar.entero(13, 15)
    barra_superior_de_color = azar.estilo == "alta" and azar.con_probabilidad(0.5)
    logo_redondo = azar.con_probabilidad(0.5)
    margen_principal = azar.entero(14, 32)
    boton_ancho_completo = azar.con_probabilidad(0.4)
    encabezados_en_mayusculas = azar.elegir("uppercase", "none")
    columnas_rejilla = 2 if dispositivo == "movil" else azar.elegir(3, 4, 5)
    return {
        "fuente": f"'{fuente}'", "tamano-letra": f"{tamano_letra}px",
        "primario": primario, "fondo": fondo, "superficie": superficie, "texto": texto,
        "secundario": secundario, "borde": borde, "tipo-borde": tipo_borde, "radio": f"{radio}px", "sombra": sombra,
        "fondo-superior": primario if barra_superior_de_color else superficie,
        "texto-superior": "#fff" if azar.estilo == "alta" else texto,
        "radio-logo": "50%" if logo_redondo else "6px",
        "margen-principal": f"{margen_principal}px",
        "fondo-boton": primario if not es_boceto else "transparent",
        "texto-boton": "#fff" if not es_boceto else "#111",
        "ancho-boton": "100%" if boton_ancho_completo else "auto",
        "subrayado": "underline" if es_boceto else "none",
        "mayusculas-tabla": encabezados_en_mayusculas,
        "radio-icono": "50%" if not es_boceto else "0",
        "fondo-icono": borde if not es_boceto else "transparent",
        "fondo-entrada": fondo if not es_boceto else "transparent",
        "columnas-rejilla": str(columnas_rejilla),
        "columnas-formulario": "1" if dispositivo == "movil" else "2",
    }


# Hoja de estilos común a todas las pantallas; los valores que cambian vienen de las variables CSS
HOJA_DE_ESTILOS = """
* { box-sizing: border-box; margin: 0 }
body { font-family: var(--fuente); font-size: var(--tamano-letra); background: var(--fondo); color: var(--texto) }
.app { display: flex; min-height: 100vh } .app.vertical { flex-direction: column }
.lateral { width: 220px; background: var(--superficie); border-right: 1px var(--tipo-borde) var(--borde); padding: 18px 12px }
.superior { display: flex; align-items: center; gap: 18px; padding: 12px 20px; background: var(--fondo-superior);
            color: var(--texto-superior); border-bottom: 1px var(--tipo-borde) var(--borde) }
.inferior { position: fixed; bottom: 0; left: 0; right: 0; display: flex; justify-content: space-around; padding: 8px 0;
            background: var(--superficie); border-top: 1px var(--tipo-borde) var(--borde); font-size: .75em }
.inferior div { text-align: center }
.marca { display: flex; align-items: center; gap: 8px; font-weight: 700; margin-bottom: 14px } .marca.sin-margen { margin: 0 }
.logo { width: 28px; height: 28px; border-radius: var(--radio-logo); background: var(--primario); color: #fff; display: grid;
        place-items: center; font-size: 12px; border: 1px var(--tipo-borde) var(--borde) }
.menu div { padding: 8px 10px; border-radius: var(--radio); color: var(--secundario) }
.menu div.sel { background: var(--borde); color: var(--texto) }
.principal { flex: 1; padding: var(--margen-principal); padding-bottom: 70px }
h1 { font-size: 1.6em; margin-bottom: 6px } h2 { font-size: 1.1em; margin: 18px 0 8px } h3 { font-size: 1em; margin: 6px 0 }
.sub, .sec { color: var(--secundario); font-size: .9em } .sub { margin-bottom: 14px }
.tarjeta { background: var(--superficie); border: 1px var(--tipo-borde) var(--borde); border-radius: var(--radio);
           padding: 16px; margin: 10px 0; box-shadow: var(--sombra) }
.tarjeta.angosta { max-width: 400px; margin: 30px auto }
.fila-tarjetas { display: flex; flex-wrap: wrap; gap: 12px } .fila-tarjetas .tarjeta { flex: 1; min-width: 150px }
.saldo .grande, .grande { font-size: 1.7em; font-weight: 700; margin: 4px 0 }
.campo { margin: 10px 0 } .campo label { display: block; font-size: .85em; color: var(--secundario); margin-bottom: 4px }
.entrada { border: 1px var(--tipo-borde) var(--borde); border-radius: var(--radio); padding: 9px 10px;
           background: var(--fondo-entrada); display: flex; justify-content: space-between; min-height: 38px }
button { font: inherit; border-radius: var(--radio); padding: 10px 18px; margin: 12px 8px 4px 0;
         border: 1px var(--tipo-borde) var(--primario) }
button.prim { background: var(--fondo-boton); color: var(--texto-boton); width: var(--ancho-boton) }
button.secu { background: transparent; color: var(--primario) }
a { color: var(--primario); display: block; margin-top: 10px; text-decoration: var(--subrayado) }
table { width: 100%; border-collapse: collapse; background: var(--superficie) }
th, td { text-align: left; padding: 8px; border-bottom: 1px var(--tipo-borde) var(--borde) }
th { font-size: .8em; color: var(--secundario); text-transform: var(--mayusculas-tabla) }
.lista .fila { display: flex; align-items: center; gap: 10px; padding: 10px 4px; border-bottom: 1px var(--tipo-borde) var(--borde) }
.crece { flex: 1 }
.icono { width: 32px; height: 32px; border-radius: var(--radio-icono); display: grid; place-items: center;
         background: var(--fondo-icono); border: 1px var(--tipo-borde) var(--borde) }
.grande-icono { width: 46px; height: 46px; font-size: 22px } .seguro .icono { margin-bottom: 4px }
.rejilla { display: grid; grid-template-columns: repeat(var(--columnas-rejilla), 1fr); gap: 10px }
.servicio { background: var(--superficie); border: 1px var(--tipo-borde) var(--borde); border-radius: var(--radio);
            padding: 14px 8px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 6px }
.filtros { display: flex; gap: 16px; flex-wrap: wrap } .filtros > * { flex: 1; min-width: 110px }
.pasos { display: flex; gap: 8px; margin: 10px 0 6px }
.paso { flex: 1; font-size: .8em; color: var(--secundario); border-top: 3px var(--tipo-borde) var(--borde); padding-top: 6px }
.paso.activo { border-color: var(--primario); color: var(--texto) } .paso b { margin-right: 6px }
.deslizador, .progreso { height: 6px; background: var(--borde); border-radius: 3px; margin: 10px 0 }
.deslizador span, .progreso span { display: block; height: 100%; background: var(--primario); border-radius: 3px }
.chip { display: inline-block; font-size: .75em; padding: 3px 8px; border-radius: 10px;
        border: 1px var(--tipo-borde) var(--borde); margin: 6px 0 }
.zona { border: 2px dashed var(--borde); border-radius: var(--radio); padding: 26px; text-align: center; margin: 12px 0 }
.dividido { display: flex; gap: 20px; align-items: center }
.ilustracion { flex: 1; min-height: 320px; border-radius: var(--radio); background: var(--borde); display: grid; place-items: center }
.columnas { display: grid; grid-template-columns: repeat(var(--columnas-formulario), 1fr); column-gap: 16px }
.check { display: flex; align-items: center; gap: 8px; margin: 10px 0; font-size: .9em }
.caja { width: 16px; height: 16px; border: 1px var(--tipo-borde) var(--secundario); border-radius: 3px }
.barra { display: inline-block; height: 9px; border-radius: 4px; background: #d0d0d0; vertical-align: middle }
.flecha { color: var(--secundario) }
"""


# --------------------------------------------------------------------------
# Página completa: marco de la aplicación + contenido + estilo
# --------------------------------------------------------------------------
def armar_pagina_html(azar: GeneradorAleatorio, tipo_pantalla: str, dispositivo: str) -> str:
    """Documento HTML completo de una pantalla: marco de la aplicación, contenido del tipo y estilo."""
    banco = azar.elegir(*BANCOS)
    iniciales = "".join(palabra[0] for palabra in banco.split()[:2])
    marca = f'<div class="marca"><span class="logo">{iniciales}</span>{banco}</div>'
    modulos = azar.elegir_varios(MODULOS_DEL_MENU, 4, 8 if dispositivo == "escritorio" else 5)
    modulo_actual = azar.posicion(len(modulos))
    menu = "".join(f'<div class="{"sel" if numero == modulo_actual else ""}">{modulo}</div>'
                   for numero, modulo in enumerate(modulos))
    contenido = CONTENIDO_POR_TIPO[tipo_pantalla](azar)  # lo que distingue a cada clase

    # Navegación: ninguna, menú lateral (escritorio), barra inferior (móvil) o barra superior.
    # El menú nombra módulos de todos los tipos para que su texto no delate la clase.
    sin_navegacion = azar.con_probabilidad(0.6 if tipo_pantalla == "login" else 0.12)
    if sin_navegacion:
        cuerpo = f'<div class="app vertical"><div class="principal">{marca}{contenido}</div></div>'
    elif dispositivo == "escritorio" and azar.con_probabilidad(0.6):
        cuerpo = (f'<div class="app"><aside class="lateral">{marca}<div class="menu">{menu}</div></aside>'
                  f'<main class="principal">{contenido}</main></div>')
    elif dispositivo == "movil" and azar.con_probabilidad(0.6):
        botones = "".join(f'<div>{azar.elegir("◻", "○", "◇")}<br>{modulo}</div>' for modulo in modulos[:5])
        marca_compacta = marca.replace("marca", "marca sin-margen")
        cuerpo = (f'<div class="app vertical"><div class="superior">{marca_compacta}</div>'
                  f'<main class="principal">{contenido}</main><nav class="inferior">{botones}</nav></div>')
    else:
        enlaces = "".join(f"<span>{modulo}</span>" for modulo in modulos) if dispositivo != "movil" else "<span>☰</span>"
        usuario = azar.elegir(*NOMBRES)
        cuerpo = (f'<div class="app vertical"><div class="superior">{marca}'
                  f'<div style="display:flex;gap:16px;flex:1;justify-content:flex-end">{enlaces}<span>{usuario}</span></div>'
                  f'</div><main class="principal">{contenido}</main></div>')

    # Estilo al final: los valores elegidos se escriben como variables CSS (--primario: ...)
    variables = "; ".join(f"--{nombre}: {valor}" for nombre, valor in elegir_estilo(azar, dispositivo).items())
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f"<style>:root {{ {variables} }}{HOJA_DE_ESTILOS}</style></head><body>{cuerpo}</body></html>")
