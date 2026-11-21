"""
Patrones de pantalla (base de conocimiento del componente visual)

El clasificador visual sólo reconoce el TIPO de pantalla; no lee sus campos. Para pasar del
tipo a servicios concretos, cada tipo tiene un patrón: las operaciones que una pantalla de
ese tipo necesita normalmente, identificadas por método y endpoint del RAML. Cada operación
se busca en el inventario por método y endpoint exactos (no por texto); si no está, es una brecha.
"""
from __future__ import annotations

# Nombre legible de cada tipo de pantalla para la respuesta
NOMBRE_DE_PANTALLA = {
    "login": "inicio de sesión", "cuentas": "resumen de cuentas y movimientos",
    "transferencia": "transferencia", "pago_servicios": "pago de servicios",
    "prestamo": "préstamo", "seguros": "seguros", "documentos": "carga de documentos",
    "alta_cliente": "alta de cliente",
}

# Patrones de pantalla: (operación que necesita la pantalla, método, endpoint del RAML)
PATRONES_DE_PANTALLA: dict[str, list[tuple[str, str, str]]] = {
    "login": [
        ("Iniciar sesión con usuario y contraseña", "POST", "/autenticacion/sesiones"),
        ("Verificar el código de segundo factor", "POST", "/autenticacion/verificacion-segundo-factor"),
        ("Cerrar sesión", "DELETE", "/autenticacion/sesiones/{idSesion}"),
    ],
    "cuentas": [
        ("Listar las cuentas del usuario", "GET", "/cuentas"),
        ("Mostrar el saldo de una cuenta", "GET", "/cuentas/{idCuenta}"),
        ("Mostrar los movimientos", "GET", "/cuentas/{idCuenta}/movimientos"),
        ("Generar el estado de cuenta", "POST", "/cuentas/{idCuenta}/estado-cuenta"),
    ],
    "transferencia": [
        ("Elegir la cuenta de origen", "GET", "/cuentas"),
        ("Elegir un beneficiario", "GET", "/clientes/{idCliente}/beneficiarios"),
        ("Ordenar la transferencia", "POST", "/transferencias"),
        ("Generar el comprobante", "POST", "/transferencias/{idTransferencia}/comprobante"),
    ],
    "pago_servicios": [
        ("Listar los convenios de pago", "GET", "/pagos/convenios"),
        ("Validar la referencia según el convenio", "GET", "/pagos/convenios/{idConvenio}"),
        ("Pagar el servicio", "POST", "/pagos"),
        ("Mostrar los pagos recientes", "GET", "/pagos"),
    ],
    "prestamo": [
        ("Mostrar el préstamo y su saldo", "GET", "/prestamos/{idPrestamo}"),
        ("Mostrar la tabla de amortización", "GET", "/prestamos/{idPrestamo}/amortizacion"),
        ("Pagar una mensualidad", "POST", "/prestamos/{idPrestamo}/pagos"),
        ("Solicitar un préstamo", "POST", "/prestamos"),
    ],
    "seguros": [
        ("Listar pólizas y cotizaciones", "GET", "/seguros"),
        ("Cotizar un seguro", "POST", "/seguros"),
        ("Contratar la póliza", "POST", "/seguros/{idSeguro}/contratacion"),
        ("Cancelar una póliza", "POST", "/seguros/{idSeguro}/cancelacion"),
    ],
    "documentos": [
        ("Listar los documentos requeridos del expediente", "GET", "/hipotecas/{idHipoteca}/documentos"),
        ("Registrar un documento aportado", "POST", "/hipotecas/{idHipoteca}/documentos"),
        ("Mostrar el estado de la solicitud", "GET", "/hipotecas/{idHipoteca}"),
    ],
    "alta_cliente": [
        ("Registrar al cliente", "POST", "/clientes"),
        ("Registrar el consentimiento de privacidad", "PUT", "/clientes/{idCliente}/consentimientos"),
        ("Abrir su cuenta", "POST", "/cuentas"),
    ],
}


def buscar_operaciones_en_inventario(tipo_pantalla: str, servicios: list[dict]) -> list[tuple[str, dict | None]]:
    """(operación, servicio que la cubre o None si es una brecha) para cada operación del patrón."""
    # Índice del inventario por (método, endpoint): ("POST", "/transferencias") -> servicio
    servicio_por_metodo_y_endpoint = {(servicio["metodo"], servicio["endpoint"]): servicio for servicio in servicios}
    # .get devuelve None si la operación no existe en el inventario: es una brecha
    return [(descripcion, servicio_por_metodo_y_endpoint.get((metodo, endpoint)))
            for descripcion, metodo, endpoint in PATRONES_DE_PANTALLA[tipo_pantalla]]
