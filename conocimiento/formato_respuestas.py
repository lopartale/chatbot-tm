"""
Formato de las respuestas

Textos que el asistente muestra sobre los servicios del inventario. Sólo da formato: no
busca ni decide nada.
"""
from __future__ import annotations

PASOS_ANTE_BRECHA = ("\n  1) Reformular con otros términos o revisar servicios similares."
                     "\n  2) Verificar si un servicio existente puede extenderse."
                     "\n  3) Registrar la brecha con el equipo de arquitectura para evaluar un servicio nuevo.")


def operacion_http(servicio: dict) -> str:
    """"GET /cuentas/{idCuenta}"."""
    return f"{servicio['metodo']} {servicio['endpoint']}"


def resumen_servicio(servicio: dict) -> str:
    """Una línea por servicio: "SRV-015 Consulta los datos de una cuenta [GET /cuentas/{idCuenta}]"."""
    return f"{servicio['id']} {servicio['nombre']} [{operacion_http(servicio)}]"


def contrato_servicio(servicio: dict) -> str:
    """Contrato completo: operación, URL, descripción, entrada, salida y controles."""
    lineas = [f"{servicio['id']} - {servicio['nombre']} (dominio {servicio['dominio']})",
              f"  Método/endpoint: {operacion_http(servicio)}",
              f"  URL: {servicio['url']}",
              f"  Descripción: {servicio['descripcion']}"]
    if servicio.get("contexto"):
        lineas.append(f"  Recurso: {servicio['contexto']}")
    lineas += [f"  Entrada: {servicio['entrada']}", f"  Salida: {servicio['salida']}"]
    if servicio.get("controles"):
        lineas.append("  Controles: " + "; ".join(servicio["controles"]))
    return "\n".join(lineas)
