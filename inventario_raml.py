"""
Lector del inventario de servicios REST a partir de un archivo RAML 1.0.

Cada combinación recurso + método HTTP del RAML (por ejemplo ``GET /clientes`` o
``POST /transferencias/{idTransferencia}/cancelacion``) se convierte en un
"servicio" del inventario con los mismos campos que usa el chatbot:

    id, nombre, dominio, metodo, endpoint, url, descripcion, contexto,
    palabras_clave, valores_enum, entrada, salida, controles

El lector resuelve lo necesario para describir bien cada operación:
  * ``!include`` de fragmentos y archivos relativos al RAML,
  * ``resourceTypes`` (con parámetros ``<<param>>``) y ``traits`` (``is:``),
  * ``types`` con herencia, propiedades opcionales (``campo?``) y arreglos,
  * parámetros de ruta, de consulta, encabezados, cuerpo y respuestas 2xx.

Sólo depende de PyYAML (RAML 1.0 es un subconjunto de YAML).

Uso rápido para revisar la conversión:
    python inventario_raml.py data/plataforma_servicios_bancarios.raml
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

METODOS_HTTP = ("get", "post", "put", "patch", "delete")

# Verbos en español que se agregan como palabras clave según el método HTTP,
# para que "dar de alta", "cambiar" o "borrar" encuentren la operación correcta.
VERBOS_POR_METODO = {
    "GET": ["consultar", "obtener"],
    "POST": ["crear", "registrar", "solicitar"],
    "PUT": ["actualizar", "reemplazar", "registrar"],
    "PATCH": ["actualizar", "modificar", "cambiar"],
    "DELETE": ["eliminar", "borrar"],
}

# Descripción legible de los traits que cambian el comportamiento de la operación
CONTROLES_POR_TRAIT = {
    "idempotente": "idempotencia (encabezado X-Clave-Idempotencia)",
    "confirmacionFuerte": "segundo factor (encabezado X-Factor-Confirmacion)",
}

# Parámetros de paginación: están en casi todas las consultas y no distinguen un servicio de otro
PARAMETROS_PAGINACION = {"pageSize", "paginationKey"}

# Traits que sólo documentan paginación o errores comunes; no son "controles"
TRAITS_INFORMATIVOS = {"rastreable", "paginacion"}


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def _mezclar(base: Any, extra: Any) -> Any:
    """Mezcla recursiva de diccionarios; en cualquier otro caso gana ``extra``."""
    if isinstance(base, dict) and isinstance(extra, dict):
        resultado = dict(base)
        for clave, valor in extra.items():
            resultado[clave] = _mezclar(base[clave], valor) if clave in base else valor
        return resultado
    return extra


def _sustituir(obj: Any, parametros: dict[str, Any]) -> Any:
    """Reemplaza los parámetros ``<<nombre>>`` de un resourceType."""
    if isinstance(obj, str):
        return re.sub(r"<<\s*([\w-]+)[^>]*>>",
                      lambda m: str(parametros.get(m.group(1), m.group(0))), obj)
    if isinstance(obj, dict):
        return {k: _sustituir(v, parametros) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sustituir(v, parametros) for v in obj]
    return obj


def _separar_camel(texto: str) -> str:
    """'idCuentaOrigen' -> 'id cuenta origen'; 'en_evaluacion' -> 'en evaluacion'."""
    texto = re.sub(r"(?<=[a-záéíóúñ0-9])(?=[A-Z])", " ", texto)
    return texto.replace("_", " ").replace("-", " ").lower().strip()


def _nombre_trait(entrada: Any) -> str | None:
    """Un trait en ``is:`` puede ser texto o ``{nombre: parámetros}``."""
    if isinstance(entrada, str):
        return entrada
    if isinstance(entrada, dict) and entrada:
        return next(iter(entrada))
    return None


def _lista(valor: Any) -> list:
    if valor is None:
        return []
    return valor if isinstance(valor, list) else [valor]


def _sin_repetidos(piezas) -> list[str]:
    """Normaliza cada pieza con _separar_camel y conserva la primera aparición."""
    return list(dict.fromkeys(limpia for pieza in piezas if (limpia := _separar_camel(str(pieza)))))


def _sin_opcional(nombre: str) -> tuple[str, bool]:
    """('campo?') -> ('campo', True)."""
    return (nombre[:-1], True) if nombre.endswith("?") else (nombre, False)


# --------------------------------------------------------------------------
# Intérprete del documento RAML
# --------------------------------------------------------------------------
class _DocumentoRAML:
    def __init__(self, datos: dict):
        self.datos = datos
        self.tipos: dict = datos.get("types") or {}
        self.traits: dict = datos.get("traits") or {}
        self.resource_types: dict = datos.get("resourceTypes") or {}
        self.base_uri: str = datos.get("baseUri", "")
        self.seguridad_global = datos.get("securedBy")
        # mediaType puede ser un tipo o una lista de tipos
        self.media_types: list[str] = _lista(datos.get("mediaType")) or ["application/json"]

    # ---- tipos y propiedades ----
    def propiedades(self, definicion: Any, _vistos: tuple = ()) -> list[dict]:
        """Lista ordenada de propiedades (con herencia) de un tipo o definición inline."""
        if isinstance(definicion, str):
            nombre = definicion.strip()
            if nombre in _vistos or nombre not in self.tipos:
                return []
            return self.propiedades(self.tipos[nombre], _vistos + (nombre,))
        if not isinstance(definicion, dict):
            return []
        if definicion.get("type") == "array":
            return self.propiedades(definicion.get("items"), _vistos)
        acumuladas: dict[str, dict] = {}
        for padre in _lista(definicion.get("type")):
            if isinstance(padre, str):
                for p in self.propiedades(padre, _vistos):
                    acumuladas[p["nombre"]] = p
        for clave, valor in (definicion.get("properties") or {}).items():
            nombre, opcional = _sin_opcional(str(clave))
            detalle = valor if isinstance(valor, dict) else {"type": valor}
            tipo = detalle.get("type", "string")
            if isinstance(tipo, dict):
                tipo = "object"
            elif isinstance(tipo, list):
                tipo = "|".join(str(t) for t in tipo)
            if tipo == "array":
                items = detalle.get("items")
                tipo = f"{self.forma(items) if items else 'object'}[]"
            acumuladas[nombre] = {
                "nombre": nombre,
                "tipo": str(tipo),
                "requerida": bool(detalle.get("required", not opcional)),
                "descripcion": detalle.get("description", ""),
                "enum": [str(e) for e in _lista(detalle.get("enum"))],
            }
        return list(acumuladas.values())

    def _campos_texto(self, props: list[dict]) -> str:
        partes = []
        for p in props:
            etiqueta = p["tipo"]
            if p["enum"]:
                etiqueta += " (" + "|".join(p["enum"]) + ")"
            partes.append(f"{p['nombre']}{'' if p['requerida'] else '?'}: {etiqueta}")
        return ", ".join(partes)

    def forma(self, definicion: Any) -> str:
        """Descripción corta del contenido de un cuerpo (tipo + campos)."""
        if isinstance(definicion, str):
            definicion = {"type": definicion}
        if not isinstance(definicion, dict):
            return "sin cuerpo"
        tipo = definicion.get("type")
        if tipo == "array":
            return "lista de " + self.forma(definicion.get("items") or "object")
        if isinstance(tipo, str) and tipo.endswith("[]"):
            return f"lista de {tipo[:-2]}"
        props = self.propiedades(definicion)
        nombre = tipo if isinstance(tipo, str) and tipo in self.tipos else None
        if props:
            campos = "{" + self._campos_texto(props) + "}"
            return f"{nombre} {campos}" if nombre else campos
        return nombre or (tipo if isinstance(tipo, str) else "objeto")

    # ---- resourceTypes y traits ----
    def _resource_type(self, nodo: dict) -> dict:
        aplicado = nodo.get("type")
        if isinstance(aplicado, str):
            nombre, parametros = aplicado, {}
        elif isinstance(aplicado, dict) and aplicado:
            nombre = next(iter(aplicado))
            parametros = aplicado[nombre] or {}
        else:
            return {}
        plantilla = self.resource_types.get(nombre) or {}
        return _sustituir(plantilla, parametros)

    def _operacion(self, nodo: dict, metodo: str) -> tuple[dict, list[str]]:
        """Mezcla la operación del resourceType con la del recurso y devuelve
        (operación_resuelta_sin_traits, nombres_de_traits_aplicados)."""
        tipo_recurso = self._resource_type(nodo)
        del_tipo = tipo_recurso.get(metodo) or {}
        propia = nodo.get(metodo) or {}
        traits = [t for t in (_nombre_trait(e) for e in
                  _lista(tipo_recurso.get("is")) + _lista(nodo.get("is"))
                  + _lista(del_tipo.get("is")) + _lista(propia.get("is"))) if t]
        traits = list(dict.fromkeys(traits))
        return _mezclar(del_tipo, propia), traits

    def _con_traits(self, operacion: dict, traits: list[str]) -> dict:
        """Operación con los traits mezclados; lo declarado en la operación tiene prioridad."""
        resuelta: dict = {}
        for nombre in traits:
            trait = {k: v for k, v in (self.traits.get(nombre) or {}).items() if k != "usage"}
            resuelta = _mezclar(resuelta, trait)
        return _mezclar(resuelta, operacion)

    # ---- recorrido de recursos ----
    def servicios(self) -> list[dict]:
        resultado: list[dict] = []
        for clave, nodo in self.datos.items():
            if isinstance(clave, str) and clave.startswith("/") and isinstance(nodo, dict):
                self._recorrer(clave, nodo, [], {}, resultado)
        for i, servicio in enumerate(resultado, 1):
            servicio["id"] = f"SRV-{i:03d}"
        return resultado

    def _recorrer(self, ruta: str, nodo: dict, ancestros: list[dict],
                  uri_params: dict, salida: list[dict]) -> None:
        recursos = ancestros + [{"ruta": ruta, "descripcion": nodo.get("description", "")}]
        uri_params = {**uri_params, **(nodo.get("uriParameters") or {})}
        tipo_recurso = self._resource_type(nodo)
        for metodo in METODOS_HTTP:
            if metodo in nodo or metodo in tipo_recurso:
                salida.append(self._construir_servicio(metodo, recursos, nodo, uri_params))
        for clave, hijo in nodo.items():
            if isinstance(clave, str) and clave.startswith("/") and isinstance(hijo, dict):
                self._recorrer(clave, hijo, recursos, uri_params, salida)

    # ---- construcción de un servicio ----
    def _construir_servicio(self, metodo: str, recursos: list[dict], nodo: dict,
                            uri_params: dict) -> dict:
        sin_traits, traits = self._operacion(nodo, metodo)
        operacion = self._con_traits(sin_traits, traits)
        encabezados_propios = sin_traits.get("headers") or {}

        metodo_http = metodo.upper()
        endpoint = "".join(r["ruta"] for r in recursos)
        segmentos = [s for s in endpoint.split("/") if s]
        literales = [s for s in segmentos if not s.startswith("{")]
        dominio = literales[0]
        es_coleccion = not segmentos[-1].startswith("{")

        descripcion = (operacion.get("description") or "").strip()
        contexto = (recursos[-1]["descripcion"] or "").strip()
        nombre = operacion.get("displayName") or self._nombre_corto(descripcion, metodo_http, endpoint)

        # ---- entradas ----
        partes_entrada: list[str] = []
        parametros_ruta = [p for p in segmentos if p.startswith("{")]
        if parametros_ruta:
            detalle = []
            for p in parametros_ruta:
                clave = p.strip("{}")
                info = uri_params.get(clave)
                tipo = info.get("type", "string") if isinstance(info, dict) else (info or "string")
                detalle.append(f"{clave}: {tipo}")
            partes_entrada.append("Ruta: " + ", ".join(detalle))

        props_consulta = self.propiedades(operacion.get("queryString"))
        if props_consulta:
            partes_entrada.append("Consulta: " + self._campos_texto(props_consulta))

        encabezados = []
        for clave, valor in (operacion.get("headers") or {}).items():
            nombre_h, opcional = _sin_opcional(str(clave))
            detalle_h = valor if isinstance(valor, dict) else {}
            requerido = bool(detalle_h.get("required", not opcional))
            # los encabezados opcionales que sólo aporta un trait común se omiten
            if requerido or clave in encabezados_propios:
                encabezados.append(f"{nombre_h} ({'obligatorio' if requerido else 'opcional'})")
        if encabezados:
            partes_entrada.append("Encabezados: " + ", ".join(encabezados))

        cuerpo = self._cuerpo(operacion.get("body"))
        cuerpo_props: list[dict] = []
        if cuerpo is not None:
            partes_entrada.append("Cuerpo: " + self.forma(cuerpo))
            cuerpo_props = self.propiedades(cuerpo)
        entrada = "; ".join(partes_entrada) if partes_entrada else "Sin parámetros"

        # ---- salidas (sólo respuestas exitosas) ----
        partes_salida = []
        respuestas = operacion.get("responses") or {}
        for codigo in sorted((c for c in respuestas if str(c).isdigit()), key=lambda c: int(c)):
            if not 200 <= int(codigo) < 300:
                continue
            respuesta = respuestas[codigo] or {}
            cuerpo_resp = self._cuerpo(respuesta.get("body"))
            texto = f"{codigo}"
            if cuerpo_resp is not None:
                texto += f" {self.forma(cuerpo_resp)}"
            if respuesta.get("description"):
                texto += f" - {respuesta['description']}"
            partes_salida.append(texto)
        salida = "; ".join(partes_salida) if partes_salida else "Sin cuerpo de respuesta"

        # ---- controles de seguridad / negocio ----
        controles = [CONTROLES_POR_TRAIT[t] for t in traits if t in CONTROLES_POR_TRAIT]
        seguridad = operacion.get("securedBy", nodo.get("securedBy", self.seguridad_global))
        if isinstance(seguridad, list) and None in seguridad:
            controles.insert(0, "no requiere token de acceso (endpoint de acceso)")
        elif seguridad:
            controles.insert(0, "token Bearer en el encabezado Authorization")

        return {
            "id": "",                      # se asigna al terminar el recorrido
            "nombre": nombre,
            "dominio": dominio,
            "metodo": metodo_http,
            "endpoint": endpoint,
            "url": self.base_uri.rstrip("/") + endpoint if self.base_uri else endpoint,
            "descripcion": descripcion,
            "contexto": contexto,
            "palabras_clave": self._palabras_clave(
                dominio, literales, metodo_http, es_coleccion,
                [p.strip("{}") for p in parametros_ruta], props_consulta,
                cuerpo_props, cuerpo),
            "valores_enum": self._valores_enum([*props_consulta, *cuerpo_props]),
            "entrada": entrada,
            "salida": salida,
            "controles": controles,
        }

    def _cuerpo(self, body: Any) -> Any:
        """Contenido del cuerpo para el tipo de medio por omisión (o el primero)."""
        if not isinstance(body, dict) or not body:
            return None
        for media_type in self.media_types:
            if media_type in body:
                return body[media_type] or {"type": "object"}
        primero = next(iter(body.values()))
        if isinstance(primero, (dict, str)):
            return primero
        # cuerpo declarado directamente (sin tipo de medio) con type/properties
        return body if ("type" in body or "properties" in body) else None

    @staticmethod
    def _nombre_corto(descripcion: str, metodo: str, endpoint: str, maximo: int = 70) -> str:
        if not descripcion:
            return f"{metodo} {endpoint}"
        primera = re.split(r"[,;.:]| - ", descripcion, maxsplit=1)[0].strip()
        if len(primera) > maximo:
            primera = primera[:maximo].rsplit(" ", 1)[0] + "…"
        return primera or f"{metodo} {endpoint}"

    def _palabras_clave(self, dominio: str, literales: list[str], metodo: str,
                        es_coleccion: bool, params_ruta: list[str],
                        props_consulta: list[dict], props_cuerpo: list[dict],
                        cuerpo: Any) -> list[str]:
        piezas: list[str] = [dominio, *literales, *VERBOS_POR_METODO.get(metodo, [])]
        if metodo == "GET" and es_coleccion:
            piezas.append("listar")
        piezas += params_ruta
        piezas += [p["nombre"] for p in [*props_consulta, *props_cuerpo]
                   if p["nombre"] not in PARAMETROS_PAGINACION]
        if isinstance(cuerpo, dict):
            tipo = cuerpo.get("type")
            if isinstance(tipo, str) and tipo in self.tipos:
                piezas.append(tipo)
        return _sin_repetidos(piezas)

    @staticmethod
    def _valores_enum(props: list[dict]) -> list[str]:
        """Valores permitidos de los campos ('carta_saldo' -> 'carta saldo').

        Se separan de las palabras clave porque describen opciones de un campo, no la
        operación: el valor 'carta_saldo' de GET /documentos no hace de él el servicio de saldo.
        """
        return _sin_repetidos(v for p in props for v in p["enum"] if v not in ("True", "False"))


def _cargar_yaml(texto: str, carpeta: Path) -> Any:
    """yaml.safe_load con soporte para ``!include``, resuelto relativo a ``carpeta``.

    Los fragmentos .raml/.yaml/.yml se interpretan; cualquier otro archivo
    (ejemplos .json, documentación .md) se incluye como texto.
    """
    class Cargador(yaml.SafeLoader):
        pass

    def incluir(cargador: yaml.SafeLoader, nodo: yaml.Node) -> Any:
        destino = carpeta / cargador.construct_scalar(nodo)
        contenido = destino.read_text(encoding="utf-8")
        if destino.suffix.lower() in (".raml", ".yaml", ".yml"):
            return _cargar_yaml(contenido, destino.parent)
        return contenido

    Cargador.add_constructor("!include", incluir)
    return yaml.load(texto, Loader=Cargador)


# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------
def cargar_inventario_raml(ruta: str | Path) -> dict:
    """Lee un archivo RAML 1.0 y devuelve el inventario de servicios.

    Resultado: {"titulo", "version", "base_uri", "servicios": [ {...}, ... ]}
    """
    ruta = Path(ruta)
    texto = ruta.read_text(encoding="utf-8")
    if not texto.lstrip().startswith("#%RAML"):
        raise ValueError(f"{ruta.name} no parece un archivo RAML (falta la cabecera '#%RAML 1.0').")
    try:
        datos = _cargar_yaml(texto, ruta.parent)
    except FileNotFoundError as error:
        raise ValueError(f"{ruta.name} incluye un archivo que no existe: {error.filename}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"No se pudo interpretar {ruta.name}: {error}") from error
    if not isinstance(datos, dict):
        raise ValueError(f"{ruta.name} está vacío o tiene un formato inesperado.")

    documento = _DocumentoRAML(datos)
    servicios = documento.servicios()
    if not servicios:
        raise ValueError(f"{ruta.name} no define recursos con métodos HTTP (rutas que inician con '/').")
    return {
        "titulo": datos.get("title", ruta.stem),
        "version": str(datos.get("version", "")),
        "base_uri": documento.base_uri,
        "servicios": servicios,
    }


if __name__ == "__main__":
    destino = sys.argv[1] if len(sys.argv) > 1 else \
        Path(__file__).parent / "data" / "plataforma_servicios_bancarios.raml"
    inventario = cargar_inventario_raml(destino)
    print(f"{inventario['titulo']} {inventario['version']} - {len(inventario['servicios'])} servicios")
    for s in inventario["servicios"]:
        print(f"{s['id']}  {s['dominio']:<15} {s['metodo']:<6} {s['endpoint']}")
