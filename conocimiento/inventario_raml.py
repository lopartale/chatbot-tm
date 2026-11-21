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
    python -m conocimiento.inventario_raml data/plataforma_servicios_bancarios.raml
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
PALABRAS_CLAVE_POR_METODO = {
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


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def _mezclar_diccionarios(base: Any, extra: Any) -> Any:
    """Mezcla recursiva de diccionarios; en cualquier otro caso gana ``extra``."""
    if isinstance(base, dict) and isinstance(extra, dict):
        resultado = dict(base)
        for clave, valor in extra.items():
            resultado[clave] = _mezclar_diccionarios(base[clave], valor) if clave in base else valor
        return resultado
    return extra


def _sustituir_parametros(obj: Any, parametros: dict[str, Any]) -> Any:
    """Reemplaza los parámetros ``<<nombre>>`` de un resourceType."""
    if isinstance(obj, str):
        return re.sub(r"<<\s*([\w-]+)[^>]*>>",
                      lambda m: str(parametros.get(m.group(1), m.group(0))), obj)
    if isinstance(obj, dict):
        return {k: _sustituir_parametros(v, parametros) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sustituir_parametros(v, parametros) for v in obj]
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


def _como_lista(valor: Any) -> list:
    """None -> []; un valor suelto -> [valor]; una lista queda igual."""
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
class DocumentoRAML:
    """Interpreta un RAML ya cargado: tipos, traits, resourceTypes y recursos con sus métodos."""

    def __init__(self, datos: dict):
        self.datos = datos
        self.tipos: dict = datos.get("types") or {}
        self.traits: dict = datos.get("traits") or {}
        self.resource_types: dict = datos.get("resourceTypes") or {}
        self.base_uri: str = datos.get("baseUri", "")
        self.seguridad_global = datos.get("securedBy")
        # mediaType puede ser un tipo o una lista de tipos
        self.media_types: list[str] = _como_lista(datos.get("mediaType")) or ["application/json"]

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
        for padre in _como_lista(definicion.get("type")):
            if isinstance(padre, str):
                for propiedad in self.propiedades(padre, _vistos):  # herencia: primero las del tipo padre
                    acumuladas[propiedad["nombre"]] = propiedad
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
                tipo = f"{self.describir_cuerpo(items) if items else 'object'}[]"
            acumuladas[nombre] = {
                "nombre": nombre,
                "tipo": str(tipo),
                "requerida": bool(detalle.get("required", not opcional)),
                "descripcion": detalle.get("description", ""),
                "enum": [str(e) for e in _como_lista(detalle.get("enum"))],
            }
        return list(acumuladas.values())

    def _describir_campos(self, props: list[dict]) -> str:
        """"campo: tipo, opcional?: tipo (valor1|valor2)"."""
        partes = []
        for propiedad in props:
            etiqueta = propiedad["tipo"]
            if propiedad["enum"]:
                etiqueta += " (" + "|".join(propiedad["enum"]) + ")"
            partes.append(f"{propiedad['nombre']}{'' if propiedad['requerida'] else '?'}: {etiqueta}")
        return ", ".join(partes)

    def describir_cuerpo(self, definicion: Any) -> str:
        """Descripción corta del contenido de un cuerpo (tipo + campos)."""
        if isinstance(definicion, str):
            definicion = {"type": definicion}
        if not isinstance(definicion, dict):
            return "sin cuerpo"
        tipo = definicion.get("type")
        if tipo == "array":
            return "lista de " + self.describir_cuerpo(definicion.get("items") or "object")
        if isinstance(tipo, str) and tipo.endswith("[]"):
            return f"lista de {tipo[:-2]}"
        props = self.propiedades(definicion)
        nombre = tipo if isinstance(tipo, str) and tipo in self.tipos else None
        if props:
            campos = "{" + self._describir_campos(props) + "}"
            return f"{nombre} {campos}" if nombre else campos
        return nombre or (tipo if isinstance(tipo, str) else "objeto")

    # ---- resourceTypes y traits ----
    def _plantilla_de_recurso(self, nodo: dict) -> dict:
        """resourceType aplicado al recurso (type: coleccion o {coleccion: {parametros}}), ya sustituido."""
        aplicado = nodo.get("type")
        if isinstance(aplicado, str):
            nombre, parametros = aplicado, {}
        elif isinstance(aplicado, dict) and aplicado:
            nombre = next(iter(aplicado))
            parametros = aplicado[nombre] or {}
        else:
            return {}
        plantilla = self.resource_types.get(nombre) or {}
        return _sustituir_parametros(plantilla, parametros)

    def _operacion_sin_traits(self, nodo: dict, metodo: str) -> tuple[dict, list[str]]:
        """Mezcla la operación del resourceType con la del recurso y devuelve
        (operación_resuelta_sin_traits, nombres_de_traits_aplicados)."""
        tipo_recurso = self._plantilla_de_recurso(nodo)
        del_tipo = tipo_recurso.get(metodo) or {}
        propia = nodo.get(metodo) or {}
        traits = [t for t in (_nombre_trait(e) for e in
                  _como_lista(tipo_recurso.get("is")) + _como_lista(nodo.get("is"))
                  + _como_lista(del_tipo.get("is")) + _como_lista(propia.get("is"))) if t]
        traits = list(dict.fromkeys(traits))
        return _mezclar_diccionarios(del_tipo, propia), traits

    def _aplicar_traits(self, operacion: dict, traits: list[str]) -> dict:
        """Operación con los traits mezclados; lo declarado en la operación tiene prioridad."""
        resuelta: dict = {}
        for nombre in traits:
            trait = {k: v for k, v in (self.traits.get(nombre) or {}).items() if k != "usage"}
            resuelta = _mezclar_diccionarios(resuelta, trait)
        return _mezclar_diccionarios(resuelta, operacion)

    # ---- recorrido de recursos ----
    def servicios(self) -> list[dict]:
        """Todos los servicios del RAML, numerados SRV-001, SRV-002... en orden de aparición."""
        resultado: list[dict] = []
        for clave, nodo in self.datos.items():
            if isinstance(clave, str) and clave.startswith("/") and isinstance(nodo, dict):
                self._recorrer_recursos(clave, nodo, [], {}, resultado)
        for i, servicio in enumerate(resultado, 1):
            servicio["id"] = f"SRV-{i:03d}"
        return resultado

    def _recorrer_recursos(self, ruta: str, nodo: dict, ancestros: list[dict],
                           parametros_uri: dict, salida: list[dict]) -> None:
        """Agrega a `salida` un servicio por cada método del recurso y baja a sus subrecursos."""
        recursos = ancestros + [{"ruta": ruta, "descripcion": nodo.get("description", "")}]
        # Los parámetros de ruta ({idCuenta}) se heredan de los recursos padre
        parametros_uri = {**parametros_uri, **(nodo.get("uriParameters") or {})}
        tipo_recurso = self._plantilla_de_recurso(nodo)
        for metodo in METODOS_HTTP:
            if metodo in nodo or metodo in tipo_recurso:
                salida.append(self._construir_servicio(metodo, recursos, nodo, parametros_uri))
        for clave, hijo in nodo.items():
            if isinstance(clave, str) and clave.startswith("/") and isinstance(hijo, dict):
                self._recorrer_recursos(clave, hijo, recursos, parametros_uri, salida)

    # ---- construcción de un servicio ----
    def _construir_servicio(self, metodo: str, recursos: list[dict], nodo: dict,
                            parametros_uri: dict) -> dict:
        """Un servicio del inventario a partir de un método HTTP de un recurso del RAML."""
        operacion_sin_traits, traits = self._operacion_sin_traits(nodo, metodo)
        operacion = self._aplicar_traits(operacion_sin_traits, traits)

        metodo_http = metodo.upper()
        endpoint = "".join(recurso["ruta"] for recurso in recursos)   # /cuentas + /{idCuenta}
        segmentos = [segmento for segmento in endpoint.split("/") if segmento]
        literales = [segmento for segmento in segmentos if not segmento.startswith("{")]
        parametros_ruta = [segmento for segmento in segmentos if segmento.startswith("{")]
        dominio = literales[0]                                         # primer segmento: /cuentas/... -> cuentas
        es_coleccion = not segmentos[-1].startswith("{")               # /cuentas sí; /cuentas/{idCuenta} no

        descripcion = (operacion.get("description") or "").strip()
        contexto = (recursos[-1]["descripcion"] or "").strip()
        nombre = operacion.get("displayName") or self._nombre_corto(descripcion, metodo_http, endpoint)

        propiedades_consulta = self.propiedades(operacion.get("queryString"))
        cuerpo = self._contenido_del_cuerpo(operacion.get("body"))
        propiedades_cuerpo = self.propiedades(cuerpo) if cuerpo is not None else []

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
                [parametro.strip("{}") for parametro in parametros_ruta], propiedades_consulta,
                propiedades_cuerpo, cuerpo),
            "valores_enum": self._valores_enum([*propiedades_consulta, *propiedades_cuerpo]),
            "entrada": self._describir_entrada(operacion, operacion_sin_traits, parametros_ruta, parametros_uri,
                                               propiedades_consulta, cuerpo),
            "salida": self._describir_salida(operacion),
            "controles": self._describir_controles(operacion, nodo, traits),
        }

    def _describir_entrada(self, operacion: dict, operacion_sin_traits: dict, parametros_ruta: list[str],
                           parametros_uri: dict, propiedades_consulta: list[dict], cuerpo: Any) -> str:
        """Parámetros de ruta, de consulta, encabezados y cuerpo: "Ruta: idCuenta: string; Consulta: ..."."""
        partes: list[str] = []
        if parametros_ruta:
            detalle = []
            for parametro in parametros_ruta:
                clave = parametro.strip("{}")
                info = parametros_uri.get(clave)
                tipo = info.get("type", "string") if isinstance(info, dict) else (info or "string")
                detalle.append(f"{clave}: {tipo}")
            partes.append("Ruta: " + ", ".join(detalle))

        if propiedades_consulta:
            partes.append("Consulta: " + self._describir_campos(propiedades_consulta))

        encabezados_propios = operacion_sin_traits.get("headers") or {}
        encabezados = []
        for clave, valor in (operacion.get("headers") or {}).items():
            nombre, opcional = _sin_opcional(str(clave))
            detalle = valor if isinstance(valor, dict) else {}
            requerido = bool(detalle.get("required", not opcional))
            # los encabezados opcionales que sólo aporta un trait común se omiten
            if requerido or clave in encabezados_propios:
                encabezados.append(f"{nombre} ({'obligatorio' if requerido else 'opcional'})")
        if encabezados:
            partes.append("Encabezados: " + ", ".join(encabezados))

        if cuerpo is not None:
            partes.append("Cuerpo: " + self.describir_cuerpo(cuerpo))
        return "; ".join(partes) if partes else "Sin parámetros"

    def _describir_salida(self, operacion: dict) -> str:
        """Respuestas exitosas (2xx) con su cuerpo: "200 Cuenta {idCuenta: string, ...} - descripción"."""
        partes = []
        respuestas = operacion.get("responses") or {}
        for codigo in sorted((c for c in respuestas if str(c).isdigit()), key=lambda c: int(c)):
            if not 200 <= int(codigo) < 300:
                continue
            respuesta = respuestas[codigo] or {}
            cuerpo_respuesta = self._contenido_del_cuerpo(respuesta.get("body"))
            texto = f"{codigo}"
            if cuerpo_respuesta is not None:
                texto += f" {self.describir_cuerpo(cuerpo_respuesta)}"
            if respuesta.get("description"):
                texto += f" - {respuesta['description']}"
            partes.append(texto)
        return "; ".join(partes) if partes else "Sin cuerpo de respuesta"

    def _describir_controles(self, operacion: dict, nodo: dict, traits: list[str]) -> list[str]:
        """Controles de seguridad y de negocio: token, idempotencia, segundo factor."""
        controles = [CONTROLES_POR_TRAIT[trait] for trait in traits if trait in CONTROLES_POR_TRAIT]
        seguridad = operacion.get("securedBy", nodo.get("securedBy", self.seguridad_global))
        if isinstance(seguridad, list) and None in seguridad:  # securedBy: [null] -> endpoint público
            controles.insert(0, "no requiere token de acceso (endpoint de acceso)")
        elif seguridad:
            controles.insert(0, "token Bearer en el encabezado Authorization")
        return controles

    def _contenido_del_cuerpo(self, body: Any) -> Any:
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
        """Nombre del servicio cuando el RAML no trae displayName: la primera frase de la descripción."""
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
        """Palabras con las que se busca el servicio: dominio, recursos, verbos del método y campos."""
        piezas: list[str] = [dominio, *literales, *PALABRAS_CLAVE_POR_METODO.get(metodo, [])]
        if metodo == "GET" and es_coleccion:
            piezas.append("listar")
        piezas += params_ruta
        piezas += [propiedad["nombre"] for propiedad in [*props_consulta, *props_cuerpo]
                   if propiedad["nombre"] not in PARAMETROS_PAGINACION]
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
        return _sin_repetidos(valor for propiedad in props for valor in propiedad["enum"]
                              if valor not in ("True", "False"))


def _cargar_yaml(texto: str, carpeta: Path) -> Any:
    """yaml.safe_load con soporte para ``!include``, resuelto relativo a ``carpeta``.

    Los fragmentos .raml/.yaml/.yml se interpretan; cualquier otro archivo
    (ejemplos .json, documentación .md) se incluye como texto.
    """
    class Cargador(yaml.SafeLoader):
        """Cargador de YAML seguro al que se agrega la etiqueta !include."""

    def incluir(cargador: yaml.SafeLoader, nodo: yaml.Node) -> Any:
        """Contenido del archivo que nombra la etiqueta !include, relativo a la carpeta del RAML."""
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

    documento = DocumentoRAML(datos)
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
    from configuracion.rutas import RUTA_INVENTARIO

    destino = sys.argv[1] if len(sys.argv) > 1 else RUTA_INVENTARIO
    inventario = cargar_inventario_raml(destino)
    print(f"{inventario['titulo']} {inventario['version']} - {len(inventario['servicios'])} servicios")
    for servicio in inventario["servicios"]:
        print(f"{servicio['id']}  {servicio['dominio']:<15} {servicio['metodo']:<6} {servicio['endpoint']}")
