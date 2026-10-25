"""
Asistente para la identificación de servicios REST - Avance 1
Chatbot basado en reglas implementado con Python 3.10+ y NLTK.

El asistente recibe consultas o requerimientos funcionales en español,
los relaciona con el inventario de servicios REST (base de conocimiento)
y recomienda los servicios existentes, señalando brechas de cobertura.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

import nltk
from nltk.chat.util import Chat
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


# --------------------------------------------------------------------------
# 1. Recursos de NLTK y preprocesamiento
# --------------------------------------------------------------------------
def asegurar_recursos_nltk() -> None:
    """Descarga los recursos de NLTK sólo si no están instalados."""
    for ruta, paquete in [("tokenizers/punkt_tab", "punkt_tab"),
                          ("corpora/stopwords", "stopwords")]:
        try:
            nltk.data.find(ruta)
        except LookupError:
            nltk.download(paquete, quiet=True)


asegurar_recursos_nltk()
STEMMER = SnowballStemmer("spanish")


def quitar_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y sin signos de apertura (¿ ¡)."""
    return quitar_acentos(texto.lower()).replace("¿", " ").replace("¡", " ").strip()


# Palabras vacías de NLTK + términos que no aportan a la búsqueda
STOPWORDS = {normalizar(p) for p in stopwords.words("spanish")} | {
    "servicio", "servicios", "api", "apis", "rest", "endpoint", "necesito",
    "quiero", "hay", "existe", "algun", "alguno", "alguna", "permita",
    "permite", "pueda", "puede", "usuario", "usuarios", "debe", "tambien",
    "ademas", "favor", "ayudas", "ayudar", "tienes", "cual", "cuales",
    "requerimiento", "pantalla", "sirva", "sirve", "busco", "mostrar",
    "muestre", "ver", "funcionalidad", "opcion",
}

# Sinónimos frecuentes en requerimientos -> término del inventario
SINONIMOS = {
    "login": "autenticar", "loguear": "autenticar", "ingresar": "autenticar",
    "entrar": "autenticar", "balance": "saldo", "orden": "pedido",
    "ordenes": "pedidos", "email": "correo", "mail": "correo",
    "notificar": "notificacion", "avisar": "notificacion",
    "transacciones": "movimientos", "domicilio": "direccion",
    "cp": "postal", "cobrar": "pago", "pagar": "pago", "logout": "cerrar",
    "subir": "cargar", "adjuntar": "cargar", "archivo": "documento",
}


def preprocesar(texto: str) -> list[str]:
    """Tokeniza, elimina palabras vacías, aplica sinónimos y stemming."""
    tokens = word_tokenize(normalizar(texto), language="spanish")
    raices = []
    for tok in tokens:
        if not tok.isalnum() or tok in STOPWORDS:
            continue
        tok = SINONIMOS.get(tok, tok)
        raices.append(STEMMER.stem(tok))
    return raices


# --------------------------------------------------------------------------
# 2. Base de conocimiento: inventario de servicios REST
# --------------------------------------------------------------------------
class BaseConocimiento:
    def __init__(self, ruta_json: str | Path):
        with open(ruta_json, encoding="utf-8") as f:
            self.servicios = json.load(f)["servicios"]
        self.por_id = {s["id"]: s for s in self.servicios}
        self.indice = self._construir_indice()

    def _construir_indice(self) -> dict[str, dict[str, float]]:
        """Para cada servicio guarda raíz -> peso (2 nombre/palabras clave, 1 descripción)."""
        indice = {}
        for s in self.servicios:
            pesos: dict[str, float] = {}
            for raiz in preprocesar(s["descripcion"]):
                pesos[raiz] = 1.0
            for frase in [s["nombre"], s["dominio"], *s["palabras_clave"]]:
                for raiz in preprocesar(frase):
                    pesos[raiz] = 2.0
            indice[s["id"]] = pesos
        # Las raíces que aparecen en muchos servicios pesan menos (rareza)
        frecuencia: dict[str, int] = {}
        for pesos in indice.values():
            for raiz in pesos:
                frecuencia[raiz] = frecuencia.get(raiz, 0) + 1
        for pesos in indice.values():
            for raiz in pesos:
                pesos[raiz] /= frecuencia[raiz]
        return indice

    def buscar(self, texto: str, umbral: float = 1.0, maximo: int = 3):
        """Regla de coincidencia ponderada: suma de pesos de raíces compartidas."""
        raices = set(preprocesar(texto))
        resultados = []
        for sid, pesos in self.indice.items():
            puntaje = sum(pesos.get(r, 0.0) for r in raices)
            if puntaje >= umbral:
                resultados.append((round(puntaje, 2), self.por_id[sid]))
        resultados.sort(key=lambda x: -x[0])
        return resultados[:maximo]

    def dominios(self) -> dict[str, list[dict]]:
        agrupados: dict[str, list[dict]] = {}
        for s in self.servicios:
            agrupados.setdefault(s["dominio"], []).append(s)
        return agrupados


# --------------------------------------------------------------------------
# 3. Reglas conversacionales (charla básica con nltk.chat.util.Chat)
# --------------------------------------------------------------------------
PARES_CHARLA = [
    [r"^(hola|buen[oa]s? (dias|tardes|noches)|que tal|hey|saludos)[\s!.,]*$",
     ["¡Hola! Soy el asistente de servicios REST. Descríbeme la funcionalidad o "
      "la pantalla que necesitas y te digo qué servicios del inventario la cubren."]],
    [r"^(adios|hasta luego|bye|nos vemos|salir|terminar)[\s!.,]*$",
     ["¡Hasta luego! Recuerda reutilizar antes de construir."]],
    [r"^(muchas )?gracias.*$|^excelente.*$|^perfecto.*$",
     ["Con gusto. ¿Quieres analizar otro requerimiento?"]],
    [r"^(ayuda|help|que puedes hacer|como funcionas|que haces|menu)[\s?!.]*$",
     ["Puedo ayudarte a:\n"
      "  1) Buscar un servicio por funcionalidad (ej. 'servicio para consultar saldo').\n"
      "  2) Analizar un requerimiento o pantalla con varias funciones y mostrar la cobertura.\n"
      "  3) Ver el detalle o contrato de un servicio (ej. 'detalle de SRV-007').\n"
      "  4) Listar los dominios y servicios del inventario.\n"
      "  5) Orientarte cuando exista una brecha (funcionalidad sin servicio)."]],
    [r"^(quien eres|como te llamas|que eres)[\s?!.]*$",
     ["Soy un asistente basado en reglas (versión 1) que relaciona requerimientos "
      "funcionales con el inventario de servicios REST de la organización."]],
]

REGLAS_INTENCION = [
    ("analizar_imagen", r"\b(imagen|imagenes|captura|mockup|screenshot|figma|foto|png|jpg|boceto|wireframe)\b"),
    ("detalle_servicio", r"\bsrv-?\d{1,3}\b"),
    ("reportar_brecha", r"\b(brecha|no existe|no hay servicio|crear un servicio|solicitar un servicio|nuevo servicio|que hago si)\b"),
    ("listar_dominios", r"\b(dominios|categorias|inventario|catalogo de servicios|que servicios (hay|tienes|existen)|lista(r|do)? (de )?(todos )?los servicios)\b"),
    ("listar_por_dominio", r"\bservicios (de|del|para el dominio de) (\w+)"),
    ("analizar_requerimiento", r"\b(requerimiento|pantalla|historia de usuario|caso de uso|formulario|vista|modulo)\b"),
    ("detalle_servicio", r"\b(detalle|contrato|parametros|entrada|salida|como se (usa|consume))\b"),
    ("buscar_servicio", r"\b(servicio|api|endpoint|consultar|buscar|necesito|existe|hay|para)\b"),
]

ORDINALES = {"primero": 0, "primer": 0, "1": 0, "segundo": 1, "2": 1, "tercero": 2, "3": 2}


# --------------------------------------------------------------------------
# 4. Motor del chatbot
# --------------------------------------------------------------------------
class AsistenteServiciosREST:
    def __init__(self, ruta_inventario: str | Path, semilla: int = 42):
        random.seed(semilla)
        self.kb = BaseConocimiento(ruta_inventario)
        self.charla = Chat(PARES_CHARLA)
        self.ultimos: list[str] = []          # contexto: últimos servicios recomendados
        self.historial: list[dict] = []

    # ---- clasificación de intención por reglas ----
    def detectar_intencion(self, texto_norm: str) -> str:
        if self.charla.respond(texto_norm):
            return "charla"
        for intencion, patron in REGLAS_INTENCION:
            if re.search(patron, texto_norm):
                if intencion == "listar_por_dominio" and not self._dominio_en(texto_norm):
                    continue
                return intencion
        # Sin palabra disparadora, pero con coincidencias en el inventario
        if self.kb.buscar(texto_norm):
            return "buscar_servicio"
        return "fuera_de_alcance"

    def responder(self, mensaje: str) -> str:
        texto = normalizar(mensaje.strip())
        intencion = self.detectar_intencion(texto)
        manejador = getattr(self, f"_r_{intencion}")
        respuesta = manejador(texto)
        self.historial.append({"usuario": mensaje, "intencion": intencion, "bot": respuesta})
        return respuesta

    # ---- respuestas por intención ----
    def _r_charla(self, texto):
        return self.charla.respond(texto)

    def _r_buscar_servicio(self, texto):
        resultados = self.kb.buscar(texto)
        if not resultados:
            return self._r_reportar_brecha(texto, sin_coincidencia=True)
        self.ultimos = [s["id"] for _, s in resultados]
        lineas = ["Encontré estos servicios que pueden cubrir la funcionalidad:"]
        for i, (puntaje, s) in enumerate(resultados, 1):
            lineas.append(f"  {i}) {s['id']} {s['nombre']} [{s['metodo']} {s['endpoint']}] "
                          f"- puntaje {puntaje}")
        lineas.append("Escribe 'detalle del primero' o el ID para ver su contrato.")
        return "\n".join(lineas)

    def _r_analizar_requerimiento(self, texto):
        necesidades = self._segmentar(texto)
        if not necesidades:
            return ("Describe las funciones de la pantalla separadas por comas. "
                    "Ej.: 'pantalla para buscar cliente, ver su saldo y registrar un pago'.")
        cubiertas, brechas, recomendados = [], [], []
        for nec in necesidades:
            res = self.kb.buscar(nec, maximo=1)
            if res:
                s = res[0][1]
                cubiertas.append(f"  [OK] '{nec}' -> {s['id']} {s['nombre']} ({s['metodo']} {s['endpoint']})")
                if s["id"] not in recomendados:
                    recomendados.append(s["id"])
            else:
                brechas.append(f"  [BRECHA] '{nec}' -> sin servicio en el inventario")
        cobertura = round(100 * len(cubiertas) / len(necesidades))
        self.ultimos = recomendados
        lineas = [f"Analicé {len(necesidades)} necesidades funcionales. Cobertura: {cobertura}%."]
        lineas += cubiertas + brechas
        if brechas:
            lineas.append("Sugerencia: valida las brechas con el equipo de arquitectura "
                          "antes de crear un servicio nuevo.")
        return "\n".join(lineas)

    def _r_detalle_servicio(self, texto):
        sid = None
        m = re.search(r"srv-?(\d{1,3})", texto)
        if m:
            sid = f"SRV-{int(m.group(1)):03d}"
        else:
            for palabra, idx in ORDINALES.items():
                if re.search(rf"\b{palabra}\b", texto) and idx < len(self.ultimos):
                    sid = self.ultimos[idx]
                    break
            if sid is None and self.ultimos:
                sid = self.ultimos[0]
        if sid is None:
            res = self.kb.buscar(texto, maximo=1)
            sid = res[0][1]["id"] if res else None
        s = self.kb.por_id.get(sid) if sid else None
        if not s:
            return "No identifiqué el servicio. Indica su ID (ej. SRV-007) o busca primero una funcionalidad."
        return (f"{s['id']} - {s['nombre']} (dominio {s['dominio']})\n"
                f"  Método/endpoint: {s['metodo']} {s['endpoint']}\n"
                f"  Descripción: {s['descripcion']}\n"
                f"  Entrada: {s['entrada']}\n  Salida: {s['salida']}\n"
                f"  Equipo responsable: {s['equipo']}")

    def _r_listar_dominios(self, texto):
        lineas = [f"El inventario tiene {len(self.kb.servicios)} servicios en estos dominios:"]
        for dom, servs in sorted(self.kb.dominios().items()):
            lineas.append(f"  - {dom}: {len(servs)} servicio(s)")
        lineas.append("Pide, por ejemplo, 'servicios de pagos' para ver el detalle de un dominio.")
        return "\n".join(lineas)

    def _r_listar_por_dominio(self, texto):
        dom = self._dominio_en(texto)
        servs = self.kb.dominios()[dom]
        self.ultimos = [s["id"] for s in servs]
        lineas = [f"Servicios del dominio {dom}:"]
        lineas += [f"  - {s['id']} {s['nombre']} [{s['metodo']} {s['endpoint']}]" for s in servs]
        return "\n".join(lineas)

    def _r_reportar_brecha(self, texto, sin_coincidencia=False):
        inicio = ("No encontré un servicio que cubra esa funcionalidad. " if sin_coincidencia
                  else "Cuando una funcionalidad no tiene servicio (brecha), te sugiero: ")
        return (inicio + "\n  1) Reformular con otros términos o revisar servicios similares."
                "\n  2) Verificar si un servicio existente puede extenderse."
                "\n  3) Registrar la brecha con el equipo de arquitectura para evaluar un servicio nuevo.")

    def _r_analizar_imagen(self, texto):
        return ("El análisis de imágenes de pantallas aún no está disponible en esta versión. "
                "Se incorporará en el Avance 2 con clasificación de imágenes. Por ahora, "
                "describe en texto los elementos de la pantalla (campos, botones, acciones).")

    def _r_fuera_de_alcance(self, texto):
        return ("No entendí tu consulta o está fuera de mi alcance. Puedo buscar servicios REST "
                "por funcionalidad, analizar requerimientos de pantallas o mostrar contratos. "
                "Escribe 'ayuda' para ver ejemplos.")

    # ---- utilidades ----
    def _dominio_en(self, texto):
        for dom in self.kb.dominios():
            if normalizar(dom) in texto or normalizar(dom).rstrip("s") in texto:
                return dom
        return None

    @staticmethod
    def _segmentar(texto):
        """Divide un requerimiento en necesidades funcionales (una por frase)."""
        texto = re.sub(r"^.*?\b(pantalla|requerimiento|formulario|vista|modulo|historia de usuario|caso de uso)\b"
                       r"\s*(de|para|que|donde|en (la|el) que|con)?\s*", "", texto)
        partes = re.split(r",|;|\n|\.\s| y | ademas | tambien | luego ", texto)
        return [p.strip(" .:") for p in partes if preprocesar(p)]


# --------------------------------------------------------------------------
# 5. Ejecución interactiva (Colab / terminal)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ruta = Path(__file__).parent / "data" / "inventario_servicios.json"
    bot = AsistenteServiciosREST(ruta)
    print("Asistente de servicios REST (escribe 'salir' para terminar)")
    while True:
        entrada = input("Tú: ")
        print("Bot:", bot.responder(entrada))
        if normalizar(entrada).strip() in {"salir", "adios", "terminar"}:
            break
