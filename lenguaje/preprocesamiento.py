"""
Preprocesamiento

Convierte el mensaje del usuario en las representaciones que usan los demás componentes.
Hay dos procesos porque tienen objetivos distintos:

  Para la búsqueda en el inventario (Avance 1): preprocesar() y extraer_raices()

          normalización -> tokenización -> palabras vacías del dominio -> sinónimos -> stemming

      Lo usan las reglas de extracción del clasificador y el gestor de respuestas. El mismo
      proceso (sin sinónimos) se aplica a los textos del inventario al construir el índice de
      búsqueda, por eso ambos lados terminan en las mismas raíces.

  Para el clasificador de intenciones (Avance 2): preprocesar_intencion()

          normalización -> tokenización -> palabras vacías -> stemming

      Es la entrada de la representación TF-IDF. Usa otra lista de palabras vacías: la de la
      búsqueda quita palabras que aquí sirven para distinguir intenciones ("pantalla", "hay",
      "existe"...). No usa sinónimos: el clasificador aprende del conjunto de datos qué palabras
      se relacionan con cada intención.

Ambos procesos comparten la normalización base y el SnowballStemmer.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


# ==========================================================================
# Recursos de NLTK y normalización base (compartidos por la búsqueda y el clasificador)
# ==========================================================================
def asegurar_recursos_nltk() -> None:
    """Descarga los recursos de NLTK sólo si no están instalados.

    punkt_tab es el modelo del tokenizador (word_tokenize) y stopwords la lista de
    palabras vacías en español. En Colab se descargan la primera vez que se importa el módulo.
    """
    for ruta, paquete in [("tokenizers/punkt_tab", "punkt_tab"),
                          ("corpora/stopwords", "stopwords")]:
        try:
            nltk.data.find(ruta)
        except LookupError:
            nltk.download(paquete, quiet=True)


asegurar_recursos_nltk()
STEMMER = SnowballStemmer("spanish")  # reduce cada palabra a su raíz: "consultar", "consulta" -> "consult"


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y sin signos de apertura (¿ ¡)."""
    # NFD separa cada letra de su acento ("á" -> "a" + "´"); los acentos (categoría Mn) se descartan
    sin_acentos = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                          if unicodedata.category(c) != "Mn")
    return sin_acentos.replace("¿", " ").replace("¡", " ").strip()


# ==========================================================================
# Preprocesamiento para la búsqueda en el inventario (Avance 1)
# ==========================================================================
# Palabras vacías de NLTK + términos frecuentes en consultas que no distinguen servicios
PALABRAS_VACIAS_BUSQUEDA = {normalizar(p) for p in stopwords.words("spanish")} | {
    "servicio", "servicios", "api", "apis", "rest", "endpoint", "necesito",
    "quiero", "hay", "existe", "algun", "alguno", "alguna", "permita",
    "permite", "pueda", "puede", "usuario", "usuarios", "debe", "tambien",
    "ademas", "favor", "ayudas", "ayudar", "tienes", "cual", "cuales", "dar",
    "requerimiento", "pantalla", "sirva", "sirve", "busco", "mostrar",
    "muestre", "ver", "funcionalidad", "opcion", "inventario", "ahora",
}

# Palabra del usuario -> palabra que usa el inventario. Sólo se aplica al mensaje y como
# alternativa (la palabra original también cuenta), así el índice conserva su vocabulario.
# El destino debe existir en el RAML; los verbos de método HTTP van en gestor_respuestas.
SINONIMOS = {
    # autenticación
    "login": "sesion", "loguear": "sesion", "ingresar": "sesion", "entrar": "sesion",
    "logout": "revocar",
    # datos de cliente y notificaciones
    "email": "correo", "mail": "correo", "direccion": "domicilio",
    "notificar": "notificacion", "avisar": "notificacion",
    # cuentas, pagos y transferencias
    "balance": "saldo", "adeudo": "pendiente", "transacciones": "movimientos",
    "cobrar": "pago", "transferir": "transferencia",
    # préstamos y seguros
    "credito": "prestamo", "creditos": "prestamos", "mensualidades": "amortizacion",
    "poliza": "seguro", "polizas": "seguros",
    # documentos y criptografía
    "archivo": "documento", "subir": "aportado", "adjuntar": "aportado",
    "encriptar": "cifrar", "desencriptar": "descifrar",
    # acciones con otro nombre en el inventario
    "alta": "registrar", "agregar": "registrar", "pedir": "solicitar",
    "editar": "actualizar", "cambiar": "actualizar",
    "baja": "eliminar", "quitar": "eliminar",
    "buscar": "consultar", "revisar": "consultar",
}


@dataclass(frozen=True)
class MensajePreprocesado:
    """Un mensaje en las formas que usan la búsqueda y las reglas de extracción."""
    original: str             # tal como lo escribió el usuario
    texto: str                # normalizado; sobre él trabajan las reglas del clasificador
    palabras: list[str]       # palabras relevantes: sin signos ni palabras vacías
    raices: list[set[str]]    # por palabra, sus raíces alternativas: la propia y la de su sinónimo


def _palabras(texto: str) -> list[str]:
    """Palabras normalizadas del texto, sin signos ni palabras vacías."""
    # isalnum() descarta los signos de puntuación que word_tokenize separa como tokens
    return [token for token in word_tokenize(normalizar(texto), language="spanish")
            if token.isalnum() and token not in PALABRAS_VACIAS_BUSQUEDA]


def extraer_raices(texto: str) -> list[str]:
    """Raíz (stem) de cada palabra relevante del texto, sin sinónimos (para el índice)."""
    return [STEMMER.stem(palabra) for palabra in _palabras(texto)]


def preprocesar(mensaje: str) -> MensajePreprocesado:
    """Aplica el preprocesamiento de búsqueda completo a un mensaje o fragmento de mensaje.

    "Necesito hacer login" -> palabras [hacer, login] -> raíces [{hac}, {login, sesion}]
    """
    texto = normalizar(mensaje)
    palabras = _palabras(texto)
    raices: list[set[str]] = []
    for palabra in palabras:
        grupo = {STEMMER.stem(palabra)}                   # raíz de la palabra escrita
        if palabra in SINONIMOS:
            grupo.add(STEMMER.stem(SINONIMOS[palabra]))   # y la del término que usa el inventario
        if grupo not in raices:                           # una palabra repetida cuenta una sola vez
            raices.append(grupo)
    return MensajePreprocesado(mensaje, texto, palabras, raices)


# ==========================================================================
# Preprocesamiento para el clasificador de intenciones (Avance 2)
# ==========================================================================
# Cualquier ID de servicio ("SRV-040", "srv 30", "srv7") se convierte en un solo término:
# para decidir la intención importa que el mensaje mencione un ID, no cuál. El mismo patrón
# usa la extracción de datos (extraccion_datos.py) para leer el número del ID.
PATRON_ID_SERVICIO = r"\bsrv[\s-]?(\d{1,3})\b"
TERMINO_ID_SERVICIO = "srvid"

# Abreviaturas frecuentes en mensajes escritos con prisa
ABREVIATURAS = {
    "q": "que", "xq": "porque", "pq": "porque", "porfa": "por favor", "pls": "por favor",
    "grax": "gracias", "thx": "gracias", "info": "informacion", "dm": "mensaje",
}

# Palabras de la lista de NLTK que NO se eliminan porque ayudan a reconocer la intención.
# Se eligieron revisando el conjunto de entrenamiento (el de prueba no se usó).
PALABRAS_CON_INTENCION = {
    "no", "si",                               # reportar_brecha: "si no existe", "no hay servicio"
    "hay", "tiene", "tienes",                 # listar: "¿qué hay?", "¿qué tiene seguros?"
    "que", "como", "cual", "quien", "donde",  # preguntas: "qué hago", "cómo funcionas", "quién eres"
    "ese", "esa", "este", "esta",             # detalle ("ese servicio") e imagen ("esta imagen")
    "mas", "eres", "y",                       # "más información", "¿eres un bot?", "y" enumera funciones
    "todo", "todos",                          # listar_dominios: "todos los servicios"
    "estado",                                 # NLTK la toma como verbo, aquí es "estado de cuenta"
}
# Lista de NLTK (sin acentos, como quedan los mensajes normalizados) menos las palabras útiles
PALABRAS_VACIAS_CLASIFICADOR = {normalizar(palabra) for palabra in stopwords.words("spanish")} - PALABRAS_CON_INTENCION


def expandir_abreviatura(coincidencia: re.Match) -> str:
    """Reemplazo para re.sub: la palabra encontrada, o su forma completa si es una abreviatura."""
    palabra = coincidencia.group()
    return ABREVIATURAS.get(palabra, palabra)


def normalizar_mensaje(texto: str) -> str:
    """Paso 1. Normalización del Avance 1 más IDs de servicio, letras repetidas y abreviaturas."""
    texto = normalizar(texto)                                       # minúsculas, sin acentos ni ¿ ¡
    texto = re.sub(PATRON_ID_SERVICIO, f" {TERMINO_ID_SERVICIO} ", texto)  # "srv-040" -> "srvid"
    texto = re.sub(r"([a-z])\1{2,}", r"\1", texto)                  # "ayudaaa" -> "ayuda"
    texto = re.sub(r"\b\w+\b", expandir_abreviatura, texto)         # "q" -> "que"
    return re.sub(r"\s+", " ", texto).strip()                       # un solo espacio entre palabras


def tokenizar(texto_normalizado: str) -> list[str]:
    """Paso 2. Tokens de NLTK (word_tokenize en español), sin signos de puntuación."""
    tokens = word_tokenize(texto_normalizado, language="spanish")
    return [token for token in tokens if token.isalnum()]


def quitar_palabras_vacias(tokens: list[str]) -> list[str]:
    """Paso 3. Elimina las palabras vacías de NLTK, salvo las que ayudan a reconocer la intención."""
    return [token for token in tokens if token not in PALABRAS_VACIAS_CLASIFICADOR]


def aplicar_stemming(tokens: list[str]) -> list[str]:
    """Paso 4. SnowballStemmer para español (el mismo del Avance 1); el término de ID no cambia."""
    # "srvid" no es una palabra del español: el stemmer lo recortaría y se perdería el término
    return [token if token == TERMINO_ID_SERVICIO else STEMMER.stem(token) for token in tokens]


def preprocesar_intencion(texto: str) -> list[str]:
    """Los cuatro pasos: del mensaje original a la lista de raíces para el clasificador.

    "Te comparto el mockup de la pantalla"
        1. normalizado:          "te comparto el mockup de la pantalla"
        2. tokens:               [te, comparto, el, mockup, de, la, pantalla]
        3. sin palabras vacías:  [comparto, mockup, pantalla]
        4. raíces:               [compart, mockup, pantall]
    """
    texto_normalizado = normalizar_mensaje(texto)
    tokens = tokenizar(texto_normalizado)
    tokens_utiles = quitar_palabras_vacias(tokens)
    return aplicar_stemming(tokens_utiles)


def mostrar_pasos(texto: str) -> dict[str, object]:
    """Resultado de cada paso, para explicar el proceso en el notebook y en el reporte."""
    texto_normalizado = normalizar_mensaje(texto)
    tokens = tokenizar(texto_normalizado)
    tokens_utiles = quitar_palabras_vacias(tokens)
    return {
        "original": texto,
        "normalizado": texto_normalizado,
        "tokens": tokens,
        "sin_palabras_vacias": tokens_utiles,
        "raices": aplicar_stemming(tokens_utiles),
    }
