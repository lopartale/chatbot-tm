"""
1. Preprocesamiento

Convierte el mensaje del usuario en una representación que el clasificador y el
gestor de respuestas pueden comparar con el inventario:

    normalización -> tokenización -> palabras vacías -> sinónimos -> stemming

El mismo proceso (sin sinónimos) se aplica a los textos del inventario al construir
el índice de búsqueda, por eso ambos lados terminan en las mismas raíces.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


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


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y sin signos de apertura (¿ ¡)."""
    sin_acentos = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                          if unicodedata.category(c) != "Mn")
    return sin_acentos.replace("¿", " ").replace("¡", " ").strip()


# Palabras vacías de NLTK + términos frecuentes en consultas que no distinguen servicios
PALABRAS_VACIAS = {normalizar(p) for p in stopwords.words("spanish")} | {
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
    original: str             # tal como lo escribió el usuario
    texto: str                # normalizado; sobre él trabajan las reglas del clasificador
    palabras: list[str]       # palabras relevantes: sin signos ni palabras vacías
    raices: list[set[str]]    # por palabra, sus raíces alternativas: la propia y la de su sinónimo


def _palabras(texto: str) -> list[str]:
    """Palabras normalizadas del texto, sin signos ni palabras vacías."""
    return [token for token in word_tokenize(normalizar(texto), language="spanish")
            if token.isalnum() and token not in PALABRAS_VACIAS]


def extraer_raices(texto: str) -> list[str]:
    """Raíz (stem) de cada palabra relevante del texto, sin sinónimos (para el índice)."""
    return [STEMMER.stem(palabra) for palabra in _palabras(texto)]


def preprocesar(mensaje: str) -> MensajePreprocesado:
    """Aplica el preprocesamiento completo a un mensaje o fragmento de mensaje."""
    texto = normalizar(mensaje)
    palabras = _palabras(texto)
    raices: list[set[str]] = []
    for palabra in palabras:
        grupo = {STEMMER.stem(palabra)}
        if palabra in SINONIMOS:
            grupo.add(STEMMER.stem(SINONIMOS[palabra]))
        if grupo not in raices:
            raices.append(grupo)
    return MensajePreprocesado(mensaje, texto, palabras, raices)
