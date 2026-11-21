"""
Rutas del proyecto

Todas las carpetas y archivos de datos se ubican desde la raíz del repositorio, así los
módulos funcionan igual sin importar en qué paquete estén ni desde dónde se ejecuten.
"""
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

CARPETA_DATOS = RAIZ_PROYECTO / "data"
CARPETA_IMAGENES = CARPETA_DATOS / "imagenes"                      # pantallas generadas (no se versiona)
CARPETA_MODELOS = RAIZ_PROYECTO / "modelos"                        # modelos entrenados y ResNet50
CARPETA_EVIDENCIAS = RAIZ_PROYECTO / "evidencias"                  # métricas, matrices y transcripciones

RUTA_INVENTARIO = CARPETA_DATOS / "plataforma_servicios_bancarios.raml"   # base de conocimiento
RUTA_CONJUNTO_INTENCIONES = CARPETA_DATOS / "intenciones.csv"            # frases etiquetadas
