# Asistente para la identificación de servicios REST — Avance 1

Chatbot basado en reglas (Python 3.10+ y NLTK) que relaciona requerimientos funcionales o descripciones de pantallas con un inventario de servicios REST (leído de un archivo RAML 1.0), recomienda los servicios existentes y señala brechas de cobertura.

Materia: Desarrollo de aplicaciones con inteligencia artificial — Maestría en Inteligencia Artificial.

## Arquitectura

Cada mensaje recorre tres componentes en secuencia; sólo el gestor de respuestas consulta la base de conocimiento.

```
mensaje ─> 1. Preprocesamiento ─> 2. Clasificador de intenciones ─> 3. Gestor de respuestas ─> respuesta
                                                                              ^
                                                     base de conocimiento (inventario RAML)
```

| Componente | Módulo | Responsabilidad |
|---|---|---|
| 1. Preprocesamiento | `preprocesamiento.py` | normalización, tokenización, palabras vacías, sinónimos y stemming; produce un `MensajePreprocesado` |
| 2. Clasificador de intenciones | `clasificador_intenciones.py` | `nltk.chat.util.Chat` para la charla y reglas regex ordenadas; produce una `Clasificacion` con la intención y sus datos (dominio, ID o posición del servicio, necesidades de un requerimiento) |
| 3. Gestor de respuestas | `gestor_respuestas.py` | búsqueda ponderada (TF-IDF, verbo → método HTTP, sustantivo → recurso), cobertura, brechas y contexto de sesión (últimos servicios recomendados) |
| Base de conocimiento | `inventario_raml.py` | lee el RAML y construye el inventario de servicios |
| Orquestador | `chatbot_servicios.py` | conecta los componentes (`AsistenteServiciosREST`) y ofrece el modo interactivo |

### Flujo de cada turno

`AsistenteServiciosREST.responder()` recorre estos nodos en orden:

| Nodo | Componente | Resultado |
|---|---|---|
| Normalizar | `preprocesar()` | minúsculas, sin acentos ni `¿ ¡`; además palabras y raíces con sinónimos |
| ¿Charla? | `ClasificadorIntenciones.charla()` (nltk `Chat`) | sí → respuesta de charla y fin del turno |
| Reglas de intención en orden | `ClasificadorIntenciones.aplicar_reglas()` | imagen → ID SRV → brecha → dominios → requerimiento → detalle → búsqueda |
| ¿Coincide con el inventario? | `GestorRespuestas.coincide_con_inventario()` | sólo si ninguna regla aplicó: sí → `buscar_servicio`; no → `fuera_de_alcance` (mensaje guía + 'ayuda') y fin del turno |
| Manejador de la intención | `GestorRespuestas.manejar()` ↔ base de conocimiento | buscar, analizar requerimiento, detalle, listar dominios, listar por dominio, brecha, imagen (aviso Avance 2) |
| Respuesta + contexto | `AsistenteServiciosREST` | guarda los últimos servicios recomendados para "detalle del primero" |

La coincidencia con el inventario usa la misma búsqueda ponderada que el manejador: TF-IDF con
`PUNTAJE_MINIMO = 2.0` y `COBERTURA_MINIMA = 0.4`, más los bonos por verbo (método HTTP) y por recurso.

## Estructura

```
chatbot-tm/
├── Avance1_Chatbot_Servicios_REST.ipynb   # Notebook (VS Code / Google Colab)
├── chatbot_servicios.py                   # Orquestador y modo interactivo
├── preprocesamiento.py                    # 1. Preprocesamiento
├── clasificador_intenciones.py            # 2. Clasificador de intenciones
├── gestor_respuestas.py                   # 3. Gestor de respuestas
├── inventario_raml.py                     # Base de conocimiento: lector RAML -> inventario de servicios
├── pruebas_conversaciones.py              # Conversaciones de prueba (las ejecuta test_chatbot.py)
├── test_chatbot.py                        # Pruebas (pytest): una por conversación
├── pytest.ini                             # Configuración de pytest
├── data/plataforma_servicios_bancarios.raml  # Base de conocimiento (API RAML 1.0)
├── evidencias/                            # Transcripción, resultados y generalización (los genera test_chatbot.py)
├── .vscode/launch.json                    # Configuraciones de ejecución
└── requirements.txt
```

## Ejecución en VS Code

1. Abre la carpeta del repositorio en VS Code (extensiones recomendadas: Python y Jupyter).
2. En la terminal integrada crea el entorno e instala dependencias:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Ejecuta desde la terminal o desde *Run and Debug* (configuraciones en `.vscode/launch.json`):

```bash
python -m pytest test_chatbot.py   # pruebas: una por conversación; al terminar regenera evidencias/
python chatbot_servicios.py        # modo interactivo
```

Al terminar, `test_chatbot.py` guarda en `evidencias/` la transcripción de las conversaciones
(`transcripcion_pruebas.txt`), los resultados por turno (`resultados_pruebas.json`) y la prueba de
generalización (`prueba_generalizacion.txt`).

Para ejecutar sólo algunas conversaciones: `python -m pytest test_chatbot.py -k c02` (o `-k "c13 or c14"`).
Desde VS Code, las pruebas también aparecen en el panel *Testing* (configurado con pytest).

Los recursos de NLTK (`punkt_tab`, `stopwords`) se descargan solos la primera vez.
El notebook `Avance1_Chatbot_Servicios_REST.ipynb` también corre en VS Code y en Google Colab.

## Inventario de servicios (RAML)

El inventario se construye desde `data/plataforma_servicios_bancarios.raml`. Cada método HTTP de cada recurso
(por ejemplo `GET /clientes` o `POST /transferencias/{idTransferencia}/cancelacion`) es un servicio; sus IDs
`SRV-001`, `SRV-002`… se asignan en el orden en que aparecen en el RAML. Del RAML se toman:

| Campo del servicio | Origen en el RAML |
|---|---|
| `dominio` | primer segmento de la ruta (`/clientes`, `/cuentas`…) |
| `metodo`, `endpoint`, `url` | método HTTP, ruta completa y `baseUri` |
| `nombre` | `displayName` del método, o la primera frase de su `description` |
| `descripcion`, `contexto` | `description` del método y del recurso |
| `entrada` | parámetros de ruta, `queryString`, encabezados y cuerpo (con `types` y `traits` resueltos) |
| `salida` | respuestas 2xx |
| `controles` | token Bearer, idempotencia y segundo factor (traits `idempotente` y `confirmacionFuerte`) |
| `palabras_clave` | ruta, verbos según el método y nombres de campos del cuerpo/consulta |
| `valores_enum` | valores permitidos de esos campos (pesan como la descripción, no como palabra clave) |

Para usar otro inventario, reemplaza el archivo `.raml` (o cambia `RUTA_INVENTARIO` en `chatbot_servicios.py`).
Para revisar cómo se interpreta, ejecuta `python inventario_raml.py data/plataforma_servicios_bancarios.raml`.
Las conversaciones de prueba (`pruebas_conversaciones.py`, ejecutadas por `test_chatbot.py`) hacen referencia a las operaciones como `"MÉTODO /endpoint"`, así que no
dependen del orden de los IDs; si cambias el RAML, revisa que sus mensajes y resultados esperados sigan aplicando.

## Intenciones soportadas

| Intención | Ejemplo |
|---|---|
| charla (saludo, ayuda, gracias, despedida) | "Hola", "ayuda" |
| buscar_servicio | "¿Existe un servicio para consultar el saldo?" |
| analizar_requerimiento | "Pantalla para buscar cliente, ver saldo y registrar pago" |
| detalle_servicio | "Detalle de SRV-007", "detalle del primero" |
| listar_dominios | "¿Qué servicios hay en el inventario?" |
| listar_por_dominio | "Servicios de pagos" |
| reportar_brecha | "¿Qué hago si no existe el servicio?" |
| analizar_imagen | "Te comparto el mockup" (se habilitará en el Avance 2) |
| fuera_de_alcance | Cualquier consulta ajena al dominio |

## Resultados del Avance 1

- `python -m pytest test_chatbot.py`: 19 conversaciones, 37 turnos, todas correctas. Cada turno verifica
  la intención y, en búsquedas y detalles, el servicio que nombra la respuesta (o que no nombre ninguno,
  cuando se espera una brecha). C13–C19 reproducen errores encontrados en las revisiones (ya corregidos).
  Estado actual: 37/37 intenciones, 21/21 servicios, 37/37 turnos correctos.
- Prueba de generalización (mejor servicio recomendado, o brecha cuando no existe servicio; ver
  `evidencias/prueba_generalizacion.txt`):
  - Conjunto de ajuste (12 consultas, usadas para elegir los parámetros y sinónimos): 7/12.
  - Conjunto de validación (24 consultas escritas antes de ajustar): 16/24
    (12/20 servicios correctos y 4/4 brechas detectadas). Algunos sinónimos ("revisar", "pedir",
    "encriptar", "buscar", "subir") se agregaron después de ver fallos de este conjunto, así que la
    cifra es optimista; para una medición limpia hace falta un conjunto nuevo no visto.
  - La búsqueda pondera las raíces con TF-IDF, exige un puntaje y una cobertura mínimos de la consulta
    y usa el verbo para preferir el método HTTP y el sustantivo para preferir el recurso del endpoint. Los fallos restantes son de vocabulario
    ("congelar", "anular", "dinero", "recibo") y no se resuelven con reglas léxicas sin sobreajustar;
    son la motivación para usar representaciones semánticas en el siguiente avance.
