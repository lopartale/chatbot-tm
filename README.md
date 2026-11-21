# Asistente para la identificación de servicios REST — Avance 2

Asistente multimodal que relaciona requerimientos funcionales, descripciones de pantallas **o capturas de pantallas de frontend** con un inventario de servicios REST (leído de un archivo RAML 1.0), recomienda los servicios existentes y señala brechas de cobertura.

- **Texto:** preprocesamiento con NLTK, representación TF-IDF y clasificador de intenciones SVM (scikit-learn).
- **Imagen:** clasificador de tipo de pantalla con ResNet50 preentrenada en ImageNet (transferencia de aprendizaje con Keras).
- **Integración:** el tipo de pantalla se traduce en las operaciones típicas de esa pantalla y se cruzan con el inventario.

Materia: Desarrollo de aplicaciones con inteligencia artificial — Maestría en Inteligencia Artificial.

## Arquitectura

Cada turno entra por `AsistenteServiciosREST.responder(mensaje, imagen=None)` y sigue uno de dos caminos según si trae una imagen. Los dos terminan en el mismo lugar: el gestor de respuestas, que es el único componente que consulta la base de conocimiento.

```
              Usuario: mensaje de texto y, opcionalmente, una captura
                                         │
                                         ▼
              AsistenteServiciosREST.responder(mensaje, imagen=None)
                                         │
                         ┌──────── ¿trae imagen? ─────────┐
                      no │                                │ sí
                         ▼                                ▼
          ┌──────────────────────────────┐ ┌──────────────────────────────┐
          │ Preprocesamiento             │ │ Clasificador visual          │
          │    NLTK: normalizar,         │ │    ResNet50 + capa softmax   │
          │    tokenizar, stemming       │ │    → tipo de pantalla        │
          └──────────────┬───────────────┘ └──────────────┬───────────────┘
                         ▼                                ▼
          ┌──────────────────────────────┐ ┌──────────────────────────────┐
          │ Clasificador de              │ │ Integración multimodal       │
          │    intenciones               │ │    tipo de pantalla →        │
          │    TF-IDF + SVM              │ │    operaciones típicas       │
          └──────────────┬───────────────┘ └──────────────┬───────────────┘
                         │                                │
                         └───────────────┬────────────────┘
                                         ▼
                          ┌──────────────────────────────┐   ┌────────────────────────┐
                          │ Gestor de respuestas         │   │ Base de conocimiento   │
                          │    servicios, cobertura,     │◄─►│    inventario RAML     │
                          │    brechas y contexto        │   │    (57 servicios)      │
                          └──────────────┬───────────────┘   └────────────────────────┘
                                         ▼
                                     Respuesta
```

- **Camino de texto (Avance 1 mejorado):** el mensaje se preprocesa con NLTK y el SVM sobre TF-IDF decide qué pide el usuario entre las 9 intenciones, incluidas la charla y el detalle de un servicio. Unas reglas extraen los datos de esa intención (dominio, ID o posición del servicio). Si es charla, NLTK Chat redacta la respuesta; si no, el gestor busca los servicios en el inventario.
- **Camino de imagen (nuevo en el Avance 2):** ResNet50 con una capa softmax reconoce el tipo de pantalla (login, transferencia, pago de servicios...). La integración multimodal traduce ese tipo en las operaciones que la pantalla suele necesitar, y el gestor las busca en el inventario por método y endpoint.
- **Salida común:** servicios recomendados, cobertura y brechas. Los servicios recomendados quedan como contexto de la sesión, así que después de una imagen se puede seguir en texto ("dame el contrato del tercero").

| Componente | Módulo | Responsabilidad |
|---|---|---|
| Preprocesamiento | `lenguaje/preprocesamiento.py` | búsqueda (Avance 1): normalización, palabras vacías del dominio, sinónimos y stemming. Clasificador (Avance 2): normalización (IDs, alargamientos, abreviaturas), tokenización NLTK, palabras vacías de NLTK con excepciones y stemming Snowball |
| Clasificador de intenciones | `lenguaje/clasificador_intenciones.py` · `lenguaje/representacion_texto.py` · `lenguaje/clasificador_svm.py` · `lenguaje/extraccion_datos.py` | SVM lineal sobre TF-IDF (unigramas y bigramas) que decide las 9 intenciones; reglas que extraen los datos de cada intención |
| Respuestas de charla | `lenguaje/respuestas_charla.py` | `nltk.chat.util.Chat` redacta saludos, ayuda y agradecimientos: patrón exacto o, si no coincide, palabra clave |
| Gestor de respuestas | `conocimiento/gestor_respuestas.py` · `conocimiento/formato_respuestas.py` | un manejador por intención: servicios recomendados, cobertura y brechas; formato de resúmenes y contratos |
| Clasificador visual | `vision/clasificador_visual.py` · `vision/capa_softmax.py` · `vision/preparacion_imagenes.py` · `vision/preparar_resnet50.py` · `vision/generar_pantallas.py` · `vision/plantillas_pantallas.py` | ResNet50 de `keras.applications` congelada (2048 características) + capa `Dense` softmax entrenada con Keras; 8 tipos de pantalla |
| Integración multimodal | `multimodal/integracion_multimodal.py` · `multimodal/patrones_pantalla.py` | tipo de pantalla → operaciones típicas (método y endpoint) → servicios del RAML; umbral de confianza 0.8 |
| Base de conocimiento | `conocimiento/base_conocimiento.py` · `conocimiento/inventario_raml.py` | lee el RAML, construye el inventario y busca servicios (TF-IDF ponderado) |
| Orquestador | `asistente/chatbot_servicios.py` | flujo conversacional de cada turno: `AsistenteServiciosREST.responder(mensaje, imagen=None)` |
| Sesión | `asistente/sesion_conversacion.py` | contexto (últimos servicios recomendados) e historial de turnos |
| Modo interactivo | `asistente/consola.py` | entrada y salida en terminal o Colab (`/imagen <ruta>`) |

### Flujo de cada turno

| Nodo | Componente | Resultado |
|---|---|---|
| ¿Imagen adjunta? | `AnalizadorPantallas` → `redactar_respuesta_de_pantalla()` | sí → tipo de pantalla, confianza (alternativas si < 0.8) y servicios de sus operaciones típicas; fin del turno |
| Normalizar | `preprocesar()` | minúsculas, sin acentos ni `¿ ¡`; palabras y raíces para la búsqueda |
| Clasificador SVM | `ClasificadorIntenciones.clasificar()` | intención entre las 9 (incluidas charla y detalle de servicio) y sus datos (dominio, ID o posición, necesidades) |
| ¿Coincide con el inventario? | `GestorRespuestas.coincide_con_inventario()` | sólo si el SVM dijo `fuera_de_alcance`: sí → `buscar_servicio`; no → mensaje guía |
| Charla | `RespuestasCharla.responder()` (nltk `Chat`) | patrón exacto ("gracias", "¿quién eres?") o, si no coincide, por palabra clave (ayuda, gracias, despedida, identidad, saludo) |
| Manejador de la intención | `GestorRespuestas.manejar()` ↔ base de conocimiento | buscar, analizar requerimiento, detalle, listar, brecha, imagen (pide adjuntarla) |
| Respuesta + contexto | `AsistenteServiciosREST` | guarda los últimos servicios recomendados para "detalle del primero" |

## Estructura

Los módulos se agrupan en paquetes según su función:

```
chatbot-tm/
├── Avance2_Asistente_Multimodal.ipynb     # Notebook del Avance 2 (VS Code / Google Colab)
├── Avance1_Chatbot_Servicios_REST.ipynb   # Notebook del Avance 1
├── asistente/                             # Asistente conversacional
│   ├── chatbot_servicios.py               #   Orquestador: flujo conversacional de cada turno
│   ├── sesion_conversacion.py             #   Contexto e historial de la conversación
│   ├── consola.py                         #   Modo interactivo (entrada, salida y comando /imagen)
│   └── __main__.py                        #   python -m asistente
├── lenguaje/                              # Procesamiento de lenguaje natural
│   ├── preprocesamiento.py                #   Para la búsqueda (Avance 1) y para el clasificador (Avance 2)
│   ├── representacion_texto.py            #   Representación TF-IDF y carga del conjunto de intenciones
│   ├── clasificador_svm.py                #   Modelo TF-IDF + SVM (crear, entrenar, cargar, explicar)
│   ├── clasificador_intenciones.py        #   Clasificador de intenciones (SVM + extracción de datos)
│   ├── extraccion_datos.py                #   Datos de la intención: dominio, ID o posición, necesidades
│   └── respuestas_charla.py               #   Respuestas de charla con nltk.chat.util.Chat
├── conocimiento/                          # Base de conocimiento y respuestas
│   ├── inventario_raml.py                 #   Lector RAML -> inventario de servicios
│   ├── base_conocimiento.py               #   Inventario de servicios y búsqueda ponderada
│   ├── gestor_respuestas.py               #   Gestor de respuestas: un manejador por intención
│   └── formato_respuestas.py              #   Formato de resúmenes, contratos y pasos ante una brecha
├── vision/                                # Componente visual
│   ├── generar_pantallas.py               #   Conjunto de imágenes: planea, captura y guarda las pantallas
│   ├── plantillas_pantallas.py            #   HTML de cada tipo de pantalla y estilos visuales
│   ├── preparar_resnet50.py               #   ResNet50 de Keras (ImageNet) congelada, sin su capa final
│   ├── preparacion_imagenes.py            #   Ajuste a 224 x 224 y aumento de datos
│   ├── capa_softmax.py                    #   Capa Dense softmax de Keras: entrenar, predecir, guardar y cargar
│   └── clasificador_visual.py             #   Extractor ResNet50 y clasificador de pantallas que usa el asistente
├── multimodal/                            # Integración multimodal
│   ├── patrones_pantalla.py               #   Operaciones típicas de cada tipo de pantalla
│   └── integracion_multimodal.py          #   De la imagen a los servicios: avisos, umbral y respuesta
├── evaluacion/                            # Entrenamiento, evaluación y evidencias
│   ├── evaluar_intenciones.py             #   Entrena y evalúa el componente textual -> evidencias/
│   ├── evaluar_pantallas.py               #   Entrena y evalúa el componente visual -> evidencias/
│   ├── pruebas_conversaciones.py          #   Conversaciones de texto (Avance 1)
│   ├── pruebas_multimodales.py            #   Conversaciones con imágenes (Avance 2)
│   ├── metricas.py                        #   Métricas compartidas (accuracy, precision, recall, F1)
│   └── registro_evidencias.py             #   Único módulo que escribe en evidencias/
├── configuracion/                         # Configuración compartida
│   ├── rutas.py                           #   Rutas del proyecto (data/, modelos/, evidencias/)
│   └── modelos_guardados.py               #   Carga de .joblib sólo si es compatible (estructura y versión)
├── tests/                                 # Pruebas (pytest)
├── data/plataforma_servicios_bancarios.raml  # Base de conocimiento (API RAML 1.0)
├── data/intenciones.csv                   # Conjunto de intenciones (508 entrenamiento / 126 prueba)
├── data/imagenes/                         # Pantallas generadas (no se versiona; vision/generar_pantallas.py)
├── modelos/                               # SVM (.joblib) y capa softmax de Keras (.keras + .json)
├── evidencias/                            # Métricas, matrices de confusión y transcripciones
├── pytest.ini                             # Las pruebas se buscan en tests/
└── requirements.txt
```

Todos los comandos se ejecutan desde la raíz del proyecto.

## Ejecución

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium            # sólo para generar las imágenes

# Texto
python -m evaluacion.evaluar_intenciones  # preprocesamiento, TF-IDF, entrena y evalúa el SVM -> modelos/, evidencias/

# Imágenes
python -m vision.preparar_resnet50        # descarga los pesos de ResNet50 para Keras (~95 MB, en ~/.keras/models)
python -m vision.generar_pantallas        # genera las 1,360 pantallas en data/imagenes/
python -m evaluacion.evaluar_pantallas    # extrae características, entrena la capa Dense softmax y evalúa

# Asistente
python -m pytest                          # todas las pruebas (no modifican evidencias/)
python -m evaluacion.pruebas_conversaciones # conversaciones de texto -> evidencias/transcripcion_pruebas.txt
python -m evaluacion.pruebas_multimodales # conversaciones con imágenes -> evidencias/transcripcion_multimodal.txt
python -m asistente                       # modo interactivo; "/imagen <ruta>" analiza una captura (ruta con espacios: entre comillas o terminada en .png/.jpg)
```

Desde Python o el notebook: `AsistenteServiciosREST().responder("¿Qué servicios necesita?", imagen="captura.png")`.

Los recursos de NLTK (`punkt_tab`, `stopwords`) se descargan solos la primera vez.

## Conjuntos de datos

**Intenciones (`data/intenciones.csv`).** 634 frases en las 9 intenciones del diseño conversacional, escritas
para el proyecto (con apoyo de IA) y revisadas: 508 de entrenamiento y 126 de prueba (80/20, 14 por intención),
112 marcadas como casos difíciles. La prueba se escribió antes de entrenar y no se usó para tomar decisiones del modelo.

**Pantallas (`vision/generar_pantallas.py`).** 1,360 capturas sintéticas de una banca digital ficticia en 8 tipos,
cada uno ligado a un dominio del RAML: `login` (autenticacion), `cuentas`, `transferencia` (transferencias),
`pago_servicios` (pagos), `prestamo` (prestamos), `seguros`, `documentos` (hipotecas: carga de documentos de la solicitud) y
`alta_cliente` (clientes). Varían banco, colores, modo claro/oscuro, tipografía, dispositivo, navegación,
campos y datos. Estilos "alta" (diseño final) y "front" (maqueta en grises) para entrenamiento (840),
validación (176) y prueba (184); el estilo "boceto" (160) nunca se usa para entrenar y mide la generalización.

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

Para usar otro inventario, reemplaza el archivo `.raml` (o cambia `RUTA_INVENTARIO` en `configuracion/rutas.py`).
Para revisar cómo se interpreta, ejecuta `python -m conocimiento.inventario_raml data/plataforma_servicios_bancarios.raml`.
Las conversaciones de prueba (`evaluacion/pruebas_conversaciones.py`, ejecutadas por `tests/test_chatbot.py`) hacen referencia a las operaciones como `"MÉTODO /endpoint"`, así que no
dependen del orden de los IDs; si cambias el RAML, revisa que sus mensajes y resultados esperados sigan aplicando.

## Intenciones soportadas

| Intención | Ejemplo |
|---|---|
| charla (saludo, ayuda, gracias, despedida) | "Hola", "ayuda", "hola, ¿me ayudas?" |
| buscar_servicio | "¿Existe un servicio para consultar el saldo?" |
| analizar_requerimiento | "Pantalla para buscar cliente, ver saldo y registrar pago" |
| detalle_servicio | "Detalle de SRV-007", "detalle del primero" |
| listar_dominios | "¿Qué servicios hay en el inventario?" |
| listar_por_dominio | "Servicios de pagos", "¿qué hay de seguros?" |
| reportar_brecha | "¿Qué hago si no existe el servicio?" |
| analizar_imagen | "Te comparto el mockup" (pide la imagen) o un turno con imagen adjunta |
| fuera_de_alcance | Cualquier consulta ajena al dominio |

## Resultados del Avance 2

Detalle en `evidencias/evaluacion_intenciones.txt` y `evidencias/evaluacion_pantallas.txt`.

| Componente (conjunto de prueba) | Exactitud | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM de intenciones (126 frases) | 0.857 | 0.856 | 0.857 | 0.855 |
| Asistente integrado: SVM + corrección por inventario (optimista: diseñado conociendo los errores del SVM) | 0.889 | 0.892 | 0.889 | 0.887 |
| ResNet50 + Dense softmax (Keras), pantallas de prueba (184) | 0.967 | 0.970 | 0.967 | 0.967 |
| ResNet50 + Dense softmax (Keras), bocetos no vistos (160) | 0.800 | 0.859 | 0.800 | 0.789 |
| Sin transferencia de aprendizaje (píxeles), prueba | 0.408 | 0.506 | 0.408 | 0.428 |

Precision, recall y F1 son promedios macro. Pruebas: `python -m pytest` → 81 aprobadas y 3 fallos esperados
(C13 y C18: el SVM clasifica mal "¿Qué ropa uso para mañana?" y "…el cliente no existe…"; M04: el boceto de
seguros se clasifica como préstamo con confianza 0.46, por lo que el asistente advierte la duda; se documentan con
`xfail` en lugar de corregirlos con reglas ad hoc). Conversaciones multimodales: 13/14 turnos correctos.

Limitaciones principales: el conjunto de intenciones y las pantallas son sintéticos; ambos modelos se ajustan
casi al 100 % en entrenamiento; `buscar_servicio` y `fuera_de_alcance` se confunden por el sesgo de "necesito";
el clasificador visual generaliza peor al estilo boceto (80.0 %; `alta_cliente` se confunde con `documentos`)
y su confianza es alta incluso cuando se equivoca; el clasificador visual reconoce el tipo de pantalla, no sus campos;
ResNet50 se usa congelada (sin ajuste fino); con Keras se puede descongelar en el siguiente avance.

## Resultados del Avance 1 (referencia)

- `python -m pytest tests/test_chatbot.py`: 19 conversaciones, 37 turnos. Cada turno verifica la intención y,
  en búsquedas y detalles, el servicio que nombra la respuesta (o que no nombre ninguno, cuando se espera una
  brecha). C13–C19 reproducen errores encontrados en las revisiones. Con las reglas del Avance 1 los 37 turnos
  eran correctos; con el clasificador SVM del Avance 2: 35/37 intenciones, 21/21 servicios y 35/37 turnos
  (C13 y C18 son los 2 fallos esperados; ver `evidencias/transcripcion_pruebas.txt`).
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
