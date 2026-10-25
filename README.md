# Asistente para la identificación de servicios REST — Avance 1

Chatbot basado en reglas (Python 3.10+ y NLTK) que relaciona requerimientos funcionales o descripciones de pantallas con un inventario de servicios REST, recomienda los servicios existentes y señala brechas de cobertura.

Materia: Desarrollo de aplicaciones con inteligencia artificial — Maestría en Inteligencia Artificial.

## Estructura

```
chatbot-tm/
├── Avance1_Chatbot_Servicios_REST.ipynb   # Notebook (VS Code / Google Colab)
├── chatbot_servicios.py                   # Motor del chatbot (reglas + NLTK)
├── pruebas_conversaciones.py              # 12 conversaciones de prueba + prueba de generalización
├── data/inventario_servicios.json         # Base de conocimiento (17 servicios REST)
├── evidencias/                            # Transcripciones y resultados de las pruebas
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
python pruebas_conversaciones.py   # 12 conversaciones + generalización; guarda evidencias/
python chatbot_servicios.py        # modo interactivo
```

Los recursos de NLTK (`punkt_tab`, `stopwords`) se descargan solos la primera vez.
El notebook `Avance1_Chatbot_Servicios_REST.ipynb` también corre en VS Code y en Google Colab.

## Cambiar el inventario

Edita `data/inventario_servicios.json`. Cada servicio necesita: `id`, `nombre`, `dominio`, `metodo`, `endpoint`,
`descripcion`, `palabras_clave` (lista), `entrada`, `salida` y `equipo`. Después ajusta los mensajes y resultados
esperados en `pruebas_conversaciones.py` y vuelve a ejecutar las pruebas.

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

- 12 conversaciones, 27 turnos: 27/27 intenciones correctas (26/27 en la primera ejecución; se corrigió un error de normalización).
- Prueba de generalización con 10 paráfrasis no vistas: 7/10 recomendaciones correctas.
