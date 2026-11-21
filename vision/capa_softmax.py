"""
Capa de clasificación del componente visual: Dense con softmax, construida y entrenada con Keras (Avance 2)

    2048 características de ResNet50 -> normalización -> Dense de 8 neuronas + softmax -> 8 probabilidades

Es la única parte del clasificador visual que se entrena (transferencia de aprendizaje por
extracción de características): ResNet50 sólo describe cada imagen y esta capa aprende a
distinguir los 8 tipos de pantalla a partir de esa descripción.

La regularización L2 penaliza los pesos grandes; su intensidad se elige con el conjunto de
validación en evaluacion/evaluar_pantallas.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEMILLA = 42
EPOCAS = 80               # pasadas completas por los ejemplos de entrenamiento
TAMANO_LOTE = 32          # ejemplos por actualización de los pesos
TASA_APRENDIZAJE = 1e-3   # tamaño del paso del optimizador Adam


def construir_red(numero_caracteristicas: int, numero_clases: int, normalizacion, regularizacion_l2: float):
    """Red de Keras: entrada de características -> normalización -> Dense softmax (única capa entrenable)."""
    import keras  # importación diferida: Keras tarda en cargar y sólo se necesita con imágenes

    red = keras.Sequential([
        keras.Input(shape=(numero_caracteristicas,)),
        normalizacion,                                            # cada característica con media 0 y desviación 1
        keras.layers.Dense(numero_clases, activation="softmax",   # un puntaje por tipo de pantalla -> probabilidades
                           kernel_regularizer=keras.regularizers.L2(regularizacion_l2)),
    ])
    # Entropía cruzada: castiga asignar poca probabilidad al tipo correcto; Adam ajusta los pesos
    red.compile(optimizer=keras.optimizers.Adam(TASA_APRENDIZAJE),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return red


class CapaSoftmax:
    """Capa Dense de 8 neuronas con softmax sobre las 2048 características de ResNet50.

    Ofrece entrenar, predecir y predecir_probabilidades; la evaluación (evaluar_pantallas.py) y el
    clasificador (clasificador_visual.py) sólo usan estos métodos.
    """

    def __init__(self, regularizacion_l2: float):
        """Capa sin entrenar; regularizacion_l2 más alta = pesos más pequeños = modelo más simple."""
        self.regularizacion_l2 = regularizacion_l2
        self.red = None                # red de Keras, se crea al entrenar o al cargar
        self.clases = None             # nombre del tipo de pantalla de cada salida de la red
        self.historial = {}            # pérdida y exactitud por época (curva de entrenamiento)

    def entrenar(self, caracteristicas: np.ndarray, tipos: np.ndarray) -> "CapaSoftmax":
        """Ajusta la capa a las características y sus tipos de pantalla; guarda la curva de entrenamiento."""
        import keras

        keras.utils.set_random_seed(SEMILLA)                    # resultados reproducibles
        self.clases = np.array(sorted(set(tipos)))
        numeros_de_clase = np.searchsorted(self.clases, tipos)  # tipo de pantalla -> número de salida
        normalizacion = keras.layers.Normalization()
        normalizacion.adapt(caracteristicas)                    # aprende la media y la desviación de cada característica
        self.red = construir_red(caracteristicas.shape[1], len(self.clases), normalizacion, self.regularizacion_l2)
        resultado = self.red.fit(caracteristicas, numeros_de_clase, epochs=EPOCAS, batch_size=TAMANO_LOTE,
                                 shuffle=True, verbose=0)
        self.historial = {medida: [float(valor) for valor in valores] for medida, valores in resultado.history.items()}
        return self

    def predecir_probabilidades(self, caracteristicas: np.ndarray) -> np.ndarray:
        """Probabilidad de cada tipo de pantalla por imagen (columnas en el orden de self.clases)."""
        return np.asarray(self.red.predict(caracteristicas, batch_size=256, verbose=0))

    def predecir(self, caracteristicas: np.ndarray) -> np.ndarray:
        """Tipo de pantalla más probable de cada imagen."""
        return self.clases[self.predecir_probabilidades(caracteristicas).argmax(axis=1)]

    def guardar(self, ruta: Path) -> None:
        """Guarda la red (.keras) y, junto a ella, los nombres de las clases y la regularización (.json)."""
        ruta.parent.mkdir(exist_ok=True)
        self.red.save(str(ruta))
        ruta.with_suffix(".json").write_text(json.dumps({"clases": self.clases.tolist(),
                                                         "regularizacion_l2": self.regularizacion_l2}))

    @classmethod
    def cargar(cls, ruta: Path) -> "CapaSoftmax | None":
        """La capa guardada, o None si no existe, si Keras no está instalado o si no se puede abrir."""
        ruta_datos = Path(ruta).with_suffix(".json")
        if not Path(ruta).exists() or not ruta_datos.exists():
            return None
        try:
            import keras
        except ImportError:
            return None
        datos = json.loads(ruta_datos.read_text())
        capa = cls(datos["regularizacion_l2"])
        try:
            capa.red = keras.models.load_model(str(ruta))
        except Exception:  # archivo dañado o guardado con una versión de Keras incompatible
            return None
        capa.clases = np.array(datos["clases"])
        return capa


def crear_capa_softmax(regularizacion_l2: float) -> CapaSoftmax:
    """Capa Dense softmax sin entrenar, con la regularización indicada."""
    return CapaSoftmax(regularizacion_l2)
