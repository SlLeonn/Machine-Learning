# Contexto de arquitecturas tabulares

Este documento conserva las decisiones aprendidas en la comparación anterior de deep learning tabular y especifica cuáles pueden reutilizarse en TLM:UAV. No reemplaza la auditoría del dataset ni fija todavía un benchmark.

## 1. Arquitecturas ya trabajadas

### TabNet

- Implementación comprobada con `pytorch-tabnet==4.1.0`.
- Usa pasos de decisión secuenciales y máscaras atencionales dispersas sobre columnas.
- Puede consumir variables numéricas directamente y categorías mediante embeddings y cardinalidades explícitas.
- Sus máscaras describen selección interna de variables; no deben interpretarse como importancia causal.
- El checkpoint nativo es un archivo ZIP y su restauración debe verificarse numéricamente.

### TabTransformer

- Contextualiza embeddings de columnas categóricas mediante self-attention.
- Las variables numéricas se incorporan después mediante una rama continua y una cabeza conjunta.
- Cuando el dataset no tiene categorías, la parte distintiva del Transformer categórico queda inactiva. En ese caso el resultado representa una variante numérica degenerada, no una evaluación general de TabTransformer.

### FT-Transformer

- Convierte cada columna numérica o categórica en un token aprendido.
- La atención opera sobre todas las variables desde la primera capa.
- Un token de agregación resume la fila para clasificación o regresión.
- Su diseño es conceptualmente adecuado para tablas completamente numéricas, aunque puede sobreajustar con pocas unidades independientes.

### SAINT supervisado

- Tokeniza columnas y permite atención entre columnas y entre filas.
- La comparación justa anterior utilizó una variante inductiva con atención entre columnas y `use_row_attention=False`.
- La atención entre filas hace que una predicción dependa de la composición del batch y complica la inferencia fila a fila.
- En datos temporales, mezclar filas de distintos splits dentro de atención intersample sería una vía adicional de contaminación.
- No se utilizó pretraining contrastivo.

## 2. Interfaz experimental reutilizable

Los modelos deben presentar operaciones equivalentes:

```text
fit
predict
predict_proba
get_training_history
save
load
get_embedding, cuando sea viable
```

La lógica común debe conservar:

- configuración central de rutas, semillas, dispositivo y presupuesto;
- GPU cuando esté disponible y fallback explícito a CPU;
- early stopping basado solo en validación;
- restauración del mejor checkpoint antes de test;
- probabilidades validadas dentro de `[0, 1]`;
- historial, mejor época, parámetros y tiempos;
- test externo aislado hasta el final;
- preprocesamiento ajustado exclusivamente con train;
- misma partición y etiquetas para todas las arquitecturas.

## 3. Lecciones de comparación

No es necesario imponer hiperparámetros idénticos a arquitecturas con sesgos distintos. La justicia experimental exige un presupuesto comparable, no una configuración artificialmente uniforme.

Tres semillas de inicialización ayudan a medir inestabilidad de optimización, pero no crean nuevas unidades estadísticas. En TLM:UAV, repetir semillas sobre los mismos dos vuelos no sustituye disponer de más vuelos.

Un baseline lineal sencillo es suficiente para saber si la capacidad no lineal aporta algo. No se añadirá una colección de modelos de boosting porque el objetivo sigue siendo observar el comportamiento de las arquitecturas de deep learning tabular.

Las comparaciones deben reportar tanto rendimiento como costo. Un modelo marginalmente mejor puede no justificar muchos más parámetros, tiempo o sensibilidad a la semilla.

## 4. Adaptación provisional a TLM:UAV

La copia local de TLM es completamente numérica. Por tanto:

| Modelo | Adaptación prevista | Advertencia principal |
|---|---|---|
| TabNet | Señales continuas y tres estados categóricos en la vista diagnóstica | Máscaras no equivalen a causalidad física |
| TabTransformer | Solo rama numérica en `sensor_core`; tres tokens categóricos en `full_diagnostic` | Su contextualización categórica sigue siendo limitada |
| FT-Transformer | Un token por señal | Capacidad alta frente a solo dos vuelos |
| SAINT inductivo | Atención entre columnas, no entre filas | Mantener predicción independiente del batch |

La reconstrucción, la regla de labels, las vistas y los folds externos ya están
congelados. El pipeline de entrenamiento deberá consumir estos objetos sin volver a
dividir filas ni reajustar transformadores con el vuelo externo.

## 5. Protocolo que condicionará los modelos

La evaluación principal deberá dividir por vuelo, nunca por fila. El vuelo externo no participará en:

- escalado;
- selección de variables;
- cálculo de pesos de clase;
- elección de época;
- ajuste de umbral;
- selección de hiperparámetros.

La validación interna se hará con bloques temporales purgados del vuelo de desarrollo. Las dos direcciones de leave-one-flight-out se reportarán por separado.

La clasificación se estudiará en dos formulaciones distintas:

1. Multiclase: normal, GPS, acelerómetro, motor y RC.
2. Binaria: normal frente a cualquier anomalía.

Un buen resultado binario no demuestra identificación de tipos de falla. Un buen resultado multiclase dentro de los mismos episodios tampoco demuestra generalización a vuelos nuevos.

## 6. Métricas previstas

Para multiclase:

- balanced accuracy;
- F1 macro;
- precision y recall macro;
- recall por clase;
- matriz de confusión;
- ROC-AUC one-vs-rest cuando todas las clases estén presentes y haya probabilidades válidas.

Para binaria:

- balanced accuracy;
- precision, recall y F1 de anomalía;
- ROC-AUC y PR-AUC con probabilidades;
- tasa de falsas alarmas;
- matriz de confusión.

Las métricas por fila se acompañarán de análisis por bloques o episodios. Accuracy aislada no será una conclusión suficiente.

## 7. Estado actual

- Auditoría estructural implementada en `src/audit.py`.
- Hallazgos documentados en `DATASET_AUDIT.md`.
- `Fusion_Data.csv` no será la fuente principal.
- Reconstrucción y alineación causal implementadas en `src/data.py`.
- Cuaderno incremental creado en `01_multiclass_classification.ipynb`.
- GPS, RATE, VIBE e IMU quedan separados en dos vuelos inferidos y monotónicos.
- El target explícito es VIBE; los demás labels permanecen aislados para auditoría.
- La tabla alineada contiene 8.634 filas y las cinco clases en ambos vuelos.
- Vistas congeladas: `sensor_core` con 15 señales y `full_diagnostic` con 32.
- Dos folds leave-one-flight-out implementados con validación por episodio y purga
  temporal de cinco segundos.
- Preprocesamiento de tuning ajustado solo con inner train y estado final ajustado
  solo con el vuelo de desarrollo completo.
- El aislamiento del vuelo externo fue validado mediante perturbaciones
  adversariales de sus features y categorías.
- `src/models.py`, `src/training.py` y `src/evaluation.py` implementan la factoría,
  las dos fases de ajuste, la restauración de checkpoints y las métricas.
- `02_multiclass_benchmark.ipynb` es el ejecutable principal; los notebooks `00`
  y `01` conservan auditoría y preparación detallada como material complementario.
- El perfil técnico `smoke` completó 20 combinaciones reales con una semilla y
  dos épocas. Todas las probabilidades, filas, métricas y restauraciones pasaron
  sus verificaciones. Estos números no constituyen el benchmark final.
- El perfil `study` queda congelado con tres semillas, hasta 60 épocas y paciencia
  de 10. No se han ajustado sus hiperparámetros a partir de resultados externos.
- El perfil `study` completó sus 60 combinaciones en GPU. El manifiesto confirma
  la cuadrícula completa; las predicciones persistidas reproducen las métricas y
  los 60 checkpoints restauran probabilidades con diferencia máxima cero.
- Las tablas run a run, por vuelo, por clase, de costo y de comparación descriptiva
  se guardan bajo `results/benchmark/study/metrics/`.
- Los archivos originales permanecen intactos.

## 8. Entorno comprobado antes del modelado

- Python `3.14.3`.
- NumPy `2.4.4`, pandas `3.0.2` y scikit-learn `1.8.0`.
- PyTorch `2.13.0+cu130` con CUDA `13.0` y cuDNN disponible.
- `pytorch-tabnet` importa correctamente con su interfaz `TabNetClassifier`.
- GPU: NVIDIA GeForce RTX 2050, capacidad de cómputo 8.6 y 4 GB de memoria.
- Ambos notebooks actuales se ejecutan desde un kernel limpio con
  `allow_errors=False` y permanecen sin outputs guardados.

Los 4 GB de VRAM obligan a usar batches conservadores y a liberar memoria entre
modelos. SAINT se mantendrá en su variante inductiva sin atención entre filas; de
este modo, además de evitar dependencia entre muestras, se reduce el costo
cuadrático y el riesgo de agotar memoria. La disponibilidad de GPU no cambia las
reglas de split ni autoriza a reutilizar el vuelo externo durante selección.

El smoke test expuso ambos vuelos externos para validar la integración de extremo
a extremo. Desde ese momento, cualquier cambio de hiperparámetros motivado por su
rendimiento sería selección sobre test. Las configuraciones base se consideran
congeladas y los resultados smoke solo pueden interpretarse como pruebas técnicas.
