# Auditoría del dataset TLM:UAV

Fecha de auditoría: 2026-07-19  
Alcance: copia local incluida en `dataset/`  
Estado: **reconstrucción y protocolo de datos listos para un benchmark exploratorio; evidencia externa aún limitada a dos vuelos inferidos**

## 1. Conclusión ejecutiva

Las advertencias iniciales no eran todas demostrables tal como estaban formuladas, pero la auditoría confirmó problemas estructurales suficientes para impedir un split aleatorio por filas y para descartar `Fusion_Data.csv` como fuente canónica del experimento.

Los hallazgos más importantes son:

1. La copia local contiene datos de al menos dos vuelos en los archivos combinados. `TimeUS` se repite entre vuelos y no funciona como clave global.
2. `Fusion_Data.csv` puede reconstruirse exactamente a partir de una interpolación inclusiva de ATT y MAG, truncada por cantidad de filas en vez de por tiempo común.
3. La fusión duplica por error `ErrRP` como `ErrYaw` y `MagY` como `MagZ`. Las fuentes originales no contienen esas duplicaciones.
4. Las señales IMU de la tabla fusionada avanzan hasta `521503815`, mientras el timestamp interpolado de ATT termina en `439963944`. Esto demuestra deriva entre modalidades.
5. Las 551 filas etiquetadas como clase 4 en la fusión se alinean temporalmente con clase 0 en ATT. El episodio de clase 4 de ATT comienza después de que termina la tabla fusionada.
6. Un split aleatorio estratificado separa timestamps interpolados repetidos entre train y test y coloca casi todas las filas de test junto a una fila de entrenamiento.
7. Un split cronológico 70/15/15 no conserva soporte multiclase: train no contiene clases 3 ni 4, validación contiene solo 0 y 3, y test contiene 0, 3 y 4.
8. Solo se recuperan dos vuelos mediante tiempo GPS absoluto. Esto permite una evaluación externa por vuelo, pero sigue siendo una muestra extremadamente pequeña de unidades independientes.

Por tanto, la cantidad de filas no debe confundirse con cantidad de vuelos
independientes. El protocolo implementado reconstruye primero `flight_id` y solo
después define los folds, las transformaciones y los modelos.

## Criterio de certeza

El informe distingue tres niveles:

- **Hecho reproducido:** surge directamente de valores, hashes o reconstrucciones exactas. Incluye timestamps duplicados, asignaciones erróneas de columnas, bloques de labels y contaminación entre splits.
- **Inferencia fuerte:** la separación mediante tiempo GPS absoluto identifica dos sesiones de vuelo. Los CSV no incluyen una columna oficial `flight_id`, por lo que el nombre del grupo sigue siendo inferido.
- **Hipótesis pendiente:** que una variable sea un proxy de misión, que un modelo memorice contexto o que una señal no generalice a otro UAV. Estas afirmaciones requieren ablaciones y vuelos externos; no se declaran demostradas por correlación o por nombre de columna.

Esta distinción evita convertir una auditoría crítica en una acusación no sustentada sobre todo el dataset.

## 2. Procedencia y significado de las clases

La [ficha oficial de Kaggle](https://www.kaggle.com/datasets/luyucwnu/tlmuav-anomaly-detection-datasets) define las clases así:

| Label | Significado |
|---:|---|
| 0 | Operación normal |
| 1 | Falla de GPS |
| 2 | Falla del acelerómetro |
| 3 | Falla de motor |
| 4 | Falla del control remoto (RC) |

El dataset fue obtenido en simulación software-in-the-loop y procesado mediante Time Line Modeling. El artículo de adquisición describe anclaje de ventanas de falla y ampliación temporal para balancear datos: [Yang et al., 2023](https://doi.org/10.3390/app13074301).

La tabla fusionada se asocia con el método MHSIA del trabajo de fusión multisensor: [Deng et al., 2024](https://doi.org/10.1016/j.engappai.2024.107961). El [código público de `Fusion.py`](https://github.com/FishLuYu/IF-HMNN/blob/main/Fusion.py) confirma el uso de interpolación lineal con tres puntos inclusivos por intervalo.

Estas fuentes describen la intención metodológica. La validez de la copia local se evaluó independientemente a partir de sus valores.

## 3. Instantánea local

- 17 archivos CSV.
- 235 columnas contando cada esquema por separado.
- Cero celdas ausentes según `pandas.isna`.
- Cero filas exactamente duplicadas dentro de cada archivo.
- Nombres de target heterogéneos: `labels` y `lables`.
- Nombres temporales heterogéneos: `TimeUS`, `abTimeUS` y `timestamp`.
- Nombres de índice heterogéneos: `LineNo` y `LineNO`.

Archivos clave:

| Archivo | Filas | Columnas | Observación |
|---|---:|---:|---|
| `Fusion_Data.csv` | 12.253 | 19 | ATT + MAG interpolados y seis señales IMU |
| `GPS/ALL_FAIL_LOG_GPS_0.csv` | 2.450 | 17 | Primer vuelo, aproximadamente 5 Hz |
| `RATE/ALL_FAIL_LOG_RATE.csv` | 4.900 | 15 | Primer vuelo, aproximadamente 10 Hz |
| `VIBE/ALL_FAIL_LOG_VIBE_0_Random.csv` | 4.900 | 9 | Primer vuelo reordenado |
| `AddNum/...GPS...csv` | 4.319 | 17 | Dos vuelos combinados y reordenados |
| `AddNum/...RATE...csv` | 8.638 | 15 | Dos vuelos combinados y reordenados |
| `AddNum/...VIBE...csv` | 8.638 | 8 | Dos vuelos combinados y reordenados |
| `IMU/...Random.csv` | 21.597 | 17 | Dos vuelos combinados y reordenados |

`AddNum/ALL_FAIL_LOG_IMU_0_Add_Random.csv` e `IMU/ALL_FAIL_LOG_IMU_0_Random.csv` son el mismo archivo byte por byte:

```text
09774d00b923d3a7b053727c78acd50d339572b8ec5fb2b4fa0af1d86a0185b5
```

La huella SHA-256 de la tabla fusionada auditada es:

```text
e17dc97292bbf32f96424902a78c64d23b8185fe6002d5f008a4d7f813244465
```

## 4. Unidad estadística y estructura temporal

En ATT, el target forma exactamente ocho bloques contiguos:

```text
normal -> GPS -> normal -> acelerómetro -> normal -> motor -> normal -> RC
```

Cada clase de falla aparece en un único episodio dentro del primer vuelo. Filas consecutivas están separadas normalmente por unos `99960` microsegundos y representan estados vecinos de la misma dinámica, no experimentos independientes.

Los archivos llamados `Random` no comparten una única interpretación:

- VIBE y XKF1 contienen las 4.900 muestras del primer vuelo en orden aleatorio; al ordenar por `TimeUS` recuperan los ocho bloques.
- Los archivos `AddNum` combinan dos vuelos y luego reordenan filas.
- IMU también contiene dos vuelos y es idéntico a su copia bajo `AddNum`.

Por esto, ordenar un archivo combinado solo por `TimeUS` mezcla vuelos distintos. Se necesita la clave compuesta `(flight_id, TimeUS)`.

## 5. Evidencia de dos vuelos

GPS conserva `GWk` y `GMS`. La variable

```text
gps_absolute_ms = GWk * 604800000 + GMS
```

revela dos registros separados por `155400000` ms, aproximadamente 43,2 horas:

| Vuelo inferido | Filas GPS | Duración | Semana GPS | Conteos 0/1/2/3/4 |
|---:|---:|---:|---:|---|
| 0 | 2.450 | 506,0 s | 2232 | 1231 / 305 / 297 / 241 / 376 |
| 1 | 1.869 | 412,4 s | 2233 | 800 / 264 / 321 / 173 / 311 |

La trazabilidad exacta depende de la modalidad:

- GPS conserva las 2.450 filas del primer registro y añade 1.869; ambas sesiones se separan directamente con tiempo GPS absoluto.
- VIBE conserva exactamente sus 4.900 filas de referencia dentro de las 8.638 combinadas.
- RATE solo conserva coincidencia exacta para 4.512 de sus 4.900 filas de referencia, incluso al excluir el label. Su total de 8.638 no demuestra por sí solo una concatenación limpia.
- IMU tiene 21.597 filas y no incluye un archivo de referencia limpio separado. La descomposición `12253 + 9344` es compatible con las tasas y la trazabilidad parcial de la fusión, pero sigue siendo una inferencia.

Este hallazgo corrige una interpretación preliminar: los labels incompatibles bajo un mismo `TimeUS` no demuestran por sí solos corrupción de etiqueta. En muchos casos corresponden a vuelos distintos cuyo reloj relativo coincide. El defecto es la pérdida de `flight_id`.

## 6. Consistencia de labels entre sensores raw

Se alinearon por `TimeUS` diez fuentes del primer vuelo: ATT, BARO, BAT, CTUN, MAG, MOTB, PSCD, RATE, VIBE y XKF1.

- Timestamps comunes: 4.817.
- Acuerdo completo entre las diez fuentes: 4.776.
- Desacuerdos: 41.
- Tasa de acuerdo: 99,1489 %.

Los desacuerdos se concentran alrededor de transiciones de falla y tienen longitudes distintas según sensor. Esto es compatible con tasas, anclajes o propagaciones diferentes, pero no permite elegir automáticamente una etiqueta verdadera.

La reconstrucción implementada declara explícitamente:

- VIBE como reloj y referencia reproducible del target;
- tolerancias causales de 220 ms para GPS y 45 ms para IMU;
- una sensibilidad secundaria de ±0,25 s en bordes de episodio;
- conservación de labels auxiliares en una tabla aislada, sin votación silenciosa.

## 7. Auditoría de `Fusion_Data.csv`

### 7.1 Interpolación reproducida

Para cada par consecutivo, el código oficial genera:

```text
[t_i, (t_i + t_(i+1)) / 2, t_(i+1)]
```

Como ambos extremos están incluidos, `t_(i+1)` vuelve a aparecer al iniciar el siguiente intervalo. Esto explica exactamente:

- 12.253 filas;
- 8.169 timestamps únicos;
- 4.084 timestamps duplicados;
- cero deltas negativos;
- la secuencia completa de `timestamp` con error absoluto cero.

Las columnas ATT y MAG de la fusión se reproducen con error numérico menor que `1e-9` mediante esa misma interpolación.

### 7.2 Columnas copiadas incorrectamente

En la tabla fusionada:

```text
ErrYaw == ErrRP
MagZ   == MagY
```

en las 12.253 filas. En las fuentes originales:

- `ATT.ErrYaw` difiere de `ATT.ErrRP` en 4.595 de 4.900 filas.
- `MAG.MagZ` difiere de `MAG.MagY` en las 4.900 filas.

No basta con eliminar una columna duplicada: la información correcta de `ErrYaw` y `MagZ` se perdió en la exportación fusionada.

### 7.3 Truncación y deriva temporal

La interpolación de ATT/MAG produce tres filas por intervalo, pero IMU tiene otra tasa. La tabla final se corta a 12.253 filas. Eso consume solo los primeros 4.085 de los 4.900 registros ATT.

- Último timestamp ATT representado en la fusión: `439963944`.
- Inicio de clase 4 en ATT: `446363883`.
- Máximo `abTimeUS` de las señales IMU trazables en la fusión: `521503815`.

Se pudieron mapear de forma unívoca 12.065 de las 12.253 filas IMU de la fusión al archivo combinado. Su `abTimeUS` es monotónico en el orden fusionado y recorre prácticamente todo el primer vuelo. Las modalidades, por tanto, avanzan a velocidades temporales distintas dentro de una misma fila.

Consecuencia observable: las 551 filas con label 4 de `Fusion_Data.csv` corresponden al estado normal de ATT cuando se comparan por el `timestamp` almacenado.

### 7.4 Decisión

`Fusion_Data.csv` puede utilizarse para estudiar cómo se comporta un modelo frente a la tabla publicada, pero no como evidencia principal de detección física multisensor. El benchmark serio debe reconstruirse desde fuentes con `flight_id` y alineación temporal explícita.

## 8. Diagnóstico de splits

Se evaluó un split 70/15/15 estratificado por filas, con semilla 42, únicamente para auditar contaminación:

- Train: 8.577 filas.
- Validación: 1.838 filas.
- Test: 1.838 filas.
- Timestamps únicos compartidos entre train y test: 891.
- Filas de test cuyo timestamp ya aparece en train: 891.
- 90,97 % de test tiene una fila de train inmediatamente adyacente.
- 99,13 % de test está a dos filas o menos de train.

Este split mide interpolación local dentro de los mismos episodios y puede separar entre conjuntos dos filas generadas desde el mismo instante fuente. No es una estimación defendible de generalización a otro vuelo.

El split cronológico 70/15/15 tampoco resuelve el problema multiclase:

| Split | Clases presentes |
|---|---|
| Train | 0, 1, 2 |
| Validación | 0, 3 |
| Test | 0, 3, 4 |

Un clasificador no puede aprender clases 3 y 4 si no aparecen en train. La caída de rendimiento bajo este split mezclaría generalización temporal con clases no observadas.

## 9. Variables y riesgo de proxies

Las siguientes categorías se tratarán por separado:

| Grupo | Ejemplos | Tratamiento inicial |
|---|---|---|
| Identificadores y reloj | `LineNo`, `LineNO`, `TimeUS`, `abTimeUS`, `timestamp`, `GMS`, `GWk` | Nunca como predictores principales |
| Señales físicas instantáneas | actitud medida, giroscopio, aceleración, magnetómetro, vibración | Conjunto físico principal, sujeto a auditoría |
| Referencias de control | `DesRoll`, `DesPitch`, `DesYaw`, `RDes`, `PDes`, `YDes` | Ablación contextual separada |
| Errores y salidas de control | `ErrRP`, `ErrYaw`, `Rout`, `POut`, `YOut` | Ablación diagnóstica separada |
| Estado acumulativo o de misión | `CurrTot`, `EnrgTot`, batería, posición GPS, altitud | Revisar como posibles proxies de progreso |
| Constantes y duplicados | columnas de una sola modalidad sin variación | Excluir tras ajuste con train |

Una variable de control no es leakage por definición. Puede ser válida si el objetivo es diagnóstico dentro del lazo de control disponible en operación. Sí sería problemática si se afirma detección generalizable basada exclusivamente en sensores físicos. Por eso se compararán conjuntos de features y no se mezclará esta decisión con el ranking de modelos.

El benchmark actual no usa mutual information, PCA ni selección supervisada: las
dos vistas se fijaron semánticamente antes del entrenamiento. Imputación, escalado,
constantes y categorías se ajustan únicamente con las filas de desarrollo que
corresponden a cada fase. Cualquier selector futuro deberá respetar esa misma regla.

## 10. Protocolo experimental aplicado

1. `Fusion_Data.csv` no se utiliza como dataset principal.
2. `flight_id` se reconstruye para GPS, RATE, VIBE e IMU sin consultar labels.
3. Los dos vuelos se mantienen separados y los sensores se alinean dentro de vuelo.
4. VIBE se declara como reloj y referencia reproducible del target; los labels
   auxiliares permanecen aislados para auditar bordes discordantes.
5. La evaluación externa es leave-one-flight-out: un vuelo completo para test y el
   otro para desarrollo.
6. La época se selecciona mediante bloques por episodio con purga temporal dentro
   del vuelo de desarrollo.
7. La evaluación se repite intercambiando el vuelo externo y ambas direcciones se
   reportan por separado.
8. Los dos folds no se interpretan como una estimación precisa de variabilidad
   poblacional.
9. Se reportan balanced accuracy, F1 macro, recall por clase, MCC, matrices de
   confusión y métricas multiclase basadas en probabilidades.
10. `sensor_core` y `full_diagnostic` forman la ablación predefinida de contexto.

Con solo dos vuelos no existe un train/validation/test completamente independiente a nivel de vuelo. La solución anterior evita usar el vuelo externo para selección, pero la evidencia seguirá siendo exploratoria.

## 11. Preguntas abiertas

- ¿Existe una versión original con `flight_id` explícito para todas las modalidades?
- ¿Cómo se generó exactamente el segundo vuelo de cada archivo `AddNum`?
- ¿Qué sensor y qué regla se usaron para construir el label final de la fusión?
- ¿Por qué el código público de fusión está incompleto y la tabla contiene dos asignaciones de columna incorrectas?
- ¿Se dispone de ATT y MAG del segundo vuelo para reconstruir una fusión comparable?
- ¿Las columnas `LineNo` conservan suficiente información para recuperar grupos fuera de GPS?

La identidad de vuelo ya puede reconstruirse con evidencia fuerte, aunque sigue sin
ser una anotación oficial del dataset. La fuente original usada para crear el label
fusionado continúa siendo desconocida. Por ello, el protocolo adopta VIBE como
referencia reproducible y mantiene esta decisión separada de cualquier afirmación
sobre una verdad física superior.

## 12. Reconstrucción y alineación implementadas

La primera etapa del benchmark quedó implementada en `src/data.py` y documentada
de forma ejecutable en `01_multiclass_classification.ipynb`.

La reconstrucción no consulta labels y produce los siguientes conteos:

| Fuente | Vuelo 0 | Vuelo 1 | Evidencia principal |
|---|---:|---:|---|
| GPS | 2.450 | 1.869 | Tiempo GPS absoluto |
| VIBE | 4.900 | 3.738 | Coincidencia exacta sin target con el archivo de referencia |
| RATE | 4.900 | 3.738 | Timestamp VIBE exacto y línea de log más próxima |
| IMU | 12.253 | 9.344 | Trayectoria VIBE y anclas IMU únicas de procedencia |

Para IMU, 12.065 filas se identifican de forma única mediante las seis señales
presentes en la fusión. El 99,9917 % de esas anclas coincide con una misma
trayectoria antes de aplicar la regla; una sola fila situada cerca del cruce inicial
queda corregida por esa evidencia de procedencia. Tras la asignación, tiempo y
número de línea son estrictamente crecientes en ambos vuelos.

La alineación utiliza VIBE como reloj canónico y label de referencia:

- RATE se une en el mismo `TimeUS`;
- GPS usa la última observación anterior dentro de 220 ms;
- IMU usa la última observación anterior dentro de 45 ms;
- ninguna observación futura es admisible;
- cuatro filas normales sin GPS causal reciente se descartan y quedan registradas.

La tabla resultante tiene 8.634 filas, 43 variables de sensores y cinco columnas de
metadatos o target. Los labels auxiliares se almacenan en un CSV separado. Los
desacuerdos respecto de VIBE son inferiores a 0,33 % para cada sensor y vuelo.

Una prueba adicional permuta todos los labels en memoria y recupera exactamente
los mismos `flight_id`. Dos ejecuciones completas producen fingerprints y
artefactos idénticos byte por byte. El manifiesto de los 17 CSV fuente permanece
sin cambios.

Esta etapa hace viable el diseño leave-one-flight-out, pero no elimina la
limitación central: solo existen dos vuelos inferidos y un episodio de cada falla
por vuelo.

## 13. Protocolo de datos congelado antes del entrenamiento

La etapa posterior define dos vistas sin utilizar rankings supervisados:

| Vista | Numéricas | Categóricas | Propósito |
|---|---:|---:|---|
| `sensor_core` | 15 | 0 | Señales físicas instantáneas sin posición, referencias, salidas ni flags |
| `full_diagnostic` | 29 | 3 | Telemetría operacional no constante, incluido contexto y control |

`full_diagnostic` no se interpreta como libre de leakage. Su comparación con
`sensor_core` medirá sensibilidad a información contextual, pero no demostrará por
sí sola qué variable es causal.

Se construyen dos folds externos:

| Desarrollo | Test | Inner train | Validación | Purga | Test completo |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 3.132 | 975 | 791 | 3.736 |
| 1 | 0 | 2.198 | 744 | 794 | 4.898 |

La validación toma el bloque central del 20 % de cada episodio y elimina del
inner train las muestras situadas hasta cinco segundos alrededor. La distancia
mínima observada entre validación y train es `5,000499 s`. Todas las clases quedan
representadas en inner train, validación y test.

Esta partición interna usa el target para conservar soporte multiclase y comparte
episodios. Por tanto, se empleará exclusivamente para early stopping y selección
de época. No se presentará como estimación de generalización.

El preprocesamiento tiene dos estados independientes:

1. Estado de tuning: ajustado solo con inner train y aplicado a validación.
2. Estado final: reajustado con el vuelo de desarrollo completo y aplicado al
   vuelo externo después de congelar la época.

Medianas, medias, desviaciones, constantes, categorías y pesos de clase se
calculan únicamente con las filas de ajuste correspondientes. El índice categórico
0 queda reservado para valores desconocidos. No se aplica resampling.

La evaluación primaria conservará todo el vuelo externo. Una sensibilidad
secundaria excluirá ±0,25 s alrededor de cada transición; este margen cubre los
desacuerdos intersensor observados, que no superan 0,2 s. La máscara secundaria no
participará en entrenamiento, selección ni ranking principal.

Una prueba adversarial modifica únicamente las features del vuelo externo e
introduce categorías inéditas. Los scalers, encoders, pesos y matrices de
desarrollo permanecen idénticos, mientras los valores desconocidos se transforman
al índice reservado. Esto verifica de forma empírica que el test no ajusta el
pipeline.
