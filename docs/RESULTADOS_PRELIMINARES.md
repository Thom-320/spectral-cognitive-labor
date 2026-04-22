# RESULTADOS PRELIMINARES

## Estado del dataset

Los archivos crudos locales coinciden con el repo oficial de SODCL:

- `data/raw/humans_only_absent.csv`
- `data/raw/parameter_fit_humans.csv`
- `data/raw/performances.csv`

Punto metodologico clave:

- `performances.csv` contiene las 60 rondas por diada.
- `humans_only_absent.csv` contiene solo rondas con unicornio ausente y ademas incluye metricas derivadas.
- Por eso el analisis espectral y el analisis de prediccion temprana deben escribirse como **reanalisos sobre rondas absent**, no como si usaran las 60 rondas completas.

## Geometria del problema

- Grafo del tablero: `P_8 ⊠ P_8`
- Nodos: `64`
- Aristas: `210`
- `lambda_2 = lambda_3 ≈ 0.4164`
- `h(Fiedler ingenuo) = 28/210 = 0.1333`
- `h(LR) = h(TB) = h(best axis-aligned) = 22/210 = 0.1048`
- En este documento, `22/210` se usa como baseline axis-aligned de referencia dentro de la familia comparada, no como optimo global de `h(G)`.

Lectura critica:

- la degeneracion del caso cuadrado es real,
- pero el baseline axial de referencia coincide numericamente con la mejor referencia symmetry-aware explorada en la familia comparada.

## Reanalisis espectral tardio

Flujo de muestra:

- diadas auditadas: `45`
- set primario de conductancia: `29`
- excluidas por particion muy pequena: `8`
- excluidas por particion muy grande: `8`

Distribucion cualitativa de las 16 excluidas:

- `ALL`: `7`
- `RS`: `6`
- `NOTHING`: `3`

Resultados en el set primario:

- axiales: `h_obs = 0.142`, `eta = 1.120`, `DLIndex = 0.919`, `MI = 0.839`, `JSD = 0.842`
- mixed: `h_obs = 0.714`, `eta = 0.196`, `DLIndex = 0.514`, `MI = 0.260`, `JSD = 0.284`

Tests axial vs mixed:

- `h_obs`: `p = 3.07e-05`
- `DLIndex`: `p = 1.33e-04`
- `MI`: `p = 4.23e-04`
- `JSD`: `p = 3.50e-04`

Gate de novedad del lente espectral:

- Spearman `eta` vs `DLIndex_mean`: `rho = 0.894`, `p = 6.23e-11`

Interpretacion:

- el lente espectral si separa axial vs mixed,
- pero no esta capturando una dimension completamente nueva frente a la medida original.

## Prediccion temprana

Target:

- `axial_target = 1` si la diada termina con orientacion estable `LR` o `TB`
- `axial_target = 0` en caso contrario

Senal temprana:

- `dominant_score` del campo firmado de visitas en las primeras `5` rondas absent comunes

Comparacion principal sobre las 45 diadas:

- geometria temprana sola: `AUC in-sample = 0.834`, `AUC LOOCV = 0.804`
- metricas tempranas oficiales (`DLIndex + Similarity + Consistency`): `AUC in-sample = 0.808`, `AUC LOOCV = 0.736`
- modelo combinado: `AUC in-sample = 0.890`, `AUC LOOCV = 0.860`

Lectura:

- la senal geometrica temprana sola ya supera al trio oficial en LOOCV,
- y la combinacion de ambas familias de variables es la mejor.

## Transferencia a desempeno posterior

Usando la orientacion estable inferida desde rondas absent y luego mirando `performances.csv`:

### Rondas con unicornio ausente

- AXIAL: `accuracy = 0.977`, `score = 21.789`, `joint = 8.169`
- MIXED: `accuracy = 0.900`, `score = 9.189`, `joint = 13.183`

### Rondas con unicornio presente

- AXIAL: `accuracy = 0.936`, `score = 24.430`, `joint = 2.192`
- MIXED: `accuracy = 0.688`, `score = -6.142`, `joint = 5.386`

Lectura:

- la ruptura de simetria hacia orientaciones axiales estables acompana mejor coordinacion general,
- especialmente en rondas `present`.

## Valor incremental de la conductancia tardia

Diagnostico LOOCV sobre las 29 diadas del set primario, prediciendo desempeno `present`:

- modelo oficial (`DLIndex + Similarity + Consistency`) para `score_present`: `R2 = 0.654`
- oficial + `h_obs`: `R2 = 0.497`
- oficial + `eta`: `R2 = 0.575`

- modelo oficial para `acc_present`: `R2 = 0.628`
- oficial + `h_obs`: `R2 = 0.444`
- oficial + `eta`: `R2 = 0.514`

Conclusion:

- `h_obs` tardia no agrega valor predictivo incremental claro sobre las metricas oficiales,
- asi que la historia buena no es "mi metrica derrota a las viejas",
- sino "la geometria temprana ayuda a anticipar la cristalizacion de roles".

## Conclusion operativa

La mejor version del proyecto hoy es:

- ruptura de simetria en una geometria degenerada,
- baseline espectral correcto,
- prediccion temprana de especializacion axial,
- y transferencia cautelosa a desempeno posterior.

Eso ya sirve para:

- segunda entrega fuerte del curso,
- paquete serio para mostrar a Andrade,
- y base razonable de un preprint corto si luego se limpia y se escribe en ingles.
