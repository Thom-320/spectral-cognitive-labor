# SODCL spectral-cognitive-labor: plan para volverlo publicable

## Veredicto

El proyecto sí tiene una semilla publicable, pero no bajo la tesis vieja de “los humanos vencen a Fiedler” ni como optimización espectral humana general. La ruta defendible es:

> En un tablero cuadrado con simetría diédrica, el eigenspace de Fiedler es degenerado; algunas díadas humanas estabilizan convenciones axiales que rompen esa simetría, y una señal geométrica temprana predice esa estabilización mejor que, o complementando a, métricas conductuales oficiales.

Esto es un preprint corto plausible si se fortalece la inferencia. Para journal, falta robustez estadística, null models y una reescritura completa en inglés.

## Hallazgos confirmados en el repo actualizado

- Dataset oficial SODCL: 45 díadas, 60 rondas por díada en `performances.csv`; `humans_only_absent.csv` contiene solo rondas con unicornio ausente y métricas derivadas.
- Grafo del tablero: `P_8 ⊠ P_8`, 64 nodos, 210 aristas.
- Degeneración: `lambda_2 = lambda_3 ≈ 0.4164`.
- Baselines: corte Fiedler ingenuo `h = 28/210 = 0.1333`; cortes axiales LR/TB `h = 22/210 = 0.1048`.
- Set primario tardío: 29 díadas, con 21 axiales y 8 mixed bajo la clasificación de ventana 40-60.
- Para orientación estable entre ventanas, el target predictivo tiene 20 axiales estables, 12 mixed válidas y 13 inválidas/no bipartitas.
- Predicción temprana, 45 díadas: geometría sola AUC LOOCV 0.804; métricas oficiales tempranas AUC LOOCV 0.736; combinado AUC LOOCV 0.860.
- Predicción temprana, solo etiquetas válidas: combinado AUC LOOCV 0.8125. Este es el número que conviene reportar junto al de 45 díadas, no esconderlo debajo de la alfombra académica.
- Transferencia descriptiva a desempeño: en rondas present, axiales tienen accuracy media 0.936 y score medio 24.43; mixed tienen accuracy 0.688 y score -6.14.
- Diagnóstico incremental negativo: `h_obs` tardía no mejora predicción de desempeño sobre `DLIndex + Similarity + Consistency`; a veces la empeora.

## Tesis publicable recomendada

Título provisional:

**Symmetry-aware graph signals predict role specialization in a degenerate spatial coordination task**

Claim principal:

> A geometry-derived early signal, defined on the first absent trials of a public dyadic coordination dataset, predicts later axial role specialization under leave-one-out validation and complements existing behavioral metrics.

Claim matemático:

> The square strong-product board induces a two-dimensional Fiedler eigenspace, making a single naive Fiedler cut non-canonical.

Claim empírico descriptivo:

> Late axial dyads have lower conductance, higher mutual information, higher Jensen-Shannon divergence, and more stable partitions than mixed dyads.

Claim negativo importante:

> Late conductance is strongly coupled to existing behavioral measures and should not be framed as replacing them.

## Qué hay que cambiar del paper

1. Reescribir `paper/preprint_base.tex`. Está más viejo que el README y no pone la predicción temprana en el centro.
2. Usar `paper/entrega2.tex` como marco teórico/metodológico, pero agregar resultados reales de `08_early_prediction.py` y `docs/RESULTADOS_PRELIMINARES.md`.
3. Pasar el manuscrito a inglés.
4. Cambiar el foco de “comparación espectral tardía” a “early prediction of symmetry breaking”.
5. Reportar siempre dos análisis: `all_dyads` y `valid_only`.
6. Convertir la transferencia a desempeño en análisis a nivel de díada, no a nivel de ronda. La unidad experimental es la díada, no cada fila del CSV, por mucho que a los p-values les guste inflarse como globos tristes.

## Análisis que faltan antes de arXiv serio

### 1. Inferencia de AUC

Crear `src/09_auc_inference.py`:

- Bootstrap estratificado por díada.
- Intervalos de confianza para AUC LOOCV.
- Permutation test para `axial_target`.
- Permutation test pareado para comparar `geometry_plus_official` vs `official_trio`.

### 2. Sensibilidad

Crear `src/10_sensitivity.py`:

- Variar `EARLY_COMMON_ROUNDS`: k = 1,...,10.
- Variar ventanas tardías: 30-60, 35-60, 40-60, 45-60, 50-60.
- Variar `ORIENT_MIN_SCORE` y `ORIENT_MIN_RATIO`.
- Reportar heatmaps de AUC y estabilidad del target.

### 3. Null models

Crear `src/11_null_models.py`:

- Permutar etiquetas `axial_target`.
- Permutar identidad de jugadores dentro de díada.
- Permutar celdas preservando D4.
- Comparar contra particiones aleatorias con igual tamaño.
- Comparar contra plantillas diagonales, no solo LR/TB.

### 4. Transferencia a desempeño por díada

Crear `src/12_dyad_level_transfer.py`:

- Guardar `dyad_transfer_summary.csv`.
- Comparar AXIAL vs MIXED usando díada como unidad.
- Usar bootstrap por díada o modelos mixtos con intercepto aleatorio por díada.
- Separar present/absent.

### 5. Reproducibilidad

- Añadir `Makefile` o `scripts/run_all.sh` no interactivo.
- Añadir hashes de los CSV originales.
- Añadir `analysis_config.yaml` con umbrales y ventanas.
- Añadir `requirements.lock` o `environment.yml`.
- Añadir una tabla `data/results/manifest.csv` con versión, fecha, script y outputs.

## Estructura recomendada del preprint

1. Introduction
2. Dataset and task
3. Graph representation and degeneracy of the square board
4. Late axial specialization as symmetry breaking
5. Early graph signal
6. Predictive evaluation
7. Transfer to behavioral performance
8. Robustness and null models
9. Limitations
10. Symmetry-breaking experimental design for future work

## Ruta PhD

Este proyecto se vuelve doctoral si deja de ser solo “reanálisis bonito” y se convierte en una teoría experimental de coordinación bajo simetrías del entorno.

Pregunta PhD:

> How do graph symmetries and spectral degeneracies shape the emergence of specialized roles in human multi-agent search?

Experimento futuro:

- Tablero cuadrado `P_8 ⊠ P_8`: eigenspace degenerado, alta ambigüedad LR/TB.
- Tablero rectangular `P_6 ⊠ P_8`: degeneración rota, una orientación debería dominar.
- Tablero perturbado con obstáculos: automorfismos reducidos, predicción por conductancia local.
- Tablero con simetría rotacional alternativa: probar si emergen roles no axiales.

Predicción fuerte:

> Al aumentar la ruptura de simetría geométrica, debería disminuir la entropía de convenciones humanas y aumentar la alineación con la dirección espectral única de menor conductancia.

Ese es el salto de curso bueno a programa de investigación.
