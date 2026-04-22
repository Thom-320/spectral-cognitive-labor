# Symmetry Breaking and Early Prediction in SODCL

## Resumen
Este repositorio reanaliza **Seeking the Unicorn / Self-Organized Division of Cognitive Labor (SODCL)** desde teoria de grafos, pero ya no con la tesis vieja de "humanos vencen a Fiedler".

La tesis de trabajo actual es mas precisa:

**la geometria cuadrada de `P_8 ⊠ P_8` induce un eigenspace de Fiedler degenerado; varias diadas humanas exitosas rompen esa simetria hacia orientaciones axiales estables, y una senal geometrica temprana ayuda a anticipar esa estabilizacion.**

## Que NO afirma este repo

- No afirma que los humanos sean optimizadores espectrales generales.
- No afirma que la conductancia derrote a las metricas oficiales del paper de 2021.
- No afirma explicar por si solo todas las estrategias focales de SODCL.

## Datos y procedencia

Fuente oficial:

- repo oficial: [EAndrade-Lotero/SODCL](https://github.com/EAndrade-Lotero/SODCL)
- protocolo experimental: [Seeking the Unicorn en protocols.io](https://www.protocols.io/view/seeking-the-unicorn-8epv5zbdnv1b/v1)

Datasets usados aqui:

- `data/raw/humans_only_absent.csv`
  - 1244 filas
  - 45 diadas
  - solo rondas con unicornio ausente
  - incluye metricas derivadas como `DLIndex`, `Similarity`, `Consistency`, `Joint`
- `data/raw/performances.csv`
  - 5400 filas
  - 45 diadas x 60 rondas
  - incluye rondas `present` y `absent`

## Hallazgos actuales

### Geometria del tablero

- El tablero `P_8 ⊠ P_8` tiene 64 nodos y 210 aristas.
- `lambda_2 = lambda_3 ≈ 0.4164`, asi que el eigenspace de Fiedler es degenerado.
- El baseline ingenuo de Fiedler tiene `h = 28/210 = 0.1333`.
- Los cortes axiales `LR/TB` tienen `h = 22/210 = 0.1048`.
- En este repo, ese `22/210` se usa como baseline axis-aligned de referencia dentro de la familia comparada; no se presenta como un optimo global de `h(G)`.

### Reanalisis espectral tardio

- 45 diadas auditadas.
- 29 diadas en el set primario de conductancia.
- 21 diadas axiales (`LR/TB`) y 8 mixed en ese set.
- Las axiales muestran mucha mejor conductancia, mayor MI/JSD y mayor estabilidad.
- Limite crucial: `eta` y `DLIndex` estan fuertemente correlacionados (`rho ≈ 0.894`), asi que la novedad no es "mi metrica reemplaza la original".

### Prediccion temprana

Con las primeras 5 rondas absent comunes de cada diada:

- senal geometrica temprana sola: `AUC LOOCV = 0.804`
- metricas tempranas oficiales (`DLIndex + Similarity + Consistency`): `AUC LOOCV = 0.736`
- modelo combinado: `AUC LOOCV = 0.860`

Interpretacion:

- la geometria temprana agrega informacion util,
- y la combinacion con metricas oficiales es mas fuerte que cualquiera por separado.

### Transferencia a desempeno

Usando la orientacion estable inferida desde las rondas absent y luego mirando `performances.csv`:

- en rondas `present`, las diadas axiales tienen `accuracy ≈ 0.936` y `score ≈ 24.43`
- las mixed tienen `accuracy ≈ 0.688` y `score ≈ -6.14`

Esto sugiere que la ruptura de simetria no es solo una curiosidad geometrica: acompana mejor coordinacion conductual.

### Cautela metodologica

Cuando se controla por `DLIndex`, `Similarity` y `Consistency`, agregar `h_obs` tardia no mejora de forma clara la prediccion de desempeno posterior y a veces la empeora. Por eso el framing correcto no es:

- "la conductancia derrota a las metricas existentes"

Sino:

- "una senal geometrica temprana, interpretable y symmetry-aware ayuda a anticipar la estabilizacion posterior de roles"

## Estructura

```text
grafos_proyecto/
├── src/
│   ├── 00_spectral_grid.py
│   ├── 01_single_dyad_analysis.py
│   ├── 02_full_comparison.py
│   ├── 03_power_iteration.py
│   ├── 03_power_iteration_fiedler.py
│   ├── 04_temporal_dynamics.py
│   ├── 05_counterexample_P6xP8.py
│   ├── 06_partition_robustness.py
│   ├── 07_entropy_analysis.py
│   └── 08_early_prediction.py
├── data/raw/
├── data/results/
├── figures/
├── docs/
└── paper/
```

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion

### Menu interactivo

```bash
./run.sh
```

### Pipeline principal

```bash
python src/00_spectral_grid.py
python src/01_single_dyad_analysis.py
python src/02_full_comparison.py
python src/04_temporal_dynamics.py
python src/05_counterexample_P6xP8.py
python src/06_partition_robustness.py
python src/07_entropy_analysis.py
python src/08_early_prediction.py
```

## Artefactos principales

### CSV

- `data/results/spectral_comparison_results.csv`
- `data/results/spectral_analysis_audit.csv`
- `data/results/partition_stability_summary.csv`
- `data/results/entropy_analysis_results.csv`
- `data/results/early_prediction_features.csv`
- `data/results/early_prediction_summary.csv`
- `data/results/performance_transfer_summary.csv`
- `data/results/present_performance_increment.csv`

### Figuras

- `figures/spectral_comparison_summary.png`
- `figures/partition_robustness_summary.png`
- `figures/entropy_analysis.png`
- `figures/andrade_summary.png`
- `figures/early_prediction_summary.png`

### Documentos

- `docs/RESULTADOS_PRELIMINARES.md`
- `docs/EXCLUSIONES.md`
- `docs/NOTA_ANDRADE.md`
- `paper/entrega2.tex`
- `paper/preprint_base.tex`

## Estado del manuscrito

- `paper/entrega2.tex` queda orientado a la **segunda entrega del curso**.
- `paper/preprint_base.tex` conserva una version mas cercana a manuscrito de investigacion.

La mejor lectura del proyecto hoy es:

- coursework fuerte para Teoria de Grafos,
- reanalisis serio y reproducible,
- semilla plausible de preprint corto,
- y base razonable para una extension futura sobre ruptura controlada de simetria.
