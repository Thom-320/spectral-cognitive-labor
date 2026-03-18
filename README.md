# Análisis Espectral de División del Trabajo Cognitivo (SODCL)

## Descripción
Análisis de la eficiencia espectral de las divisiones del trabajo en el experimento "Seeking the Unicorn" de Andrade-Lotero & Goldstone (PLOS ONE 2021).

## Pregunta de Investigación
¿Son los humanos optimizadores espectrales intuitivos? Comparamos las particiones humanas con la bisección de Fiedler usando conductancia como métrica.

## Resultados Principales
- **15/15 díadas con splits focales (LR/TB) superan a Fiedler** (η = 1.27)
- **12/14 díadas mixed son peores que Fiedler** (η = 0.44)
- **p = 0.000017** (diferencia estadísticamente significativa)

## Estructura del Repositorio

```
grafos_proyecto/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
├── run.sh                             # Script de ejecución interactivo
│
├── src/                               # Código fuente
│   ├── 00_spectral_grid.py            # Análisis espectral del grafo 8×8
│   ├── 01_single_dyad_analysis.py     # Análisis de díada individual (435-261)
│   ├── 02_full_comparison.py          # Comparación completa de todas las díadas
│   ├── 03_power_iteration.py          # Power iteration básico
│   └── 03_power_iteration_fiedler.py  # Power iteration con vector de Fiedler
│
├── data/
│   ├── raw/                           # Datos de entrada (no modificar)
│   │   ├── humans_only_absent.csv     # Dataset principal (SODCL repo)
│   │   └── parameter_fit_humans.csv   # Parámetros ajustados por díada
│   └── results/                       # Resultados generados por los scripts
│       ├── spectral_results.npz       # Eigenvalores, eigenvectores, S_Fiedler
│       ├── spectral_comparison_results.csv  # Resultados por díada
│       └── power_iteration_results.npz
│
├── figures/                           # Figuras generadas por los scripts
│   ├── fiedler_grid.png               # Visualización del vector de Fiedler
│   ├── spectrum.png                   # Espectro del Laplaciano
│   ├── dyad_435_261_analysis.png      # Análisis de la díada ejemplo
│   ├── spectral_comparison_summary.png
│   ├── power_iteration_analysis.png
│   ├── power_iteration_fiedler.png
│   └── fiedler_analysis.png
│
├── paper/                             # Manuscrito LaTeX
│   ├── entrega1_borrador.tex          # Fuente LaTeX
│   ├── entrega1_borrador.pdf          # PDF compilado
│   ├── IEEEtran.cls                   # Clase IEEE
│   ├── algorithm.sty                  # Paquete de algoritmos
│   └── algorithmic.sty
│
└── docs/                              # Documentación del proceso
    └── CORRECCIONES.md                # Historial de correcciones
```

## Instalación

```bash
# Opción 1: instalar en entorno virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Opción 2: instalar directamente
pip install numpy scipy pandas matplotlib
```

## Ejecución

### Opción rápida: script interactivo
```bash
./run.sh
```

### Paso a paso

**Paso 1:** Análisis espectral del grafo base
```bash
python src/00_spectral_grid.py
```
Genera: `data/results/spectral_results.npz`, `figures/fiedler_grid.png`, `figures/spectrum.png`

**Paso 2:** Análisis de una díada individual (ejemplo 435-261)
```bash
python src/01_single_dyad_analysis.py
```
Requiere: `data/results/spectral_results.npz` (del paso 1)
Genera: `figures/dyad_435_261_analysis.png`

**Paso 3:** Comparación completa de todas las díadas
```bash
python src/02_full_comparison.py
```
Genera: `data/results/spectral_comparison_results.csv`, `figures/spectral_comparison_summary.png`

## Métricas Clave

| Símbolo | Significado |
|---------|-------------|
| λ₂ | Conectividad algebraica (segundo eigenvalor del Laplaciano) |
| h(S) | Conductancia de la partición S |
| η | Ratio de eficiencia: h(Fiedler) / h(S_obs) |

## Referencias
- Andrade-Lotero, E., & Goldstone, R. L. (2021). Self-Organized Division of Cognitive Labor. PLOS ONE.
- Chung, F. R. (1997). Spectral Graph Theory. AMS.
- Spielman, D. A. (2007). Spectral Graph Theory and its Applications.

## Autor
Thomas Chísica - Universidad del Rosario
Teoría de Grafos 2026-I
