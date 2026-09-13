# Certificado del mínimo global de conductancia en P₈ ⊠ P₈

## Resultado

En la grilla 8×8 no ponderada de ocho vecinos (64 vértices, 210 aristas, vol(V) = 420):

- La conductancia mínima sobre todas las particiones es exactamente **h\* = 22/210 ≈ 0,104762**.
- Solo cuatro conjuntos la alcanzan: las mitades de arriba, abajo, izquierda y derecha. Es decir, las particiones LR y TB.
- El siguiente valor posible es 23/205 ≈ 0,1122.

Esto reemplaza la cautela histórica del repositorio, que presentaba 22/210 como una referencia favorable "dentro de la familia evaluada".

## Definición

h(S) = cut(S, Sᶜ) / min(vol(S), vol(Sᶜ)), con vol(S) = suma de grados de S. Es la misma definición que usa el pipeline histórico (`src/02_full_comparison.py`) con el Laplaciano combinatorio.

## Dos formulaciones independientes

**1. Programación entera (`optimum_milp.py`).** Toda partición tiene un lado con 1 ≤ vol ≤ 210, y ese lado es el que fija el mínimo en el denominador. Por tanto:

```
min h(S) ≥ 22/210   ⇔   min { 210·cut(S) − 22·vol(S) : 1 ≤ vol(S) ≤ 210 } ≥ 0
```

- No fija el tamaño de S: no es una bisección.
- Todos los coeficientes son enteros, así que un óptimo de 0 demostrado con gap 0 es exacto.
- Después enumera con cortes no-good todos los S con objetivo ≤ 0. Encuentra exactamente cuatro y luego un óptimo de 320 con gap 0, que corresponde a vol = 205 y cut = 23.
- Salida: `milp_certificate.json`.

**2. Programación dinámica exhaustiva (`optimum_dp_exhaustive.py`).** Recorre las columnas con estado igual al patrón de la columna (2⁸) por el volumen acumulado (0 a 420). Así cubre los 2⁶⁴ subconjuntos y obtiene el corte mínimo para cada volumen exacto.

- Da el mismo h\*, alcanzado solo con vol = 210.
- Da el mismo segundo mejor valor.
- También calcula la anchura de bisección (22 aristas) y el caso de la grilla de cuatro vecinos, donde h\* = 8/112.
- Salidas: `dp_output.txt` y `dp_results.npz`.

Las dos usan métodos distintos: un MILP resuelto por ramificación y acotación con HiGHS, y una enumeración exacta sin solver.

**Validación de la programación dinámica (`validate_dp_small_grids.py`).** Reescribe la misma recursión parametrizada por filas y columnas y la compara con fuerza bruta. Lo hace en siete grillas entre 3×3 y 5×4, con ocho y con cuatro vecinos. En las 14 combinaciones coincide el corte mínimo para cada volumen, no solo el óptimo. Después comprueba que la versión parametrizada reproduce `dp_results.npz` en el 8×8. Salida: `validation_output.txt`.

## Reproducción

Desde esta carpeta, con NumPy y SciPy ≥ 1.9 (HiGHS):

```bash
python optimum_dp_exhaustive.py > dp_output.txt
python optimum_milp.py > milp_certificate.json
python validate_dp_small_grids.py > validation_output.txt
```

Ejecutado el 2026-09-13 con Python 3.14.4, NumPy 2.5.2 y SciPy 1.18.0. La programación dinámica tarda menos de un segundo, la validación unos dos segundos y el MILP unos 46 segundos. No usa datos de participantes ni ajusta modelos.

## Alcance

Es un resultado sobre el grafo elegido, no sobre las personas.

- El grafo de ocho vecinos es una hipótesis de proximidad espacial. En la tarea los clics son libres, así que no es el espacio de acciones.
- Que LR y TB sean los únicos minimizadores explica por qué son cortes baratos en este grafo. No explica la saliencia de IN/OUT ni de ALL/NOTHING, que el estudio original también cuenta como divisiones exitosas.
- En la grilla de cuatro vecinos el resultado análogo es un teorema conocido (Bollobás y Leader, 1991).

Experimento y datos: Andrade-Lotero y Goldstone (2021), PLOS ONE 16(7): e0254532.
