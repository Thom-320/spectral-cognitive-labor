# Reconstrucción temporal y comparación de representaciones

Estado: primer entregable ejecutado, sin ajustar modelos predictivos ni definir un outcome nuevo.
Evaluación histórica de referencia: commit `3f3b70c29be44d4b9d9e485642e54cc1807c870a`.

## Alcance y atribución

El experimento y los datos son de Andrade-Lotero y Goldstone (2021),
[Self-organized division of cognitive labor](https://doi.org/10.1371/journal.pone.0254532).
Thomas Chisica realiza el reanálisis computacional; esta reconstrucción no implica
creación, recolección ni propiedad de los datos originales.

No se modificaron los datos, resultados, figuras, scripts ni documentación históricos.
La carpeta `audit/temporal_repair_v1` contiene salidas separadas y un archivo
`historical_evaluation.tar.gz` de esos materiales, con SHA-256 en `provenance.json`.
Las AUC históricas 0.804 y 0.860 corresponden a una evaluación anterior con
solapamiento temporal y no se validan de nuevo en este entregable.

## Archivos y reproducción

- `scripts/audit_temporal_repair.py`: reconstrucción offline; Python >=3.10 y NumPy.
- `tests/test_temporal_repair.py`: seis pruebas de integridad científica.
- `audit/temporal_repair_v1/early_cohort.csv`: una fila por díada, cinco rondas,
  instante de corte, tres métricas originales y representaciones tempranas.
- `audit/temporal_repair_v1/absent_round_metrics.csv`: 2.644 filas jugador-ronda,
  con referencia explícita a la ronda ausente anterior. No contiene etiquetas futuras.
- `audit/temporal_repair_v1/historical_vs_reconstructed_rounds.csv`: comparación
  por díada entre las oportunidades históricas y las reconstruidas.
- `audit/temporal_repair_v1/projectors.npz`: bases, proyectores y autovalores.
- `audit/temporal_repair_v1/summary.json`: comprobaciones y resultados descriptivos.
- `audit/temporal_repair_v1/provenance.json`: versiones, hash del script,
  fuentes locales preservadas y hashes de salidas.

Comandos desde la raíz, con un Python que disponga de NumPy:

```bash
python3 -B -m unittest discover -s tests -p 'test_temporal_repair.py' -v
python3 -B scripts/audit_temporal_repair.py --output /tmp/sodcl-reconstruction-new
```

El directorio de salida debe ser nuevo. El script rechaza sobreescrituras.
No ejecuta `make pipeline`, no necesita red, no importa sklearn ni ajusta modelos.
La ejecución verificada utilizó Python y NumPy del runtime de Codex; las versiones
exactas están en el manifiesto. Se completó en segundos y las seis pruebas pasaron.

## Cohorte y momento de predicción

Fuente: `data/raw/performances.csv`, 5.400 filas, 45 díadas y 60 rondas por díada.
Se reconstruyen TODAS las rondas `Unicorn_Absent`: 2.644 filas / 1.322 díada-rondas.
Cada ronda debe contener exactamente dos jugadores y visitas binarias; claves
duplicadas o parejas incompletas provocan un error, no una exclusión silenciosa.

Las primeras cinco rondas ausentes de cada díada definen el instante de predicción:
**al terminar la quinta oportunidad de búsqueda con objetivo ausente**.
Todas las díadas tienen cinco oportunidades y diez observaciones jugador-ronda.
Los cortes oscilan entre la ronda 6 y la 19. Se registra `cutoff_round` y
`elapsed_present_rounds = cutoff_round - 5`; no se decide todavía su inclusión
como control predictivo. Cinco oportunidades no equivalen a igual tiempo total.

Comprobación temporal: para las 45 díadas, reconstruir únicamente con el prefijo
de observaciones hasta el corte produce exactamente la misma fila temprana.
Una prueba sintética modifica visitas posteriores al corte sin cambiar los features.

El archivo histórico `humans_only_absent.csv` tiene 1.244 filas. Se comprobó que
sus claves son exactamente las rondas ausentes seguidas por otra ronda ausente.
La evaluación anterior toma cinco rondas de ese archivo: 15 díadas alcanzan
`Round >= 30`, dentro de las ventanas candidatas del outcome histórico.
La selección futura depende de la presencia experimental del objetivo; esto no
demuestra selección por comportamiento ni permite cuantificar el sesgo de las AUC.

## Definiciones reconstruidas

Sean A y B los conjuntos de casillas visitadas por los dos jugadores en una ronda.

- `Joint = |A intersección B|`. Se coteja con la columna cruda y se exige igualdad.
- `DLIndex = (|A unión B| - |A intersección B|)/64 = |A diferencia simétrica B|/64`.
  Se promedian cinco valores de díada-ronda (equivalente al promedio de diez filas
  porque cada ronda completa aporta dos copias iguales).
- `Similarity`: máximo Jaccard entre las visitas del jugador y las ocho regiones
  ALL, NOTHING, BOTTOM, TOP, LEFT, RIGHT, IN y OUT. IN es el interior 6x6;
  OUT es su complemento. Se promedian las diez observaciones tempranas.
- `Consistency`: Jaccard con las visitas del MISMO jugador en la ronda AUSENTE
  anterior, aunque haya rondas presentes intermedias. Esta convención reproduce
  el código original antes de su filtro final, no una comparación entre rondas
  consecutivas de cualquier tipo. En la primera ronda ausente es faltante;
  se promedian ocho valores disponibles por díada, sin imputación.
- Jaccard de dos conjuntos vacíos vale 1; de vacío y no vacío vale 0.

No se utilizan `ScoreLEAD`, `RegionGo`, etiquetas tardías ni otras columnas futuras.
Las métricas de todas las rondas son trazabilidad; los features tempranos solo
agregan las primeras cinco oportunidades de cada díada.

Cotejo contra las filas compartidas con el archivo histórico:

| Métrica | Valores coincidentes | Faltantes coincidentes | Error absoluto máximo |
|---|---:|---:|---:|
| DLIndex | 1.244 | 0 | 0 |
| Similarity | 1.244 | 0 | 0 |
| Consistency | 1.194 | 50 | 0 |
| Joint | 1.244 | 0 | 0 |
| Size_visited | 1.244 | 0 | 0 |

Esto verifica equivalencia de implementación en las filas compartidas, no una
revalidación independiente de la teoría conductual original.

## Definición matemática y verificaciones

Espacio común: R^64 en orden por filas, producto interno euclídeo sin ponderación.
Grafo fijado para este entregable: cuadrícula no ponderada con ocho vecinos,
Laplaciano combinatorio L = D - A. Es una hipótesis de proximidad espacial;
no se interpreta como restricción de acciones del juego.

Q_F se obtiene del espacio propio de lambda_2 usando
`isclose(lambda, lambda_2, atol=1e-10, rtol=1e-8)` y QR.
Se exige dimensión exactamente dos, sin incluir el modo constante.
Q_xy se obtiene por QR de las coordenadas x,y centradas, sin ponderación.
En ambos casos P = Q Q^T y rango(P) = 2.

Resultados numéricos: lambda_2 = lambda_3 = 0.4164003105;
el siguiente autovalor es 0.7399233311. Los dos ángulos principales son
6.590464612 grados. Distancia de Frobenius entre proyectores: 0.229543656;
distancia de operador: 0.114771828. Los espacios NO son idénticos.

Se verificaron ortonormalidad, simetría, idempotencia, anulación de constantes,
residuo espectral e invariancia del proyector al rotar su base. También se
verificó equivariancia ante las ocho simetrías D4 del tablero. El residuo máximo
para Fiedler es aproximadamente 1.02e-14. Las energías son invariantes al cambiar
jugadores (m -> -m), escalar m por una constante no nula y cambiar la base del
mismo subespacio.

Para el margen temprano m = suma de visitas jugador 1 - suma de visitas jugador 2:

E_F = ||Q_F^T m||² / ||m||²; E_xy = ||Q_xy^T m||² / ||m||².

Si m = 0 se guardan ambas energías como 0 por CONVENCIÓN, con `margin_zero=True`.
El cociente matemático no está definido en cero. No hay márgenes cero en estas
45 díadas, pero la convención está probada. No se inventa una orientación axial.

`dominant_score = max(|Q_xy^T m|)/||m||` se conserva como referencia secundaria
geométrica. Su valor usa los ejes físicos x,y: no es invariante ante rotaciones
arbitrarias de esa base, aunque sí ante intercambio/reflexión de los ejes.

Energía y axialidad no son lo mismo: x normalizado y (x+y)/sqrt(2), usando ejes
ortonormales, tienen E_xy = 1; sus dominant_score son 1 y 0.7071 respectivamente.
En este ejemplo E_F también coincide entre ambos márgenes (aprox. 0.98683).

## Comparación descriptiva ejecutada

En las 45 díadas reconstruidas:

- Correlación Pearson entre E_F y E_xy: **0.9994862483**.
- Diferencia absoluta media: **0.0109498941**.
- Diferencia absoluta máxima: **0.0624926756**.

Las energías son muy redundantes en estos datos, pero no idénticas. Correlación
alta no implica igualdad, equivalencia predictiva ni ausencia de una contribución
condicional; no se calcularon AUC, p-valores ni intervalos predictivos nuevos.
La alta semejanza reduce la expectativa de una gran diferencia entre estas dos
representaciones, pero no descarta todas las posibles medidas espectrales.

## Decisiones listas para Andrade

1. ¿El outcome será axialidad tardía o coordinación eficaz general? No son sinónimos.
2. ¿Qué evidencia distingue no axialidad legítima de insuficiencia de observación?
   No transformar automáticamente `INVALID` en cero ni excluirlo sin justificación.
3. ¿Qué ventana posterior y mínimo de observaciones permiten clasificar? Las
   ventanas anidadas históricas no son tres réplicas independientes de estabilidad.
4. ¿Se incluirá `cutoff_round` como control común, fijándolo antes de nuevos ajustes?
5. ¿Se comparará sustitución (conducta + E_xy versus conducta + E_F) o información
   condicional adicional? La segunda pregunta exige otro contraste y más complejidad.

Todavía debe fijarse el procedimiento de incertidumbre: remuestrear predicciones
ya calculadas condiciona en esos ajustes; reajustar exige mantener juntas las
copias de cada díada para no contaminar entrenamiento y evaluación. No se ha
elegido un método ni un criterio de utilidad después de observar nuevas AUC.

Thomas puede explicar y ejecutar esta reconstrucción. Andrade valida el significado
conductual y el alcance científico. No se encontró evidencia sobre identidad,
competencia o disponibilidad de Sebastián; no se le asigna revisión ni autoría.
Este documento no constituye un acuerdo del equipo ni una preregistración confirmatoria.

## Procedencia de las definiciones

Fuentes originales consultadas en el commit upstream
`b5cf4d4d5334f3b7d048d7c7a1ba9d37722dc89c`:

- [Measures.py](https://github.com/EAndrade-Lotero/SODCL/blob/b5cf4d4d5334f3b7d048d7c7a1ba9d37722dc89c/Python/Measures.py):
  regiones líneas 15–24; secuencia de filtrado y métricas líneas 52–72 y 167.
  SHA-256: `8693c49a36d664d3de18f338c963dbf88ef83c4c8d38e55e4473d2450cb10ea8`.
- [FRA.py](https://github.com/EAndrade-Lotero/SODCL/blob/b5cf4d4d5334f3b7d048d7c7a1ba9d37722dc89c/Python/FRA.py):
  Jaccard líneas 32–49; máxima similitud líneas 199–207.
  SHA-256: `4e9b9abe329ebe1dce60a7c912548ea0a7f27776e7ed2d8975517ebde0e00b05`.

Datos de ese commit cotejados con hashes locales:
`performances.csv`: `fecb6977312fce3a8d64c63d8fdcfecc5e292caf72f17d6276b6174ceb72291f`;
`humans_only_absent.csv`: `49aab9a6dc9b68c94447f8ae82a280d0a25ed7ed61e4f94cd081de19146b757e`.

No se propone una nueva arquitectura, transferencia a otra tarea ni explicación
mecanística a partir de estas comprobaciones.

## Packaging update, 2026-09-10

The original preservation statement above describes the first audit execution.
README, STATUS and CITATION metadata were subsequently corrected for temporal
validity and attribution; historical bytes remain in the archive and Git. Do not
compare the archived README/STATUS hashes to those intentionally updated files.
The script snapshots its current input checkout when rerun; a new output directory
will therefore have a different provenance snapshot after these documentation changes.

Minimal installable environment (Python >=3.11, 3.12 recommended for this pinned NumPy; no Codex-specific paths):

```bash
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install -r requirements-audit.txt
.venv-audit/bin/python -B -m unittest discover -s tests -p 'test_temporal_repair.py' -v
```

Python 3.12.14 / NumPy 2.3.5 ran the six checks again on 2026-09-10. The install
commands are provided for portability; a fresh environment installation was not
performed in this review.
