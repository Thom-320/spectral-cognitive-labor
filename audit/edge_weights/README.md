# ¿Cambia el óptimo si las aristas pesan distinto?

Responde a la pregunta que Esteban planteó sobre el grafo espacial: cuál es el racional de los pesos. Solo geometría, sin datos de participantes y sin ajustar modelos.

## Punto de partida

En todo el pipeline el grafo es no ponderado: cada arista vale uno, tanto la ortogonal como la diagonal. Esa decisión está en una sola línea, repetida en cada script, por ejemplo `A[node, neighbor] = 1` en [`src/00_spectral_grid.py`](../../src/00_spectral_grid.py). De ahí salen el Laplaciano, el vector de Fiedler y el [certificado del mínimo global](../optimum_certificate/README.md).

El candidato natural a peso no uniforme es descontar la diagonal, que en una retícula es más larga que la ortogonal.

## Resultado

Conductancia de las particiones candidatas cuando la arista diagonal pesa `w`:

| Partición | w = 1 | w = 0,7071 | w = 0,5 | w = 0,3 |
|---|--:|--:|--:|--:|
| Izquierda/derecha y arriba/abajo | 0,1048 | 0,0987 | 0,0932 | 0,0863 |
| Diagonal | 0,1475 | 0,1467 | 0,1459 | 0,1449 |
| Dentro 4x4 / fuera | 0,3438 | 0,3277 | 0,3125 | 0,2933 |
| Dentro 6x6 / fuera | 0,5152 | 0,4720 | 0,4340 | 0,3891 |

El óptimo global, resuelto por MILP con el método de Dinkelbach, sigue siendo axial en todo el rango probado. **Un descuento uniforme a las diagonales no cambia la partición óptima.** El corte diagonal nunca gana, y las particiones de dentro y fuera quedan siempre muy por detrás.

## Consecuencia

Si los pesos han de cambiar algo, tienen que ser heterogéneos y venir de la conducta, por ejemplo de frecuencias de visita, de transiciones dentro de una ronda o de coselección entre casillas. Y entonces aparece un problema de identificabilidad: con pesos libres casi cualquier partición puede volverse óptima, basta bajar el peso de las aristas que cruzan la frontera deseada. Encontrar los pesos que hacen óptima la división observada no sería evidencia, salvo que la familia esté restringida y se estime con información independiente del resultado.

Por eso la [propuesta mecanística](../../docs/MECHANISM_PROPOSAL.md) no usa pesos sobre el grafo de casillas. Sus parámetros son probabilidades de transición entre estrategias, que sí tienen lectura conductual directa.

## Reproducción

```bash
python audit/edge_weights/weight_sensitivity.py
```

Requiere NumPy y SciPy. Tarda unos 26 segundos. La salida registrada está en `weight_sensitivity_output.txt`. En la primera iteración del método con `w = 1` el solver devuelve el conjunto vacío como certificado de que ningún conjunto mejora 22/210; el valor óptimo es el mismo que el del certificado exacto.
