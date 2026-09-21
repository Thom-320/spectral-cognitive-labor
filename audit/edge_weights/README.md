# ¿Cambia el óptimo si las aristas pesan distinto?

Responde a la pregunta que Esteban planteó sobre el grafo espacial: cuál es el
racional de los pesos. Solo geometría, sin datos de participantes y sin ajustar
modelos.

## Punto de partida

En todo el pipeline el grafo es no ponderado: cada arista vale uno, tanto la
ortogonal como la diagonal. Esa decisión está en una sola línea, repetida en cada
script, por ejemplo `A[node, neighbor] = 1` en
[`src/00_spectral_grid.py`](../../src/00_spectral_grid.py). De ahí salen el
Laplaciano, el vector de Fiedler y el
[certificado del mínimo global](../optimum_certificate/README.md).

## Qué se probó

Tres familias de ponderación, todas derivadas de la geometría del tablero y
ninguna ajustada a la conducta observada. Para **cada** ponderación se resuelve el
mínimo global de conductancia con el método de Dinkelbach sobre un MILP entero, no
solo se comparan particiones candidatas.

| Familia | Parámetro | Motivación |
|---|---|---|
| A. Descuento de la diagonal | `w_diagonal = w`, ortogonales 1 | una diagonal es más larga que una ortogonal; `w = 1/√2` es el caso natural |
| B. Anisotropía | `w_vertical = b`, horizontal y diagonal 1 | el tablero podría recorrerse con más facilidad en un eje que en otro |
| C. Núcleo gaussiano | `w_ij = exp(-d²/2s²)` sobre **todas** las parejas | la proximidad no tiene por qué cortarse en los ocho vecinos |

Las particiones candidatas incluyen las mitades axiales, la diagonal, dentro/fuera
en dos tamaños, un cuadrante, los cuadrantes opuestos y una escalera de dos
escalones.

## Resultado

**El óptimo global es axial en las tres familias y en los dieciséis valores de
parámetro probados**, con una salvedad de certificación: quince de los dieciséis
casos terminan con `status = 0`, es decir con el óptimo demostrado y gap cero. El
decimosexto, el núcleo gaussiano más ancho (`s = 2,5`, del orden de dos mil
aristas), agotó el límite de tiempo y devolvió `status = 1`. Su incumbente también
es axial, pero eso es una solución, no un certificado. La tabla completa, con el
estado del solver en cada caso, está en
[`weight_sensitivity_output.txt`](weight_sensitivity_output.txt).

Familia A, conductancia de las candidatas:

| Partición | w = 1 | w = 0,7071 | w = 0,5 | w = 0,3 |
|---|--:|--:|--:|--:|
| Izquierda/derecha y arriba/abajo | 0,1048 | 0,0987 | 0,0932 | 0,0863 |
| Escalera de dos escalones | 0,1238 | 0,1176 | 0,1118 | 0,1047 |
| Diagonal | 0,1475 | 0,1467 | 0,1459 | 0,1449 |
| Cuadrantes opuestos | 0,1905 | 0,1819 | 0,1739 | 0,1641 |
| Dentro 4x4 / fuera | 0,3438 | 0,3277 | 0,3125 | 0,2933 |
| Dentro 6x6 / fuera | 0,5152 | 0,4720 | 0,4340 | 0,3891 |

La familia B es la única que hace algo interesante: rompe el empate entre las dos
mitades axiales. Con `b < 1`, es decir con enlaces verticales más débiles, gana
arriba/abajo; con `b = 1,5` gana izquierda/derecha. Pero nunca gana una partición
no axial.

## Consecuencia

La conclusión defendible es la estrecha:

> El óptimo axial es robusto al descuento uniforme de las diagonales, a la
> anisotropía entre ejes y a los núcleos de distancia probados. No hemos
> identificado ninguna ponderación geométrica independiente que cambie esa
> conclusión.

Lo que **no** se ha demostrado es que no exista tal ponderación. Quedan fuera los
pesos dependientes de la posición, los costes perceptuales y cualquier familia
derivada de la estructura de la tarea. Una versión anterior de este documento decía
que «si los pesos han de cambiar algo, tienen que ser heterogéneos y venir de la
conducta»; era demasiado fuerte y queda retirada.

Lo que sí se sostiene es la advertencia de identificabilidad: con pesos
suficientemente libres casi cualquier frontera puede volverse óptima, basta bajar
el peso de las aristas que la cruzan. Por eso una ponderación futura tiene que
definirse **independientemente del reparto que pretende explicar**, y estimarse con
información que no sea ese reparto.

### Sobre la anisotropía en particular

La familia B podría, en principio, racionalizar una preferencia de eje. Conviene
saber que en estos datos no hay tal preferencia que explicar. Contando pares
complementarios en rondas ausentes, izquierda/derecha aparece en 133 rondas de
díada y arriba/abajo en 89; pero con la **díada** como unidad el reparto es 7 a 7,
con 31 díadas sin ninguno de los dos, y una prueba binomial da p = 1,0. La
asimetría por rondas la producen unas pocas díadas con muchas rondas.

## Correcciones a la versión anterior de este artefacto

Una revisión externa señaló cuatro problemas en el commit `5eeb204`. Los cuatro
eran correctos y están arreglados:

1. **Un candidato duplicado.** La «escalera» estaba definida como
   `((x<4)&(y<4)) | ((x<4)&(y>=4))`, que es algebraicamente `x < 4`, es decir
   izquierda/derecha otra vez. Daba exactamente la misma conductancia en las siete
   columnas. Ahora es una escalera real de dos escalones.
2. **El óptimo global solo se resolvía en tres puntos** (`w = 1; 0,7071; 0,5`)
   mientras la tabla mostraba siete, de modo que la frase «en todo el rango
   probado» no estaba respaldada. Ahora el MILP corre en todos.
3. **El conjunto vacío aparecía como solución**, con un `RuntimeWarning` por
   dividir entre volumen cero. Venía de calcular el nuevo cociente antes de
   comprobar la terminación de Dinkelbach. Ahora se conserva el incumbente y el
   conjunto vacío se interpreta como lo que es, un certificado de que ningún
   conjunto mejora el valor actual.
4. **El alcance de la conclusión.** Una sola familia no cierra la cuestión de los
   pesos geométricos. Se añadieron dos familias más y se reescribió la conclusión.

## Reproducción

```bash
python audit/edge_weights/weight_sensitivity.py
```

Requiere NumPy y SciPy. La familia C es la cara: el MILP sobre el núcleo ancho
tiene del orden de dos mil aristas. Cada resolución lleva un límite de 300
segundos, y la columna `status` de la salida indica si el valor está certificado
(`0`) o si el solver se detuvo antes (`1`); una fila con `status` distinto de cero
no es un certificado.
