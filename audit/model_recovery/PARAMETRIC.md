# Los modelos del artículo, reimplementados y puestos a prueba

La [simulación con tablas](README.md) mostró que el contraste de la propuesta, escrito
como tablas de frecuencias, no tiene potencia: 0,73 de recuperación y 0,11 de
potencia. La conclusión era reescribir los modelos en forma paramétrica y repetir
todo. Esto es eso.

Aquí no hay modelos inventados: son MBIASES, WSLS y FRA tal como los define
Andrade-Lotero y Goldstone (2021), reimplementados desde las ecuaciones del artículo
y contrastados contra el código fuente de los autores
([EAndrade-Lotero/SODCL](https://github.com/EAndrade-Lotero/SODCL), `R/MODELpred.R`
y `R/fitModel.R`). La reimplementación es de Thomas Chisica; el modelo y el
experimento son de los autores originales.

**No se reporta ninguna comparación de modelos sobre las personas** más allá de
reproducir las cifras que los autores ya publicaron. El contraste fuera de díada
sobre humanos sigue sin correrse.

> **Estado.** Los resultados 1 y 2 vienen de una corrida de 60 réplicas cuya salida
> registrada se añade en el commit siguiente, junto con el resultado 3, que estaba
> ejecutándose cuando se escribió esto.

## Los modelos

Un modelo da la probabilidad de que el jugador explore cada una de las nueve
categorías de región en la ronda siguiente, dado el estado `(i, s, j)`: la región
que acaba de destapar, el score obtenido y las casillas que solapó con el compañero.

```
attract(k, i, j, s) = bias_k
                    + alpha * thresh(s, beta, gamma) * I(k, i)
                    + delta * thresh(FRAsim(i, j, k), epsilon, z)

P(k) = attract(k) / sum_r attract(r)

FRAsim(i, j, k) = sim(i, k) * Focal(k) + sim(j, ~k) * Focal'(k)
```

`sim` es el índice de Jaccard, con la convención `sim(vacío, vacío) = 1`. `Focal(k)`
vale cero solo para RS; `Focal'(k)`, para RS y para ALL. MBIASES fija `alpha = delta = 0`
y WSLS fija `delta = 0`, así que los tres están anidados. Encima va la mano
temblorosa: `P_ns(k) = 0,88 · P(k)` para las focales y `P_ns(RS) = 1 − 0,88 · (1 − P(RS))`.

Los sesgos respetan la simetría del artículo: uno para ALL, uno para NOTHING, uno
compartido por las cuatro mitades y uno compartido por IN y OUT, con el de RS
determinado por que sumen uno. Son **4, 6 y 8 parámetros libres**.

## Qué hubo que averiguar leyendo el código

El artículo no basta para reproducir sus cifras. Cinco cosas salieron del código:

1. **El conjunto de ajuste son 1.244 transiciones, no 5.400 filas.** `fitModel.R`
   carga `humans_only_absent.csv`. Se comprueba aquí que esas 1.244 filas son
   exactamente las rondas ausentes cuya ronda siguiente también es ausente. Esto
   importa para el proyecto: **el ajuste publicado no sufre el truncamiento** de las
   rondas con objetivo presente, ni en el predictor ni en el objetivo.
2. **Su «Dev.» es `−logL`, no `−2logL`.** El objetivo es `-sum(log(dmultinom(...)))`,
   y `dmultinom` devuelve una probabilidad. Se comprueba con la propia Tabla 3:
   `AIC = Dev + 2k` da 1620, 668 y 577 exactamente.
3. **Por tanto su AIC usa `−logL + 2k`, no `2(−logL) + 2k`.** Eso pesa la penalización
   por complejidad el doble de lo que corresponde. Con la fórmula estándar, la
   ventaja de FRA sobre WSLS sería de 188 puntos en lugar de 91. **La comparación
   publicada es conservadora en contra de FRA, no a su favor.**
4. **`beta` y `epsilon` sí se optimizan**, dentro de `[29, 30]`, aunque el texto dice
   que están fijos en 30. Por eso los conteos 4, 7 y 10 de la Tabla 3 son
   coherentes: incluyen esos dos. Los parámetros realmente libres son 4, 6 y 8.
5. **Las probabilidades se recortan a `[1e-4, 0,9999]` sin renormalizar**, y la
   agregación por situaciones `(región exacta, score exacto, solapamiento exacto)`
   añade el coeficiente multinomial, que aquí se calcula en 119,6 sobre las 787
   situaciones que forman las 1.244 transiciones.

## Reconciliación con las cifras publicadas

Ajustando aquí, sobre el mismo conjunto:

| Modelo | `−logL` aquí | menos 119,6 | Dev. publicada | libres | contados |
|---|--:|--:|--:|--:|--:|
| MBIASES | 1821,3 | 1701,7 | 1612 | 4 | 4 |
| WSLS | 671,4 | 551,8 | 654 | 6 | 7 |
| FRA | 589,9 | 470,3 | 557 | 8 | 10 |

Diferencias de log-verosimilitud aquí: **1149,9 y 81,5**; publicadas: **958 y 97**.
El orden y el orden de magnitud se reproducen y los parámetros recuperados se
parecen a los publicados, pero no es una reproducción exacta. Las causas probables
son el redondeo de la Tabla 3, las cotas del optimizador en el código original
(los sesgos están acotados en `[0, 0,125]` y `gamma` en `[0, 32]`) y el detalle de
la agregación. Reproducirlo al dígito exigiría ejecutar su tubería en R.

## Un problema del optimizador que había que resolver primero

Con `beta = epsilon = 30`, las dos funciones umbral son escalones y la verosimilitud
es casi plana en `gamma` y en `z`. Un optimizador local depende muchísimo del
arranque. Medido aquí: ajustando con un solo arranque, **WSLS quedaba hasta 46 nats
por debajo de FRA sobre datos generados por WSLS**, lo cual es imposible, porque FRA
contiene a WSLS. Es decir, una simulación ingenua habría medido el fallo del
optimizador y lo habría reportado como ventaja de FRA.

La solución que se usa aquí es ajustar la escalera imponiendo el anidamiento: cada
modelo arranca también desde la solución del modelo anterior con el mecanismo nuevo
apagado y desde una rejilla sobre el umbral correspondiente, y después se hace una
pasada descendente para que el modelo menor pueda alcanzar cualquier región mejor
que el mayor haya encontrado. Con eso, sobre datos generados por WSLS la diferencia
`WSLS − FRA` pasa a ser cero en la mayoría de las réplicas, que es lo correcto.

Esto conecta con algo que el propio artículo reconoce: su ejercicio de recuperación
de parámetros (Fig 8) encuentra que «el ajuste es subóptimo en el caso de FRA,
probablemente por la interacción entre mecanismos». Lo que se añade aquí es la causa
concreta y una forma de evitarla.

## La simulación

El generador es el juego completo, no una cadena sobre categorías. Cada ronda los
dos jugadores eligen región con la `P(k)` del modelo; la región se realiza en
casillas con mano temblorosa; de ahí salen el solapamiento y el score; y el estado
resultante alimenta la ronda siguiente. Después se conserva el mismo subconjunto que
usa el estudio, las transiciones ausente-ausente.

Tres constantes del juego se estiman de los datos y son idénticas para los tres
modelos, así que no son lo que los distingue: la frecuencia con que el objetivo está
ausente (0,490), la probabilidad de acertar según cuántas casillas se destapan, y la
forma de las regiones RS.

**La forma de RS no se puede inventar.** El artículo describe RS como una región al
azar con todas las casillas equiprobables, pero la categoría RS de los datos recoge
todo lo que no coincide exactamente con una focal, y esas regiones se parecen mucho
más a las focales que una región aleatoria: la similitud máxima con alguna focal
tiene percentil 90 de **0,94** en los datos frente a **0,54** si se sortean casillas
independientes. Como FRA se dispara por un umbral sobre esa similitud, sortear
casillas independientes apagaría el mecanismo por construcción, y en una primera
versión de esta simulación lo apagó. Por eso las regiones RS se remuestrean de las
observadas en rondas ausentes.

## Resultado 1: recuperación de modelo

60 réplicas, 45 díadas, unas 1.250 transiciones por réplica, validación cruzada por
díada en cinco pliegues. Cada fila es un generador; cada columna, el modelo elegido.

| Generador | fuera de díada | | | AIC dentro de muestra | | |
|---|--:|--:|--:|--:|--:|--:|
| | MBIASES | WSLS | FRA | MBIASES | WSLS | FRA |
| MBIASES | **0,80** | 0,15 | 0,05 | **1,00** | 0,00 | 0,00 |
| WSLS | 0,00 | **0,73** | 0,27 | 0,00 | **1,00** | 0,00 |
| FRA | 0,00 | 0,00 | **1,00** | 0,00 | 0,00 | **1,00** |

**La respuesta a la pregunta que motivó todo esto es que sí: con estos modelos y
este diseño, si FRA es el proceso verdadero, se recupera en el 100 % de las
réplicas.** La versión en tablas lo recuperaba el 73 %. La diferencia no está en los
datos sino en cuántos parámetros hay que estimar: 8 frente a 1.080.

El punto débil está en la otra dirección. Cuando el generador es WSLS, es decir
cuando FRA es falso, **la validación fuera de díada elige FRA el 27 % de las veces**,
y cuando el generador es MBIASES elige un modelo más rico el 20 %. El AIC dentro de
muestra no se equivoca nunca. Es lo contrario de lo que sugiere la intuición de que
validar fuera de muestra protege del sobreajuste: elegir por menor log-loss, sin
intervalo y sin penalización, es un criterio ruidoso. Por eso el contraste tiene que
exigir un intervalo que excluya el cero, no simplemente el menor valor.

## Resultado 2: potencia frente a la fuerza del mecanismo

`delta` es el parámetro que separa FRA de WSLS: con `delta = 0`, FRA **es** WSLS. El
efecto oráculo es la ganancia esperada de log-loss de FRA sobre su mejor
aproximación WSLS, estimada ajustando WSLS a una muestra grande generada por FRA.
El valor ajustado sobre humanos es `delta = 0,372`.

| delta | efecto oráculo | P(se elige FRA) | P(el IC excluye 0) | Δ medio |
|--:|--:|--:|--:|--:|
| 0,000 | −0,0020 | **0,27** | **0,02** | −0,0000 |
| 0,186 | 0,0565 | 1,00 | 1,00 | 0,0530 |
| 0,372 | 0,0845 | 1,00 | 1,00 | 0,0951 |
| 0,557 | 0,1187 | 1,00 | 1,00 | 0,1228 |
| 0,929 | 0,1611 | 1,00 | 1,00 | 0,1752 |
| 1,486 | 0,2042 | 1,00 | 1,00 | 0,2145 |

La primera fila es la tasa de error de tipo I, porque ahí FRA es falso: **elegir por
menor log-loss se equivoca el 27 % de las veces; exigir que el intervalo excluya el
cero, el 2 %**, cerca del 2,5 % nominal de una prueba unilateral. El criterio de
cierre tiene que ser el intervalo.

A partir de ahí la potencia es plena. Incluso con la mitad del `delta` ajustado, el
contraste detecta el efecto en las 60 réplicas. La curva salta de 0,02 a 1,00 entre
las dos primeras filas, así que el efecto mínimo detectable queda por debajo de
0,057 nats y hay que localizarlo con una rejilla más fina.

## Resultado 3: efecto mínimo detectable

*(pendiente: `fine_power.py`)*

## Reproducción

```bash
python audit/model_recovery/parametric_power.py --reps 60 --folds 5
python -m unittest discover -s tests -p 'test_parametric_models.py' -v
```

Requiere NumPy, SciPy y pandas. La corrida reparte las réplicas entre procesos y fija
los hilos de álgebra a uno; sin eso los trabajadores se estorban y tarda un orden de
magnitud más. Las salidas registradas quedan en `parametric_power_output.txt`,
`fine_power_output.txt` y los `.json` del mismo nombre. Las quince pruebas de
`tests/test_parametric_models.py` comprueban el anidamiento de los tres modelos, la
forma de la similitud, la capa de mano temblorosa, el recorte y el simulador; el
cargador comprueba además que las máscaras focales reproducen la columna `Category`
en las 5.400 filas.
