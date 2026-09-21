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

> **Estado.** Todos los resultados vienen de corridas de 60 réplicas cuyas salidas
> registradas están en `parametric_power_output.txt` y `fine_power_output.txt`, con
> los valores en los `.json` del mismo nombre.

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

## Resultado 3: recuperar el modelo no es recuperar los parámetros

Sobre las 60 réplicas generadas por FRA, comparando el parámetro que generó los
datos con el que se estima:

| Parámetro | verdad | media estimada | de |
|---|--:|--:|--:|
| bias ALL (log) | −3,46 | −3,63 | 0,22 |
| bias NOTHING (log) | −3,13 | −3,36 | 0,20 |
| bias mitades (log) | −7,01 | −64,40 | 374,7 |
| bias IN/OUT (log) | −34,72 | −25,50 | 41,6 |
| log alpha | 3,31 | 1,87 | 0,26 |
| gamma | −63,89 | −104,86 | 126,6 |
| **log delta** | **−0,99** | **−1,28** | **0,23** |
| **z** | **0,956** | **0,952** | **0,017** |

**Los dos parámetros que definen el mecanismo propio de FRA, `delta` y `z`, se
recuperan bien.** `z` con desviación 0,017 alrededor del valor verdadero. Los dos
sesgos que en el generador valen prácticamente cero están en la frontera del
espacio y no son identificables, lo cual es esperable y no tiene consecuencias.

El caso interesante es `gamma`, y explica también el sesgo de `alpha`. En estos
datos `Score = 32 − Joint` cuando el jugador acierta y `−64 − Joint` cuando no, así
que los dos rangos de score son **disjuntos**: [−32, 32] frente a [−128, −64]. Se
comprueba aquí que cualquier `gamma` entre −64 y −32 produce **exactamente el mismo
modelo**, y que ese modelo es el indicador de haber acertado.

> El umbral de win-stay, tal como lo ajusta la verosimilitud sobre estos datos, no
> dice «el score fue suficientemente alto». Dice «acerté». La penalización por
> solapamiento, que es lo que hace continuo al score, no interviene en dispararlo.

Por tanto `gamma` solo está identificada dentro de un intervalo de ancho 32, su
media y su desviación entre réplicas no son interpretables, y `alpha` se desplaza
para compensar cuando `gamma` cae fuera de ese intervalo. **`alpha` y `gamma` no
están identificados por separado.**

Un perfil de verosimilitud sobre `gamma`, reajustando el resto, lo muestra: los
valores −50, −40 y −33 dan los tres exactamente 592,6, mientras que el `gamma = 15`
publicado queda unos 22 nats peor. Bajo esta reimplementación la verosimilitud
prefiere con claridad la regla «quédate si acertaste» a un umbral positivo. Como no
se reproduce exactamente la tubería original, esto se registra como una **diferencia
que hay que consultar con Andrade**, no como una corrección de su resultado.

Esto extiende con un diagnóstico concreto lo que el propio artículo admite en su
Fig 8, que el ajuste de parámetros es subóptimo en el caso de FRA: el problema no
está en `delta` ni en `z`, sino en la pareja `alpha`-`gamma`.

## Resultado 4: cuántas díadas harían falta

Generador FRA con el `delta` ajustado, 30 réplicas por fila.

| Díadas | Transiciones | P(se elige FRA) | P(el IC excluye 0) |
|--:|--:|--:|--:|
| 45 | 1.285 | 1,00 | 1,00 |
| 90 | 2.542 | 1,00 | 1,00 |
| 180 | 5.102 | 1,00 | 1,00 |

Con este tamaño de efecto el experimento original ya basta; no hace falta recoger
más díadas. Es la diferencia con la versión en tablas, que necesitaba unas 90.

## Resultado 5: efecto mínimo detectable

La curva principal salta de 0,02 a 1,00 entre sus dos primeras filas, así que
[`fine_power.py`](fine_power.py) rellena ese tramo con 60 réplicas por punto.

| delta | % del ajustado | Δ log-loss observado | P(se elige FRA) | P(el IC excluye 0) |
|--:|--:|--:|--:|--:|
| 0,0186 | 5 % | 0,0018 | 0,65 | 0,07 |
| 0,0372 | 10 % | 0,0080 | 0,88 | 0,37 |
| 0,0557 | 15 % | 0,0159 | 0,95 | **0,77** |
| 0,0743 | 20 % | 0,0221 | 0,98 | **0,88** |
| 0,1115 | 30 % | 0,0331 | 1,00 | 1,00 |

**El 80 % de potencia se alcanza en torno a `delta ≈ 0,06`, un 16 % del valor
ajustado, que corresponde a una diferencia de log-loss fuera de díada de unos 0,017
nats.** Es decir, el experimento original puede detectar un mecanismo FRA unas seis
veces más débil que el que sus propios datos sugieren.

Aquí la curva se indexa por la diferencia **observada** de log-loss y no por el
efecto oráculo. La razón es que a estos tamaños el estimador del oráculo, que ajusta
WSLS a una muestra grande generada por FRA, tiene ruido comparable al efecto que
mide y sale no monótono: 0,0077 en la primera fila y 0,0077 otra vez en la tercera.
La columna observada sí es monótona.

## Qué se sigue de todo esto

1. **El contraste se puede correr.** Con los modelos del artículo y 45 díadas, si
   FRA es el proceso verdadero se recupera siempre, y el diseño detecta mecanismos
   hasta seis veces más débiles que el ajustado. No hace falta recoger más datos.
   Esto revierte la conclusión de la [versión en tablas](README.md), que no era
   informativa; el problema estaba en la parametrización, no en el experimento.
2. **El criterio de cierre tiene que ser el intervalo.** Elegir el modelo de menor
   log-loss fuera de díada se equivoca el 27 % de las veces cuando FRA es falso.
   Exigir que un intervalo agrupado por díada excluya el cero lo baja al 2 %.
3. **Lo que no se puede concluir es sobre parámetros.** `delta` y `z` se recuperan;
   `alpha` y `gamma` no están identificados por separado. Cualquier afirmación del
   tipo «el umbral de win-stay vale tanto» necesita antes resolver eso.
4. **Hay dos cosas que consultar con Andrade antes de nada.** Que la verosimilitud
   prefiera aquí un `gamma` que equivale a «acerté» frente al positivo publicado, y
   que los modelos ajustados no reproduzcan la frecuencia marginal de RS al
   simularlos hacia adelante. Ninguna de las dos es una refutación; las dos son
   preguntas.

## Reproducción

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
