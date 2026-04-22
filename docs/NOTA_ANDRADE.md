# NOTA BREVE PARA ANDRADE

## Hallazgo actual

Reencuadre el proyecto para salir de la tesis vieja de "humanos vs Fiedler". La version actual es esta:

- el tablero cuadrado `P_8 ⊠ P_8` tiene un eigenspace de Fiedler degenerado,
- varias diadas humanas exitosas rompen esa simetria hacia orientaciones axiales estables,
- y una senal geometrica temprana extraida de las primeras 5 rondas absent comunes ayuda a anticipar esa estabilizacion posterior.

## Que ya muestra el repositorio

En el reanalisis espectral tardio:

- `lambda_2 = lambda_3`
- el baseline ingenuo da `28/210`
- los cortes axiales dan `22/210` como valor de referencia dentro de la familia comparada aqui
- las diadas axiales se separan con fuerza de las mixed en conductancia, MI, JSD y estabilidad

En la capa nueva de prediccion temprana:

- senal geometrica temprana sola: `AUC LOOCV = 0.804`
- trio oficial temprano (`DLIndex + Similarity + Consistency`): `AUC LOOCV = 0.736`
- modelo combinado: `AUC LOOCV = 0.860`

Ademas, al llevar la orientacion estable a `performances.csv`, las diadas axiales muestran mejor accuracy y score, sobre todo en rondas `present`.

## Limite actual

La conductancia tardia no debe venderse como reemplazo de las metricas originales. Cuando controlo por `DLIndex`, `Similarity` y `Consistency`, agregar `h_obs` no mejora de forma clara la prediccion de desempeno posterior.

Por eso la historia que estoy defendiendo no es:

- "mi metrica derrota a las viejas"

Sino:

- "una senal geometrica temprana, symmetry-aware e interpretable ayuda a anticipar la cristalizacion posterior de roles"

## Siguiente prediccion

La extension que me parece mas interesante es romper la simetria del tablero de forma controlada. Si esta lectura tiene contenido real, al pasar de un cuadrado a un rectangulo o a una version perturbada deberian cambiar:

- la degeneracion de `lambda_2`
- la ambiguedad entre orientaciones axiales
- y la distribucion de estrategias humanas

## Mensaje sugerido

Profesor Andrade,

Le comparto una version mas pulida del proyecto. Ya no lo estoy planteando como "humanos vs Fiedler", sino como un reanalisis sobre ruptura de simetria y especializacion de roles: en el tablero cuadrado, la degeneracion del eigenspace de Fiedler parece convivir con la emergencia de orientaciones axiales estables, y una senal geometrica temprana ayuda a anticipar esa estabilizacion. Tambien integre la capa de informacion/entropia y separo explicitamente el analisis sobre `humans_only_absent.csv` del analisis sobre `performances.csv`.

Me serviria mucho saber si esta version le parece una extension conceptualmente util del paper de 2021, sobre todo pensando en una continuacion donde se rompa la simetria del tablero de forma controlada. Si ve una debilidad principal en el planteamiento, me ayudaria mucho saber cual es.
