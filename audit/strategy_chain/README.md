# Cadena de Markov sobre las estrategias del estudio original

Descriptivo. No construye ningún outcome de especialización ni ajusta modelos predictivos sobre él, y no toca los resultados históricos. La comparación de órdenes de la última sección ajusta tablas de frecuencias por conteo; es un diagnóstico del proceso.

## Fuente y procedencia

`data/raw/humans_full.csv` del repositorio original
([enlace](https://raw.githubusercontent.com/EAndrade-Lotero/SODCL/master/Data/humans_full.csv),
sha256 `9452e653a5ffd08f8148f15269d73b754f782ed7f48bc1c3292dc78b42105c54`, 5.400 filas, 60 rondas).
Es el archivo que el proyecto no estaba usando: incluye rondas presentes y ausentes, con `Category` en cada fila.

El script comprueba y falla si algo no cuadra:

- el hash del archivo;
- que `Category` equivale, en las 5.400 filas, a la coincidencia exacta de las visitas de esa fila con una de las ocho regiones focales, y RS en cualquier otro caso;
- que las 1.244 filas compartidas con `humans_only_absent.csv` coinciden en `Category`, tras renombrar UP y DOWN a TOP y BOTTOM, y en `DLIndex`.

`Consistency` **no** coincide entre los dos archivos y no se compara: el filtrado la define contra la ronda ausente anterior del mismo jugador.

## Qué se construye

- Cadena por jugador sobre los nueve estados LEFT, RIGHT, TOP, BOTTOM, IN, OUT, ALL, NOTHING y RS, entre rondas consecutivas de calendario, y restringida a 1–20, a 41–60 y a rondas ausentes consecutivas.
- Cadena por díada sobre el par de estrategias, con y sin identidad de jugador.
- Duración de las rachas, observada e implicada por la cadena.
- Comparación de órdenes 0, 1 y 2 dejando una díada fuera.

La distribución estacionaria solo se reporta cuando la cadena estimada es irreducible.

## Hallazgos

**La condición de la ronda cambia el estado observado.** En rondas presentes se destapan 18,8 casillas de media y el 85,7 % de las filas son RS. En ausentes se destapan 32,3 y RS baja al 58,6 %. La ronda presente termina al encontrar el objetivo, así que la estrategia apenas se manifiesta. La lectura razonable es que la estrategia es una variable latente observada de forma intermitente, no que la conducta cambie de golpe.

**Por eso la persistencia depende de qué rondas se miren.**

| Cadena | LEFT | RIGHT | TOP | BOTTOM | ALL | RS |
|---|--:|--:|--:|--:|--:|--:|
| Rondas de calendario consecutivas | 0,53 | 0,53 | 0,50 | 0,53 | 0,35 | 0,84 |
| Solo ausentes consecutivas | 0,92 | 0,90 | 0,86 | 0,91 | 0,68 | 0,88 |

La persistencia de 0,9 que a veces se cita es propiedad del archivo filtrado, no de la conducta por ronda.

**Duración de las rachas.** El resumen guarda la duración observada y la implicada por la cadena, `markov_implied_mean_dwell`, que es `1/(1−Pii)`. No son lo mismo:

| Estado | Rachas | Observada | Solo rachas completas | Implicada |
|---|--:|--:|--:|--:|
| LEFT | 91 | 2,08 | 2,09 | 2,13 |
| BOTTOM | 71 | 2,04 | 1,92 | 2,14 |
| NOTHING | 88 | 5,30 | 3,08 | 6,04 |
| RS | 659 | 5,93 | 4,31 | 6,32 |

Las rachas que llegan a la ronda 60 están cortadas, así que la duración observada subestima la real. La diferencia con el valor implicado no mide por sí sola el ajuste del supuesto de Markov.

**La ocupación y la persistencia de los pares focales complementarios aumentan con el tiempo.**

| Ventana | Rondas de díada en par complementario | Permanencia a un paso | Entrada desde fuera |
|---|--:|--:|--:|
| 1–20 | 4,4 % | 0,29 | 0,038 |
| 41–60 | 19,6 % | 0,49 | 0,133 |

No se afirma que sean atractores. Otros estados tienen más autopersistencia: RS/RS 0,79 y NOTHING/NOTHING 0,88, frente a LEFT/RIGHT 0,50 y BOTTOM/TOP 0,51. Llamarlos atractores exigiría definir cuenca, recurrencia o metastabilidad.

**Con identidad de jugador, no hay intercambios directos de rol.** De 342 transiciones que parten de un par focal complementario: 162 permanecen con el mismo reparto, 180 salen del estado y **cero** intercambian lados entre rondas consecutivas.

| Familia | Permanece | Intercambio directo | Sale |
|---|--:|--:|--:|
| LEFT/RIGHT | 66 | 0 | 67 |
| BOTTOM/TOP | 47 | 0 | 46 |
| ALL/NOTHING | 45 | 0 | 64 |
| IN/OUT | 4 | 0 | 3 |

Esto no excluye intercambios que pasen por otro estado intermedio.

**La tipología real es axial y ALL/NOTHING.** Estados de díada más frecuentes: RS/RS 1.652, NOTHING/RS 229, ALL/RS 201, LEFT/RIGHT 136, ALL/NOTHING 112, BOTTOM/TOP 99. El par IN/OUT casi no aparece.

**Orden del proceso.** Log-loss por observación, dejando una díada fuera, con el mismo conjunto de posiciones para los tres órdenes. Se reporta una rejilla, porque el resultado depende del suavizado y del soporte de entrenamiento:

| Secuencias | Orden 0 | Orden 1 | Orden 2 | Ventaja del orden 2 |
|---|--:|--:|--:|---|
| Todas las rondas, 5.220 obs | 1,12 | 0,736 a 0,739 | 0,627 a 0,642 | de +0,097 a +0,110; todos los intervalos excluyen cero |
| Ausentes consecutivas, 556 obs | 1,56 a 1,59 | 0,541 a 0,581 | 0,471 a 0,572 | de −0,007 a +0,071; el signo cambia con el suavizado |

Sobre todas las rondas, la dependencia más allá del estado anterior es estable en las seis variantes. En triples de rondas ausentes consecutivas **no se ha establecido** una mejora material del orden 2 sobre el orden 1: con suavizado 0,1 la ventaja es +0,071 y su intervalo excluye cero, con 1,0 es −0,007. Por parsimonia se toma el orden 1 como baseline inicial en ese régimen, que es una decisión de modelado y no un resultado.

Que haga falta más memoria al mezclar todas las rondas es compatible con el truncamiento de la observación, pero también con no estacionariedad, con heterogeneidad entre díadas o con dependencia real de orden superior. La cadena observada no distingue entre esas explicaciones.

## Límites

- La comparación de órdenes ajusta tablas por conteo; sus cifras dependen del suavizado, que por eso se reporta como rejilla.
- Las transiciones agregan 45 díadas y el proceso no es homogéneo en el tiempo, como muestra la comparación entre ventanas. El tiempo esperado hasta un par complementario que aparece en el resumen es el valor implicado por esa cadena agregada, no una estimación de cuánto tardan las personas.
- Tratar la estrategia como estado observado es incorrecto en las rondas presentes.
- Nada de esto establece un mecanismo. Describe el proceso que cualquier modelo candidato tendría que reproducir. La propuesta de modelos rivales está en [`docs/MECHANISM_PROPOSAL.md`](../../docs/MECHANISM_PROPOSAL.md).

## Reproducción

```bash
python audit/strategy_chain/build_strategy_chain.py
```

Requiere NumPy y pandas. Tarda unos 25 segundos y escribe en esta carpeta el resumen en JSON y las matrices de conteos y de transición.
