# Cadena de Markov sobre las estrategias del estudio original

Descriptivo. No ajusta modelos predictivos, no construye ningún outcome y no toca los resultados históricos.

## Fuente

`data/raw/humans_full.csv` del repositorio original
([enlace](https://raw.githubusercontent.com/EAndrade-Lotero/SODCL/master/Data/humans_full.csv),
sha256 `9452e653a5ffd08f8148f15269d73b754f782ed7f48bc1c3292dc78b42105c54`, 5.400 filas).
Es el archivo que el proyecto no estaba usando: contiene las 60 rondas, presentes y ausentes, con `Category` en cada fila.

Comprobaciones hechas al descargarlo:

- Las 1.244 filas compartidas con `humans_only_absent.csv` coinciden en `Category`, tras renombrar UP y DOWN a TOP y BOTTOM, y en `DLIndex`.
- `Category` equivale, en las 5.400 filas, a la coincidencia exacta de las visitas de esa fila con una de las ocho regiones focales, y RS en cualquier otro caso.
- `Consistency` **no** coincide entre los dos archivos. El filtrado compara con la ronda ausente anterior del mismo jugador.

## Qué se construye

- Cadena por jugador sobre los nueve estados LEFT, RIGHT, TOP, BOTTOM, IN, OUT, ALL, NOTHING y RS, con transiciones entre rondas consecutivas de calendario.
- La misma cadena restringida a las rondas 1–20, a las 41–60 y a las rondas ausentes consecutivas.
- Cadena por díada sobre el par no ordenado de estrategias, 18 estados observados.
- Tiempos de permanencia, escalas implícitas, y tiempo esperado hasta alcanzar un par focal complementario.

La distribución estacionaria solo se reporta cuando la cadena estimada es irreducible. En las cadenas restringidas a rondas ausentes no lo es, porque IN aparece como absorbente con muy pocos datos.

## Hallazgos

**La condición de la ronda cambia el estado observado.** En rondas presentes se destapan 18,8 casillas de media y el 85,7 % de las filas son RS. En rondas ausentes se destapan 32,3 y RS baja al 58,6 %. La ronda presente termina al encontrar el objetivo, así que la estrategia apenas se manifiesta.

**Por eso la persistencia depende de qué rondas se miren.**

| Cadena | LEFT | RIGHT | TOP | BOTTOM | ALL | RS |
|---|--:|--:|--:|--:|--:|--:|
| Rondas de calendario consecutivas | 0,53 | 0,53 | 0,50 | 0,53 | 0,35 | 0,84 |
| Solo ausentes consecutivas | 0,92 | 0,90 | 0,86 | 0,91 | 0,68 | 0,88 |

La persistencia de 0,9 que se citaba antes es propiedad del archivo filtrado, no de la conducta por ronda. La consecuencia de modelado es que la estrategia es una variable latente observada de forma censurada en las rondas presentes.

**Los pares focales complementarios se comportan como atractores que se forman con el tiempo.**

| Ventana | Rondas de díada en par focal complementario | Permanencia | Entrada desde fuera |
|---|--:|--:|--:|
| 1–20 | 4,4 % | 0,29 | 0,038 |
| 41–60 | 19,6 % | 0,49 | 0,133 |

Desde el estado RS/RS, el tiempo esperado hasta un par focal complementario es de unas 13 rondas bajo la cadena estimada con todas las rondas.

**La tipología real es axial y ALL/NOTHING.** Estados de díada más frecuentes: RS/RS 1.652, NOTHING/RS 229, ALL/RS 201, LEFT/RIGHT 136, ALL/NOTHING 112, BOTTOM/TOP 99. El par IN/OUT casi no aparece.

## Límites

- Las transiciones son estimaciones agregadas de 45 díadas; la cadena no es homogénea en el tiempo, como muestra la comparación entre ventanas.
- Tratar la estrategia como estado observado es incorrecto en las rondas presentes; un modelo serio necesita estado latente y observación censurada.
- Nada de esto establece un mecanismo. Describe el proceso que cualquier modelo candidato, FRA o un Ising mínimo, tendría que reproducir.

## Reproducción

```bash
python audit/strategy_chain/build_strategy_chain.py
```

Requiere NumPy y pandas. Escribe en esta carpeta el resumen en JSON y las matrices de conteos y de transición.
