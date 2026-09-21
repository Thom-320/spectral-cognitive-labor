# ¿Puede este diseño distinguir los modelos que queremos comparar?

Antes de correr el contraste de la [propuesta mecanística](../../docs/MECHANISM_PROPOSAL.md)
sobre los datos humanos, hay que saber si el diseño del experimento original puede
distinguir sus modelos rivales. Si no puede, tanto un resultado positivo como uno
negativo serían ininterpretables, y el trabajo de escribirlos sería tiempo perdido.

Esta carpeta responde esa pregunta con datos sintéticos. **No contiene ninguna
comparación de modelos sobre las personas.** Ese contraste sigue sin correrse y no
debe correrse antes de acordar el protocolo.

## Respuesta corta

**El contraste, tal como está escrito hoy en la propuesta, no se debe correr.**
Con 45 díadas recupera el modelo verdadero el 73 % de las veces cuando el
generador es M3, y el intervalo de confianza excluye el cero solo el **11 %** de
las veces con un efecto del tamaño que los propios datos sugieren. Un resultado
nulo no significaría nada.

El problema no es la idea sino la parametrización: M3 escrito como tabla de
frecuencias tiene 135 celdas de condicionamiento y solo 44 tienen datos.

## Qué se simuló

Estado y covariables son los de la propuesta, sobre las transiciones entre rondas
ausentes consecutivas del mismo jugador: 2.554 transiciones en 45 díadas, 56,8 por
díada.

| Modelo | Condiciona en | Celdas con soporte |
|---|---|--:|
| M0 | nada | 1 de 1 |
| M1 | estrategia actual `z` | 9 de 9 |
| M2 | `z` y el bin de score `s` | 22 de 27 |
| M3 | `z`, `s` y la dirección del solapamiento `o` | **44 de 135** |

El predictor de M3 es la **dirección** del solapamiento, no su tamaño. La razón es
una identidad que se cumple exactamente en las 5.400 filas del conjunto:

```
Score = (32 si acierta, -64 si no) - Joint,   y  Joint = |casillas destapadas por ambos|
```

El script lo comprueba antes de seguir. Como el score determina el tamaño del
solapamiento dado el acierto, condicionar en el score ya lo incluye: si M3 usara el
tamaño, M3 estaría contenido en M2 y el contraste sería vacío.

El generador simula la cadena secuencialmente: el estado inicial sale de la
distribución empírica, las covariables de `p(s,o | z)` estimada sobre los datos
reales, y el estado siguiente del modelo generador. La evaluación es la de la
propuesta: dejar una díada fuera, log-loss por observación, bootstrap agrupado por
díada.

## 1. Recuperación

Fracción de 200 réplicas en las que cada generador es recuperado, con la constante
de suavizado elegida por validación cruzada interna.

| Generador | Recuperado | Se elige en su lugar |
|---|--:|---|
| M0 | 0,98 | M1 (0,01) |
| M1 | 0,99 | M2 (0,01) |
| M2 | 1,00 | — |
| **M3** | **0,73** | **M2 (0,27)** |

Los tres primeros se recuperan casi siempre. M3 no. Y el sesgo tiene una dirección:
cuando falla, se elige el modelo más simple, nunca uno más complejo.

## 2. Potencia frente al tamaño de efecto

`P_theta` interpola geométricamente entre la mejor aproximación M2 y M3, de modo que
`theta = 1` es el anclaje estimado sobre humanos y `theta > 1` amplifica la
dependencia del patrón de solapamiento. El efecto oráculo es analítico: la ganancia
esperada de log-loss del generador sobre su mejor aproximación M2, en nats.

| Efecto oráculo | P(se elige M3) | P(el IC excluye 0) |
|--:|--:|--:|
| 0,0009 | 0,00 | 0,00 |
| 0,0033 | 0,03 | 0,00 |
| 0,0070 | 0,32 | 0,01 |
| **0,0118** (anclaje) | 0,78 | **0,11** |
| 0,0249 | 1,00 | 0,88 |
| 0,0431 | 1,00 | 1,00 |

**El efecto mínimo detectable con 80 % de potencia está entre 0,020 y 0,025 nats,
aproximadamente el doble del que los datos sugieren.**

Conviene separar dos cosas que se confunden: elegir el mejor modelo y poder afirmar
que gana. Con el efecto anclado, M3 se elige el 78 % de las veces pero solo el 11 %
de los intervalos excluye el cero. Un protocolo que declare ganador al de menor
log-loss parece funcionar; uno que exija un intervalo, que es lo correcto, no.

## 3. Cuántas díadas harían falta

Generador M3, efecto anclado.

| Díadas | Transiciones | P(se elige M3) | P(el IC excluye 0) |
|--:|--:|--:|--:|
| 45 | 2.554 | 0,70 | 0,16 |
| 90 | 5.108 | 1,00 | 0,82 |
| 180 | 10.216 | 1,00 | 1,00 |

Con esta parametrización harían falta unas **90 díadas**, el doble del experimento
original, para alcanzar 80 % de potencia.

## 4. Heterogeneidad entre díadas

Esta sección responde a una objeción concreta: si las díadas difieren entre sí, un
modelo puede ajustar bien dentro de muestra y no transferir, y eso no refutaría el
mecanismo. Se generó desde **M2**, es decir con M3 falso, dando a cada díada su
propia tabla `Dirichlet(tau · M2)`.

| tau | Δ dentro de muestra | Δ fuera de díada | Se elige M3 |
|---|--:|--:|--:|
| homogéneo | +0,0140 | −0,0094 | 0,00 |
| 200 | +0,0139 | −0,0097 | 0,00 |
| 50 | +0,0143 | −0,0096 | 0,00 |
| 10 | +0,0157 | −0,0108 | 0,00 |

El cambio de signo entre dentro y fuera aparece **ya con díadas idénticas**: lo
produce el sobreajuste del modelo rico, no la heterogeneidad, que solo lo ensancha
un poco. Es una corrección a la objeción tal como se planteó. La parte que sí se
sostiene es la conclusión: una ventaja dentro de muestra que no transfiere no
demuestra por sí sola que el mecanismo sea falso, porque aquí ocurre exactamente
eso en datos donde M2 es verdadero.

Lo que la simulación añade es que, en este diseño, la heterogeneidad **no** fabrica
victorias espurias de M3 fuera de díada: nunca se eligió M3 en 100 réplicas.

## 5. El suavizado decide el resultado

Fracción de réplicas en que se recupera el generador, variando la constante `kappa`
del retroceso jerárquico.

| kappa | recupera M1 | recupera M2 | recupera M3 |
|---|--:|--:|--:|
| 0,5 | 1,00 | 1,00 | 0,00 |
| 2 | 1,00 | 1,00 | 0,16 |
| 5 | 1,00 | 1,00 | 0,26 |
| 20 | 0,06 | 1,00 | 0,96 |
| 80 | 0,00 | 0,00 | 1,00 |
| **elegido por CV interna** | **1,00** | **1,00** | **0,72** |

Con `kappa = 80` M3 se «recupera» el 100 % de las veces, pero M1 y M2 el 0 %: no es
recuperación, es un sesgo hacia el modelo complejo. Fijar esa constante a mano
permite obtener casi cualquier conclusión, así que **el protocolo debe elegirla
dentro del entrenamiento**, y decirlo por escrito antes de mirar los datos.

Elegirla con validación cruzada interna sobre toda la réplica, como hace el script
principal, es un atajo: la díada que luego se deja fuera participa en esa elección.
[`nested_kappa_check.py`](nested_kappa_check.py) mide cuánto importa reeligiendo
`kappa` dentro de cada pliegue de entrenamiento, con 40 réplicas. El atajo no infla
nada: M3 se recupera 0,72 con el atajo y **0,78** con la versión estricta, y las
diferencias medias de log-loss coinciden en la tercera cifra.

## 6. Incluir las rondas presentes empeora las cosas

La variante `--mode all` usa las transiciones de calendario, es decir incluye las
rondas en que el objetivo aparece: **5.310 transiciones, el doble**. Aun así es
peor en todo lo que importa.

| | ausentes consecutivas | todas las rondas |
|---|--:|--:|
| Transiciones | 2.554 | 5.310 |
| Efecto oráculo anclado | 0,0118 nats | **0,0066** |
| Recuperación de M3 | 0,73 | **0,48** |
| Potencia con el efecto anclado | 0,11 | **0,04** |

La razón es la censura ya documentada en la [cadena de estrategias](../strategy_chain/README.md):
las rondas presentes terminan al encontrar el objetivo y el 85,7 % de sus filas se
leen como búsqueda aleatoria. Añaden observaciones pero diluyen la señal, y el
segundo efecto domina. Es evidencia independiente a favor de restringir el análisis
a rondas ausentes consecutivas, como hace la propuesta.

## Réplica con otra semilla

Todo lo anterior se repitió con la semilla 991, con 200 réplicas nuevas. La
recuperación de M3 es 0,73 y 0,77; la potencia con el efecto anclado, 0,11 y 0,13;
la fracción en que se elige M3 con ese efecto, 0,78 en las dos. Las conclusiones no
dependen de una sola corrida. La salida está en
[`power_recovery_absent_seed991_output.txt`](power_recovery_absent_seed991_output.txt).

## Alcance, y por qué importa

Estos modelos son **tablas de frecuencias**, no los modelos del estudio original.
MBIASES, WSLS y FRA son paramétricos y de baja dimensión. Sus conteos se leen de la
Tabla 3 del artículo: con `AIC = devianza + 2k`, los pares publicados 1612/1620,
654/668 y 557/577 dan **4, 7 y 10 parámetros libres**. M3 como tabla tiene 1.080.
Por tanto:

- **La falta de potencia medida aquí es de la parametrización propuesta, no del
  experimento ni de FRA.** Una versión paramétrica del mismo contraste tendría muchos
  más grados de libertad por observación y probablemente mucha más potencia.
- Por la misma razón, el AIC calculado aquí no dice nada sobre el AIC del estudio
  original. Con estos conteos de parámetros el AIC dentro de muestra elige M1 el
  71 % de las veces cuando el generador es M2; es un diagnóstico del modelo de
  tablas, no de la Tabla 3 del artículo.

Otras limitaciones declaradas:

- El estado observado se trata como si fuera el estado real. La propuesta reconoce
  que está truncado en las rondas presentes; por eso el análisis se restringe a
  rondas ausentes consecutivas. Un modelo con estado latente tendría **menos**
  información por observación, así que estas cifras de potencia son una **cota
  superior optimista**.
- El extremo `theta = 1` está anclado en el ajuste completo sobre humanos y hereda su
  sobreajuste, de modo que también es optimista.
- El modelo de covariables `p(s,o | z)` ignora que el solapamiento depende también
  del estado del compañero.
- El bootstrap percentil agrupado cubre algo por debajo de su nivel nominal con
  pocas díadas: medido en 0,92–0,94 frente a 0,95 con 30 y con 45 díadas. Es decir,
  la prueba de «el intervalo excluye el cero» es ligeramente anticonservadora, y aun
  así la potencia es del 11 %.

## Qué se sigue de esto

1. **No correr el contraste con esta parametrización.** Un resultado nulo no
   distinguiría entre «no hay efecto» y «no hay potencia».
2. **Reescribir M2 y M3 como modelos paramétricos** en la forma de WSLS y FRA, y
   repetir esta simulación antes de tocar los datos humanos. Es el trabajo que
   decide si el proyecto tiene un artículo.
3. **Fijar por escrito, antes de mirar**, la regla de suavizado (elegida dentro del
   entrenamiento), el margen de utilidad y la unidad de validación.
4. Si tras la versión paramétrica la potencia sigue siendo baja, eso **también es un
   resultado**: que el diseño original no permite separar los mecanismos que compara,
   pese a las diferencias de AIC publicadas. Pero es un resultado que hay que
   demostrar sobre los modelos del artículo, no sobre estas tablas.

## Reproducción

```bash
python audit/model_recovery/power_recovery.py --reps 200 --mode absent
python audit/model_recovery/power_recovery.py --reps 200 --mode all
python audit/model_recovery/nested_kappa_check.py --reps 40
python -m unittest discover -s tests -p 'test_power_recovery.py' -v
```

Requiere NumPy y pandas. Medido en este equipo: la corrida principal, unos 5
minutos con 200 réplicas; la anidada, unos 10 con 40. Una semilla distinta escribe
su propio `.json` y no pisa el resultado registrado. Las salidas registradas están en `power_recovery_absent_output.txt`,
`power_recovery_all_output.txt` y `nested_kappa_check_output.txt`, con los valores en
los `.json` del mismo nombre. Las quince pruebas de `tests/test_power_recovery.py`
comprueban las propiedades del ajuste, del generador, del efecto oráculo analítico y
de la cobertura del intervalo.
