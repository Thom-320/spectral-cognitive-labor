# Propuesta de contraste mecanístico

Borrador para discutir con Edgar Andrade-Lotero y Esteban Vargas Bernal. No es un acuerdo del equipo ni una preregistración. Nada de esto se ejecuta antes de fijar por escrito el estado, la observación y el criterio de cierre.

Experimento y datos: Andrade-Lotero y Goldstone (2021), *PLOS ONE* 16(7): e0254532. Reanálisis: Thomas Chisica.

## Pregunta

¿Qué información hace falta para explicar cómo una díada llega a un reparto complementario y lo mantiene: solo la inercia de la propia estrategia, además el feedback de la ronda, o además la estructura espacial del conflicto con el compañero?

Esta pregunta sustituye **como prioridad** a la anterior, que era si una representación espectral predice la especialización. No se afirma que aquella quede refutada: es una decisión de alcance. La respaldan la cercanía entre los subespacios de Fiedler y de coordenadas, 6,59° de ángulo principal, el hecho de que el corte axial sea el mínimo global de conductancia del tablero, y la ausencia de una ventaja espectral estable tras una exploración retrospectiva extensa. Lo espectral se conserva como descripción geométrica.

## Estado y observación

Se propone **como hipótesis de modelado** que existe una estrategia de ronda que no siempre coincide con la categoría reconstruida. El estudio original ya distingue decisión interna de conducta observada mediante su capa de *shaky hand*, con una probabilidad de no temblor de aproximadamente 0,88. Lo que aquí se añade es una segunda fuente de discrepancia, el truncamiento cuando el objetivo aparece.

- **Estado** `Z_t`: la región que el jugador está usando en la ronda `t`, con los nueve valores del estudio original.
- **Observación** `Y_t`: la categoría que se puede leer de las casillas destapadas.

La observación no está truncada por el hallazgo del objetivo en las rondas ausentes, y por eso informa mucho mejor sobre la región ejecutada. Eso no garantiza que el jugador recorriera la región que tenía en mente, porque puede declarar «ausente» cuando considera que ya tiene evidencia suficiente. En rondas presentes la partida termina al encontrar el objetivo, así que la lectura está truncada: el 85,7 % de esas filas aparecen como búsqueda aleatoria, frente al 58,6 % en ausentes, y se destapan 18,8 casillas de media frente a 32,3. Por eso el modelo necesita un proceso de observación explícito, con la probabilidad de leer RS cuando el estado es focal creciendo con la brusquedad del corte de la ronda.

Tratar `Y_t` como si fuera `Z_t` cambia las conclusiones. La persistencia de una estrategia es 0,92 entre rondas ausentes consecutivas y 0,53 entre rondas de calendario.

## Modelos rivales

Todos sobre el mismo estado y la misma observación, y todos por díada.

| Modelo | Información que usa | Qué afirma |
|---|---|---|
| M0 | Frecuencias marginales | No hay dinámica: la estrategia se sortea cada ronda |
| M1 | `Z_t` | Solo inercia: se repite lo que se estaba haciendo |
| M2 | `Z_t` y el score de la ronda | Ganar y perder mueve la estrategia, al estilo WSLS |
| M3 | `Z_t`, el score y el patrón espacial del solapamiento | Interacción: el conflicto espacial y la cercanía a complementos focales mueven la estrategia |

M3 es la estructura del modelo FRA, *Focal Regions as Attractors*, del estudio original, escrita como proceso sobre estados. Su estado formal allí es `(i, s, j)`: la región propia de la ronda anterior, el score y la región de casillas solapadas. **El jugador nunca ve la región completa del compañero**, solo el solapamiento, del que puede inferirla. Un modelo que estime explícitamente una creencia sobre la región del compañero sería un cuarto modelo distinto, no FRA.

**El score ya contiene una señal social.** En este juego vale 32 o −64 según el acierto, menos las casillas destapadas por ambos, y esa identidad se cumple en las 5.400 filas del conjunto. Por eso M2 frente a M3 **no** contrasta información propia contra información del compañero. Contrasta un feedback escalar que ya incorpora el coste del solapamiento frente a la estructura espacial de ese conflicto. Si se quisiera separar ambas cosas habría que descomponer el score en acierto y penalización, lo que ya no replica WSLS ni FRA y sería otra familia de modelos.

## Predicciones que los distinguen

1. **Formación.** Una cadena homogénea de primer orden sí puede aumentar la ocupación con el tiempo si arranca lejos de su distribución estacionaria, así que el aumento por sí solo no distingue modelos. Lo que sí distingue es cuánto aumento explica: la cadena de pares estimada con las rondas 1 a 20, proyectada hacia adelante desde la distribución observada en la ronda 1, se estabiliza en el 5,2 % de ocupación, mientras que lo observado en 41 a 60 es el 19,6 %. Las propias probabilidades de transición cambian con el tiempo. El contraste correcto es condicional: dado el mismo estado actual, ¿mejoran la predicción el score y el patrón de solapamiento?
2. **Asimetría de entrada.** M3 predice que la probabilidad de entrar en un par complementario sube cuando el solapamiento de la ronda anterior fue alto. M1 y M2 no distinguen esa condición.
3. **Ausencia de intercambio directo.** De 342 transiciones desde un par complementario no hay ni un intercambio de lados entre rondas consecutivas. Es una restricción que cualquier modelo ajustado debe reproducir, y se comprueba simulando desde el modelo y contando cuántos intercambios genera. No se asume de antemano que haga falta un término de identidad: un modelo simétrico en parámetros puede conservar el reparto solo por condicionar en el estado propio.
4. **Efecto del truncamiento.** Con el proceso de observación explícito, los tres modelos deben predecir la caída de categorías focales en rondas presentes sin necesidad de un parámetro nuevo. Si hace falta ajustarlo a mano, el proceso de observación está mal escrito.

## Evaluación

- **Unidad de validación: la díada.** Dejar una díada fuera, nunca rondas sueltas.
- **Métrica: log-loss por observación** sobre el mismo conjunto de posiciones para todos los modelos, más el acierto como referencia secundaria.
- **Incertidumbre: bootstrap agrupado por díada**, con el estimando declarado antes de mirar.
- **Baseline con banda, no con un número.** Con las categorías como estado observado, una cadena de primer orden da entre 0,541 y 0,581 de log-loss en triples de rondas ausentes consecutivas y entre 0,736 y 0,739 sobre todas las rondas, según el suavizado y el soporte de entrenamiento. Antes del contraste se fija una única regla de suavizado y de respaldo, y se aplica igual dentro de cada pliegue. El detalle está en la [auditoría de la cadena](../audit/strategy_chain/README.md).
- **Margen de utilidad declarado antes.** Se acuerda qué diferencia de log-loss se considera materialmente relevante, para poder distinguir un efecto pequeño de una muestra insuficiente.
- **Sin selección posterior.** Los cuatro modelos se fijan antes; no se añaden variables para rescatar un resultado.

## Criterio de cierre

- Si M3 no mejora a M2 con un intervalo que excluya cero, la conclusión es que **no se obtiene evidencia** de valor predictivo incremental de la estructura del solapamiento bajo estos modelos y estos datos. Un intervalo que cruza cero puede significar efecto nulo, efecto pequeño, muestra insuficiente o modelo mal especificado, así que no demuestra que esa información no haga falta. Afirmar que no es materialmente necesaria exige que el intervalo quepa dentro del margen acordado.
- Si ningún modelo bate a la cadena de primer orden, la conclusión es que ninguno de los mecanismos candidatos demuestra valor incremental sobre ese baseline, no que el proceso verdadero sea inercia más truncamiento. El proyecto cerraría con la parte descriptiva: la medida, el certificado del mínimo global y la auditoría del proceso de observación.
- Si M3 gana, el siguiente paso es una manipulación que cambie la información disponible sobre el compañero, que ya requiere experimento nuevo y otra conversación.

## Qué no entra todavía

Un modelo tipo Ising sobre las 64 casillas queda fuera de este contraste. Sobre este tablero, sin restricción su mínimo de energía es ALL/NOTHING y con ambos jugadores repartiéndose es el corte de 22 aristas, es decir LR o TB, lo que lo hace un candidato atractivo. Pero entra solo si produce una predicción que M3 no produzca. Ajustar sus acoplamientos hasta reproducir lo ya conocido no sería evidencia.

## Trabajo estimado

Unas 12 a 16 horas de Thomas para escribir los cuatro modelos y el proceso de observación, y de 4 a 6 más para la evaluación y el informe, suponiendo que el estado y la observación queden acordados antes. Cómputo de minutos en un portátil.

## Reparto propuesto, sujeto a acuerdo

- **Andrade** decide si el estado latente y el proceso de observación son fieles a la tarea, y si M3 representa el modelo original.
- **Esteban** revisa la formulación como cadena oculta, la identificabilidad de los parámetros y si conviene una dinámica alternativa.
- **Thomas** implementa, evalúa y documenta, y mantiene el registro de lo ya explorado.

Evidencia de apoyo: [cadena de estrategias](../audit/strategy_chain/README.md), [certificado del mínimo global](../audit/optimum_certificate/README.md), [registro de AUC exploradas](EXPLORED_AUC_REGISTER.md).
