# Propuesta de contraste mecanístico

Borrador para discutir con Edgar Andrade-Lotero y Esteban Vargas Bernal. No es un acuerdo del equipo ni una preregistración. Nada de esto se ejecuta antes de fijar por escrito el estado, la observación y el criterio de cierre.

Experimento y datos: Andrade-Lotero y Goldstone (2021), *PLOS ONE* 16(7): e0254532. Reanálisis: Thomas Chisica.

## Pregunta

¿Qué información hace falta para explicar cómo una díada llega a un reparto complementario y lo mantiene: solo la inercia de la propia estrategia, además el resultado propio, o además la conducta del compañero?

Esta pregunta sustituye a la anterior, que era si una representación espectral predice la especialización. Esa quedó cerrada por dos hechos ya certificados en el repositorio: el subespacio de Fiedler está a 6,59° del de coordenadas, y el corte axial es el mínimo global de conductancia del tablero. Lo espectral queda como descripción geométrica estática, que es lo que los datos sostienen.

## Estado y observación

El objeto no es la categoría observada, sino una estrategia latente por jugador.

- **Estado** `Z_t`: la región que el jugador está usando en la ronda `t`, con los nueve valores del estudio original.
- **Observación** `Y_t`: la categoría que se puede leer de las casillas destapadas.

La observación es fiable en rondas ausentes, donde el jugador recorre su región entera. En rondas presentes la partida termina al encontrar el objetivo, así que la lectura está truncada: el 85,7 % de esas filas aparecen como búsqueda aleatoria, frente al 58,6 % en ausentes, y se destapan 18,8 casillas de media frente a 32,3. Por eso el modelo necesita un proceso de observación explícito, con la probabilidad de leer RS cuando el estado es focal creciendo con la brusquedad del corte de la ronda.

Tratar `Y_t` como si fuera `Z_t` cambia las conclusiones. La persistencia de una estrategia es 0,92 entre rondas ausentes consecutivas y 0,53 entre rondas de calendario.

## Modelos rivales

Todos sobre el mismo estado y la misma observación, y todos por díada.

| Modelo | Información que usa | Qué afirma |
|---|---|---|
| M0 | Frecuencias marginales | No hay dinámica: la estrategia se sortea cada ronda |
| M1 | `Z_t` | Solo inercia: se repite lo que se estaba haciendo |
| M2 | `Z_t` y el resultado propio de la ronda | Ganar y perder mueve la estrategia, al estilo WSLS |
| M3 | `Z_t`, el resultado propio, el solapamiento y la región del compañero | Interacción: el reparto se negocia con evidencia del otro |

M3 es la estructura del modelo FRA del estudio original, escrita como proceso sobre estados. La comparación interesante es M2 frente a M3, que pregunta si hace falta información del compañero, y M1 frente a M2, que pregunta si hace falta resultado.

## Predicciones que los distinguen

1. **Formación.** M1 no puede aumentar la ocupación de pares complementarios con el tiempo salvo por azar. En los datos pasa del 4,4 % de las rondas de díada en 1–20 al 19,6 % en 41–60. M2 puede hacerlo solo si el resultado correlaciona con la complementariedad; M3 lo predice directamente.
2. **Asimetría de entrada.** M3 predice que la probabilidad de entrar en un par complementario sube cuando el solapamiento de la ronda anterior fue alto. M1 y M2 no distinguen esa condición.
3. **Ausencia de intercambio directo.** De 342 transiciones desde un par complementario no hay ni un intercambio de lados entre rondas consecutivas. Un M3 simétrico, sin memoria de quién ocupaba qué, tiende a permitirlos: el modelo necesita un término de identidad, y eso es una predicción comprobable, no un ajuste.
4. **Efecto del truncamiento.** Con el proceso de observación explícito, los tres modelos deben predecir la caída de categorías focales en rondas presentes sin necesidad de un parámetro nuevo. Si hace falta ajustarlo a mano, el proceso de observación está mal escrito.

## Evaluación

- **Unidad de validación: la díada.** Dejar una díada fuera, nunca rondas sueltas.
- **Métrica: log-loss por observación** sobre el mismo conjunto de posiciones para todos los modelos, más el acierto como referencia secundaria.
- **Incertidumbre: bootstrap agrupado por díada**, con el estimando declarado antes de mirar.
- **Baseline ya medido.** Con las categorías como estado observado, una cadena de primer orden da log-loss 0,559 en rondas ausentes consecutivas y 0,736 sobre todas las rondas. Cualquier modelo mecanístico tiene que batir eso para ser interesante.
- **Sin selección posterior.** Los cuatro modelos se fijan antes; no se añaden variables para rescatar un resultado.

## Criterio de cierre

- Si M3 no mejora a M2 con intervalo que excluya cero, se concluye que la información del compañero no es necesaria para explicar esta dinámica en estos datos, y se publica como resultado negativo.
- Si ningún modelo bate a la cadena de primer orden, se concluye que la dinámica observable es inercia más truncamiento, y el proyecto cierra con la parte descriptiva: la medida, el certificado del mínimo global y la auditoría del proceso de observación.
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
