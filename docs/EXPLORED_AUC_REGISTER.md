# Registro de AUC exploradas: Spectral/SODCL

Generado el 2026-09-11 por `scripts/build_auc_register.py` a partir de artefactos existentes. **No recalcula ni ajusta modelos.** Cada fila es un valor que alguien del proyecto o de sus auditorías ya vio. Se incluye aunque no se haya reportado, porque haberlo visto condiciona cualquier análisis posterior.

Experimento y datos: Andrade-Lotero y Goldstone (2021), PLOS ONE 16(7): e0254532. Reanálisis: Thomas Chisica. Todas las AUC son exploratorias y retrospectivas; ninguna es confirmatoria.

## Resumen

- **163 valores de AUC** en **6 familias** y **45 combinaciones** distintas de ventana temprana, outcome y población.
- Las AUC LOOCV de la señal geométrica sola van de **0,361** a **1,000**. Los valores por encima de 0,92 usan ventanas tempranas que ya contienen parte del outcome.
- **La cohorte reconstruida ya fue evaluada.** Con las primeras cinco oportunidades ausentes reales (E_P5) y orientación tardía por plantillas, el paquete para Esteban obtuvo AUC 0,361 con n=28 (fila A015). Cualquier análisis futuro sobre esa cohorte no es ciego.
- **La AUC depende de cuánto se solapa la ventana temprana con el outcome.** En all_dyads, con 2 a 4 rondas del archivo filtrado cae a 0,64–0,69; con 8 o más supera 0,92 porque invade las ventanas del target.
- **Con un outcome de desempeño (T_SCORE), las métricas oficiales superan a la geometría**: 0,856 frente a 0,745 en LOOCV.
- **Plantillas lineales, proyectadas o norma en el espacio de Fiedler dan prácticamente la misma AUC**: difieren como mucho 0,01 en LOOCV y 0,017 sin modelo.
- Ninguna comparación tiene un intervalo de incertidumbre con reajuste por díada ni agrupación por sesión.

## Estado de las fuentes

| Código | Significado |
|---|---|
| DOC-REPO | Guardado en el repositorio; reproducido desde las features congeladas por Claude, el auditor de datos y la sesión de Codex. |
| DOC-PKT | Documentado en el paquete para Esteban; no reejecutado en septiembre. |
| CLAUDE | Cálculo exploratorio de Claude; script disponible; no revisado por una persona. Ajustó modelos pese a la regla de la fase uno. |
| AGENT | Cálculo exploratorio de un subagente; script y salida disponibles; no verificado adversarialmente ni revisado por una persona. |

## Códigos

**Ventana temprana**

| Código | Definición |
|---|---|
| E_H5 | Primeras 5 rondas comunes de humans_only_absent.csv (filtro ausente→ausente). La 5ª llega a ronda ≥30 en 15/45 díadas. |
| E_Hk | Primeras k rondas comunes de humans_only_absent.csv. |
| E_H≤r | Rondas calendario ≤ r de humans_only_absent.csv. |
| E_H5-noov | E_H5 excluyendo las 15 díadas cuya 5ª ronda es ≥30. |
| E_P5 | Primeras 5 oportunidades realmente ausentes de performances.csv; cortes en rondas 6–19. Es la cohorte reconstruida de TEMPORAL_REPAIR. |
| E_P≤r | Todas las rondas ausentes con ronda ≤ r de performances.csv. |

**Outcome tardío**

| Código | Definición |
|---|---|
| T_STABLE | stable_orientation de 06: mayoría de las ventanas anidadas 30–60, 40–60 y 50–60, con plantillas LR/TB sobre humans_only_absent. Positivo = LR o TB. |
| T_STABLE* | Como T_STABLE, pero con umbrales de orientación o de tamaño de partición cambiados (ver notas). |
| T_O4060 | Orientación por plantillas solo en la ventana 40–60 de humans_only_absent. |
| T_CAT40 | external_label_40_60 de 06: la unión de Category de ambos jugadores en rondas ≥40 contiene LEFT y RIGHT, o TOP y BOTTOM. |
| T_CAT41 | Igual que T_CAT40 con rondas ≥41 (check3.py, ext_label). |
| T_CATFRAC | Al menos la mitad de las filas de rondas ≥41 tienen Category axial (LEFT, RIGHT, TOP, BOTTOM). |
| T_MODES | Category modal de cada jugador en rondas ≥40: el par es LEFT/RIGHT o BOTTOM/TOP. |
| T_MODES_IO | Como T_MODES, incluyendo el par IN/OUT. |
| T_SCORE | Score medio en rondas presentes 41–60 de performances.csv mayor que la mediana. |
| T_ORIENT_H | Orientación por plantillas en rondas 41–60 de humans_only_absent.csv; solo díadas con partición tardía de tamaño válido. |
| T_ORIENT_P | Orientación por plantillas en todas las rondas ausentes 41–60 de performances.csv; solo díadas con partición tardía de tamaño válido. |

**Predictores**

| Código | Definición |
|---|---|
| G | dominant_score: máxima |correlación| del margen temprano con plantillas lineales LR/TB. No usa el Laplaciano. |
| G_proj | dominant_score con plantillas proyectadas al espacio de Fiedler. |
| G_eig | ||P_F m||/||m||: norma del margen proyectado al espacio de Fiedler, independiente de la base. |
| H | early_h_obs: conductancia de la partición temprana. |
| S | Similarity_early. |
| D | DLIndex_early. |
| C | Consistency_early. |
| TRIO | D + S + C. |
| COMB | G + D + S + C. |
| G+S | G + S. |

**Tipo de AUC.** LOOCV: regresión logística con imputación por mediana y estandarización, ajustada dejando fuera una díada cada vez. In-sample: el mismo modelo evaluado sobre sus datos de ajuste. Sin modelo: el predictor usado directamente como puntuación.

## F1. Pipeline histórico del repositorio

Autor: Thomas (pipeline histórico). Fecha: 2026-05-23 (commit 7271fa0; congelado en 3f3b70c). Estado: DOC-REPO. Fuente: `data/results/early_prediction_summary.csv; src/08_early_prediction.py`.

| ID | Ventana | Outcome | Población | n | Pos. | Predictores | Tipo | AUC | Nota |
|---|---|---|---|--:|--:|---|---|--:|---|
| A001 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,804 |  |
| A002 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | in-sample | 0,834 |  |
| A003 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | TRIO | LOOCV | 0,736 |  |
| A004 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | TRIO | in-sample | 0,808 |  |
| A005 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | COMB | LOOCV | 0,860 |  |
| A006 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | COMB | in-sample | 0,890 |  |
| A007 | E_H5 | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 0,779 |  |
| A008 | E_H5 | T_STABLE | valid_only | 32 | 20 | G | in-sample | 0,825 |  |
| A009 | E_H5 | T_STABLE | valid_only | 32 | 20 | TRIO | LOOCV | 0,775 |  |
| A010 | E_H5 | T_STABLE | valid_only | 32 | 20 | TRIO | in-sample | 0,858 |  |
| A011 | E_H5 | T_STABLE | valid_only | 32 | 20 | COMB | LOOCV | 0,812 |  |
| A012 | E_H5 | T_STABLE | valid_only | 32 | 20 | COMB | in-sample | 0,908 |  |

## F2. Paquete metodológico para Esteban

Autor: Sesión Codex/Claude previa (paquete para Esteban). Fecha: 2026-08-21. Estado: DOC-PKT. Fuente: `docs/esteban_meeting/methodology_audit.csv y reproduction_log.md (sin trackear en main; copia en ~/Downloads/sodcl-pro-review/claude_artifacts/esteban_packet_copy)`.

| ID | Ventana | Outcome | Población | n | Pos. | Predictores | Tipo | AUC | Nota |
|---|---|---|---|--:|--:|---|---|--:|---|
| A013 | E_H≤20 | T_ORIENT_H | partición tardía válida | 29 | 21 | G | LOOCV | 0,744 | Ventanas calendario 1–20 y 41–60, sin solapamiento. |
| A014 | E_P≤20 | T_ORIENT_P | partición tardía válida | 28 | 21 | G | LOOCV | 0,789 | Todas las ausentes ≤20 frente a 41–60. |
| A015 | E_P5 | T_ORIENT_P | partición tardía válida | 28 | 21 | G | LOOCV | 0,361 | Misma cohorte temprana que TEMPORAL_REPAIR. Un AUC LOOCV <0,5 con una sola variable indica ausencia de señal, no señal inversa. |
| A016 | E_H5 | T_STABLE | valid_only | 32 | 20 | G_proj | LOOCV | 0,787 | reproduction_log.md; plantillas proyectadas. |

## F3. Claude, sesión principal: recálculo y comparadores

Autor: Claude (sesión principal). Fecha: 2026-09-10. Estado: CLAUDE. Fuente: `~/Downloads/sodcl-pro-review/claude_artifacts/scratch_scripts_claude/check1.py, check2.py`.

| ID | Ventana | Outcome | Población | n | Pos. | Predictores | Tipo | AUC | Nota |
|---|---|---|---|--:|--:|---|---|--:|---|
| A017 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,834 |  |
| A018 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | S | sin modelo | 0,780 |  |
| A019 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | D | sin modelo | 0,753 |  |
| A020 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | C | sin modelo | 0,770 |  |
| A021 | E_H5 | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 0,825 |  |
| A022 | E_H5 | T_STABLE | valid_only | 32 | 20 | S | sin modelo | 0,838 |  |
| A023 | E_H5 | T_STABLE | valid_only | 32 | 20 | D | sin modelo | 0,767 |  |
| A024 | E_H5 | T_STABLE | valid_only | 32 | 20 | C | sin modelo | 0,808 |  |
| A025 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | S | LOOCV | 0,752 |  |
| A026 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G+S | LOOCV | 0,818 |  |
| A027 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | H | LOOCV | 0,772 |  |
| A028 | E_H5 | T_STABLE | valid_only | 32 | 20 | S | LOOCV | 0,800 |  |
| A029 | E_H5 | T_STABLE | valid_only | 32 | 20 | G+S | LOOCV | 0,787 |  |
| A030 | E_H5 | T_STABLE | valid_only | 32 | 20 | H | LOOCV | 0,771 |  |

## F4. Claude: ventana calendario y etiqueta por Category

Autor: Claude (sesión principal). Fecha: 2026-09-10. Estado: CLAUDE. Fuente: `~/Downloads/sodcl-pro-review/claude_artifacts/scratch_scripts_claude/check3.py, strict_external_contrast.csv`.

| ID | Ventana | Outcome | Población | n | Pos. | Predictores | Tipo | AUC | Nota |
|---|---|---|---|--:|--:|---|---|--:|---|
| A031 | E_P≤20 | T_CAT41 | todas | 45 | 15 | G | LOOCV | 0,853 |  |
| A032 | E_P≤10 | T_CAT41 | todas | 45 | 15 | G | LOOCV | 0,622 |  |
| A033 | E_H5 | T_CAT41 | todas | 45 | 15 | G | LOOCV | 0,836 |  |
| A034 | E_P≤20 | T_CAT41 | todas | 45 | 15 | S | LOOCV | 0,793 | Métricas oficiales tempranas de humans_only_absent ≤20. |
| A035 | E_P≤20 | T_CAT41 | todas | 45 | 15 | D | LOOCV | 0,700 | Métricas oficiales tempranas de humans_only_absent ≤20. |
| A036 | E_P≤20 | T_CAT41 | todas | 45 | 15 | C | LOOCV | 0,780 | Métricas oficiales tempranas de humans_only_absent ≤20. |
| A037 | E_P≤20 | T_CAT41 | todas | 45 | 15 | TRIO | LOOCV | 0,784 | Métricas oficiales tempranas de humans_only_absent ≤20. |
| A038 | E_P≤20 | T_CAT41 | todas | 45 | 15 | COMB | LOOCV | 0,880 | Métricas oficiales tempranas de humans_only_absent ≤20. |
| A039 | E_P≤20 | T_CAT41 | todas | 45 | 15 | G+S | LOOCV | 0,871 | Métricas oficiales tempranas de humans_only_absent ≤20. |
| A040 | E_P≤20 | T_CATFRAC | todas | 45 | 13 | G | LOOCV | 0,853 |  |
| A041 | E_P≤20 | T_CATFRAC | todas | 45 | 13 | S | LOOCV | 0,781 |  |
| A042 | E_P≤20 | T_CATFRAC | todas | 45 | 13 | TRIO | LOOCV | 0,825 |  |
| A043 | E_P≤20 | T_CATFRAC | todas | 45 | 13 | COMB | LOOCV | 0,899 |  |

## F5. Agente auditor de datos: sensibilidad

Autor: Subagente 'datos' del workflow de Claude. Fecha: 2026-09-10. Estado: AGENT. Fuente: `~/Downloads/sodcl-pro-review/claude_artifacts/agent_outputs/datos/04_circularity.py, 05_sensitivity.py y sus *_out.txt`.

| ID | Ventana | Outcome | Población | n | Pos. | Predictores | Tipo | AUC | Nota |
|---|---|---|---|--:|--:|---|---|--:|---|
| A044 | E_Hk (k=1) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,790 | k=1; la k-ésima ronda llega a ≥30 en 0/45 díadas. |
| A045 | E_Hk (k=1) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,820 | k=1; la k-ésima ronda llega a ≥30 en 0/45 díadas. |
| A046 | E_Hk (k=1) | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 0,654 | k=1; la k-ésima ronda llega a ≥30 en 0/45 díadas. |
| A047 | E_Hk (k=1) | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 0,742 | k=1; la k-ésima ronda llega a ≥30 en 0/45 díadas. |
| A048 | E_Hk (k=2) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,642 | k=2; la k-ésima ronda llega a ≥30 en 0/45 díadas. |
| A049 | E_Hk (k=2) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,698 | k=2; la k-ésima ronda llega a ≥30 en 0/45 díadas. |
| A050 | E_Hk (k=2) | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 0,529 | k=2; la k-ésima ronda llega a ≥30 en 0/45 díadas. |
| A051 | E_Hk (k=2) | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 0,654 | k=2; la k-ésima ronda llega a ≥30 en 0/45 díadas. |
| A052 | E_Hk (k=3) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,658 | k=3; la k-ésima ronda llega a ≥30 en 1/45 díadas. |
| A053 | E_Hk (k=3) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,710 | k=3; la k-ésima ronda llega a ≥30 en 1/45 díadas. |
| A054 | E_Hk (k=3) | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 0,592 | k=3; la k-ésima ronda llega a ≥30 en 1/45 díadas. |
| A055 | E_Hk (k=3) | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 0,675 | k=3; la k-ésima ronda llega a ≥30 en 1/45 díadas. |
| A056 | E_Hk (k=4) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,686 | k=4; la k-ésima ronda llega a ≥30 en 6/45 díadas. |
| A057 | E_Hk (k=4) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,736 | k=4; la k-ésima ronda llega a ≥30 en 6/45 díadas. |
| A058 | E_Hk (k=4) | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 0,646 | k=4; la k-ésima ronda llega a ≥30 en 6/45 díadas. |
| A059 | E_Hk (k=4) | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 0,713 | k=4; la k-ésima ronda llega a ≥30 en 6/45 díadas. |
| A060 | E_Hk (k=5) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,804 | k=5; la k-ésima ronda llega a ≥30 en 15/45 díadas. |
| A061 | E_Hk (k=5) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,834 | k=5; la k-ésima ronda llega a ≥30 en 15/45 díadas. |
| A062 | E_Hk (k=5) | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 0,779 | k=5; la k-ésima ronda llega a ≥30 en 15/45 díadas. |
| A063 | E_Hk (k=5) | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 0,825 | k=5; la k-ésima ronda llega a ≥30 en 15/45 díadas. |
| A064 | E_Hk (k=6) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,844 | k=6; la k-ésima ronda llega a ≥30 en 18/45 díadas. |
| A065 | E_Hk (k=6) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,862 | k=6; la k-ésima ronda llega a ≥30 en 18/45 díadas. |
| A066 | E_Hk (k=6) | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 0,825 | k=6; la k-ésima ronda llega a ≥30 en 18/45 díadas. |
| A067 | E_Hk (k=6) | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 0,858 | k=6; la k-ésima ronda llega a ≥30 en 18/45 díadas. |
| A068 | E_Hk (k=8) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,926 | k=8; la k-ésima ronda llega a ≥30 en 28/45 díadas. |
| A069 | E_Hk (k=8) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,936 | k=8; la k-ésima ronda llega a ≥30 en 28/45 díadas. |
| A070 | E_Hk (k=8) | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 0,925 | k=8; la k-ésima ronda llega a ≥30 en 28/45 díadas. |
| A071 | E_Hk (k=8) | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 0,937 | k=8; la k-ésima ronda llega a ≥30 en 28/45 díadas. |
| A072 | E_Hk (k=10) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,958 | k=10; la k-ésima ronda llega a ≥30 en 37/45 díadas. |
| A073 | E_Hk (k=10) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,974 | k=10; la k-ésima ronda llega a ≥30 en 37/45 díadas. |
| A074 | E_Hk (k=10) | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 0,958 | k=10; la k-ésima ronda llega a ≥30 en 37/45 díadas. |
| A075 | E_Hk (k=10) | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 0,979 | k=10; la k-ésima ronda llega a ≥30 en 37/45 díadas. |
| A076 | E_Hk (k=15) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,994 | k=15; la k-ésima ronda llega a ≥30 en 45/45 díadas. |
| A077 | E_Hk (k=15) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,994 | k=15; la k-ésima ronda llega a ≥30 en 45/45 díadas. |
| A078 | E_Hk (k=15) | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 1,000 | k=15; la k-ésima ronda llega a ≥30 en 45/45 díadas. |
| A079 | E_Hk (k=15) | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 1,000 | k=15; la k-ésima ronda llega a ≥30 en 45/45 díadas. |
| A080 | E_Hk (k=20) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | LOOCV | 0,996 | k=20; la k-ésima ronda llega a ≥30 en 45/45 díadas. |
| A081 | E_Hk (k=20) | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G | sin modelo | 0,996 | k=20; la k-ésima ronda llega a ≥30 en 45/45 díadas. |
| A082 | E_Hk (k=20) | T_STABLE | valid_only | 32 | 20 | G | LOOCV | 1,000 | k=20; la k-ésima ronda llega a ≥30 en 45/45 díadas. |
| A083 | E_Hk (k=20) | T_STABLE | valid_only | 32 | 20 | G | sin modelo | 1,000 | k=20; la k-ésima ronda llega a ≥30 en 45/45 díadas. |
| A084 | E_H5-noov | T_STABLE | all_dyads sin solapadas | 30 | 14 | G | LOOCV | 0,875 |  |
| A085 | E_H5-noov | T_STABLE | all_dyads sin solapadas | 30 | 14 | G | sin modelo | 0,906 |  |
| A086 | E_H5-noov | T_STABLE | valid_only sin solapadas | 21 | 14 | G | LOOCV | 0,867 |  |
| A087 | E_H5-noov | T_STABLE | valid_only sin solapadas | 21 | 14 | G | sin modelo | 0,929 |  |
| A088 | E_H≤10 | T_STABLE | all_dyads con rondas | 37 | — | G | LOOCV | 0,653 |  |
| A089 | E_H≤10 | T_STABLE | all_dyads con rondas | 37 | — | G | sin modelo | 0,732 |  |
| A090 | E_H≤10 | T_STABLE | valid_only con rondas | 26 | — | G | LOOCV | 0,497 |  |
| A091 | E_H≤10 | T_STABLE | valid_only con rondas | 26 | — | G | sin modelo | 0,667 |  |
| A092 | E_H≤20 | T_STABLE | all_dyads con rondas | 45 | — | G | LOOCV | 0,824 |  |
| A093 | E_H≤20 | T_STABLE | all_dyads con rondas | 45 | — | G | sin modelo | 0,848 |  |
| A094 | E_H≤20 | T_STABLE | valid_only con rondas | 32 | — | G | LOOCV | 0,800 |  |
| A095 | E_H≤20 | T_STABLE | valid_only con rondas | 32 | — | G | sin modelo | 0,825 |  |
| A096 | E_H≤29 | T_STABLE | all_dyads con rondas | 45 | — | G | LOOCV | 0,838 |  |
| A097 | E_H≤29 | T_STABLE | all_dyads con rondas | 45 | — | G | sin modelo | 0,860 |  |
| A098 | E_H≤29 | T_STABLE | valid_only con rondas | 32 | — | G | LOOCV | 0,829 |  |
| A099 | E_H≤29 | T_STABLE | valid_only con rondas | 32 | — | G | sin modelo | 0,850 |  |
| A100 | E_H5 | T_CAT40 | todas | 45 | 15 | G | sin modelo | 0,856 |  |
| A101 | E_H5 | T_CAT40 | todas | 45 | 15 | S | sin modelo | 0,827 |  |
| A102 | E_H5 | T_CAT40 | todas | 45 | 15 | D | sin modelo | 0,733 |  |
| A103 | E_H5 | T_CAT40 | todas | 45 | 15 | C | sin modelo | 0,818 |  |
| A104 | E_H5 | T_CAT40 | todas | 45 | 15 | G | LOOCV | 0,836 |  |
| A105 | E_H5 | T_CAT40 | todas | 45 | 15 | TRIO | LOOCV | 0,767 |  |
| A106 | E_H5 | T_CAT40 | todas | 45 | 15 | COMB | LOOCV | 0,858 |  |
| A107 | E_H5 | T_CAT40 | valid_only | 32 | 15 | G | sin modelo | 0,824 |  |
| A108 | E_H5 | T_CAT40 | valid_only | 32 | 15 | S | sin modelo | 0,863 |  |
| A109 | E_H5 | T_CAT40 | valid_only | 32 | 15 | D | sin modelo | 0,725 |  |
| A110 | E_H5 | T_CAT40 | valid_only | 32 | 15 | C | sin modelo | 0,847 |  |
| A111 | E_H5 | T_CAT40 | valid_only | 32 | 15 | G | LOOCV | 0,796 |  |
| A112 | E_H5 | T_CAT40 | valid_only | 32 | 15 | TRIO | LOOCV | 0,804 |  |
| A113 | E_H5 | T_CAT40 | valid_only | 32 | 15 | COMB | LOOCV | 0,820 |  |
| A114 | E_H5 | T_O4060 | todas | 45 | 23 | G | sin modelo | 0,834 |  |
| A115 | E_H5 | T_O4060 | todas | 45 | 23 | S | sin modelo | 0,759 |  |
| A116 | E_H5 | T_O4060 | todas | 45 | 23 | D | sin modelo | 0,659 |  |
| A117 | E_H5 | T_O4060 | todas | 45 | 23 | C | sin modelo | 0,741 |  |
| A118 | E_H5 | T_O4060 | todas | 45 | 23 | G | LOOCV | 0,800 |  |
| A119 | E_H5 | T_O4060 | todas | 45 | 23 | TRIO | LOOCV | 0,692 |  |
| A120 | E_H5 | T_O4060 | todas | 45 | 23 | COMB | LOOCV | 0,775 |  |
| A121 | E_H5 | T_MODES | todas | 45 | 11 | G | sin modelo | 0,837 |  |
| A122 | E_H5 | T_MODES | todas | 45 | 11 | S | sin modelo | 0,818 |  |
| A123 | E_H5 | T_MODES | todas | 45 | 11 | D | sin modelo | 0,845 |  |
| A124 | E_H5 | T_MODES | todas | 45 | 11 | C | sin modelo | 0,818 |  |
| A125 | E_H5 | T_MODES | todas | 45 | 11 | G | LOOCV | 0,816 |  |
| A126 | E_H5 | T_MODES | todas | 45 | 11 | TRIO | LOOCV | 0,818 |  |
| A127 | E_H5 | T_MODES | todas | 45 | 11 | COMB | LOOCV | 0,834 |  |
| A128 | E_H5 | T_MODES_IO | todas | 45 | 12 | G | sin modelo | 0,846 |  |
| A129 | E_H5 | T_MODES_IO | todas | 45 | 12 | S | sin modelo | 0,795 |  |
| A130 | E_H5 | T_MODES_IO | todas | 45 | 12 | D | sin modelo | 0,831 |  |
| A131 | E_H5 | T_MODES_IO | todas | 45 | 12 | C | sin modelo | 0,783 |  |
| A132 | E_H5 | T_MODES_IO | todas | 45 | 12 | G | LOOCV | 0,823 |  |
| A133 | E_H5 | T_MODES_IO | todas | 45 | 12 | TRIO | LOOCV | 0,783 |  |
| A134 | E_H5 | T_MODES_IO | todas | 45 | 12 | COMB | LOOCV | 0,848 |  |
| A135 | E_H5 | T_SCORE | todas | 45 | 22 | G | sin modelo | 0,765 |  |
| A136 | E_H5 | T_SCORE | todas | 45 | 22 | S | sin modelo | 0,864 |  |
| A137 | E_H5 | T_SCORE | todas | 45 | 22 | D | sin modelo | 0,827 |  |
| A138 | E_H5 | T_SCORE | todas | 45 | 22 | C | sin modelo | 0,824 |  |
| A139 | E_H5 | T_SCORE | todas | 45 | 22 | G | LOOCV | 0,745 |  |
| A140 | E_H5 | T_SCORE | todas | 45 | 22 | TRIO | LOOCV | 0,856 |  |
| A141 | E_H5 | T_SCORE | todas | 45 | 22 | COMB | LOOCV | 0,905 |  |
| A142 | E_H5 | T_STABLE* | all_dyads | 45 | 21 | G | LOOCV | 0,784 | ORIENT_MIN_SCORE=0,20 con cualquier ratio. Con 0,25 o 0,30 no cambia: 0,804. |
| A143 | E_H5 | T_STABLE* | valid_only | 32 | 21 | G | LOOCV | 0,753 | ORIENT_MIN_SCORE=0,20. |
| A144 | E_H5 | T_STABLE* | all_dyads | 45 | — | G | LOOCV | 0,817 | Límites de tamaño de partición [1,63] en lugar de [10,54]. |
| A145 | E_H5 | T_STABLE* | valid_only | 39 | — | G | LOOCV | 0,807 | Límites de tamaño de partición [1,63]. |
| A146 | E_H5 | T_STABLE* | all_dyads | 45 | — | G | LOOCV | 0,817 | Límites de tamaño de partición [5,59] en lugar de [10,54]. |
| A147 | E_H5 | T_STABLE* | valid_only | 36 | — | G | LOOCV | 0,800 | Límites de tamaño de partición [5,59]. |
| A148 | E_H5 | T_STABLE* | all_dyads | 45 | — | G | LOOCV | 0,817 | Límites de tamaño de partición [8,56] en lugar de [10,54]. |
| A149 | E_H5 | T_STABLE* | valid_only | 36 | — | G | LOOCV | 0,800 | Límites de tamaño de partición [8,56]. |
| A150 | E_H5 | T_STABLE* | all_dyads | 45 | — | G | LOOCV | 0,804 | Límites de tamaño de partición [12,52] en lugar de [10,54]. |
| A151 | E_H5 | T_STABLE* | valid_only | 30 | — | G | LOOCV | 0,775 | Límites de tamaño de partición [12,52]. |
| A152 | E_H5 | T_STABLE* | all_dyads | 45 | — | G | LOOCV | 0,813 | Límites de tamaño de partición [16,48] en lugar de [10,54]. |
| A153 | E_H5 | T_STABLE* | valid_only | 30 | — | G | LOOCV | 0,788 | Límites de tamaño de partición [16,48]. |
| A154 | E_H5 | T_STABLE* | all_dyads | 45 | — | G | LOOCV | 0,804 | Límites de tamaño de partición [20,44] en lugar de [10,54]. |
| A155 | E_H5 | T_STABLE* | valid_only | 23 | — | G | LOOCV | 0,433 | Límites de tamaño de partición [20,44]. |

## F6. Agente auditor de geometría: plantillas proyectadas

Autor: Subagente 'geometría' del workflow de Claude. Fecha: 2026-09-10. Estado: AGENT. Fuente: `~/Downloads/sodcl-pro-review/claude_artifacts/agent_outputs/geometria/D_templates_projection.py, D_out.txt`.

| ID | Ventana | Outcome | Población | n | Pos. | Predictores | Tipo | AUC | Nota |
|---|---|---|---|--:|--:|---|---|--:|---|
| A156 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G_proj | LOOCV | 0,806 |  |
| A157 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G_proj | sin modelo | 0,834 |  |
| A158 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G_eig | LOOCV | 0,808 |  |
| A159 | E_H5 | T_STABLE | all_dyads (13 INVALID=0) | 45 | 20 | G_eig | sin modelo | 0,842 |  |
| A160 | E_H5 | T_STABLE | valid_only | 32 | 20 | G_proj | LOOCV | 0,787 |  |
| A161 | E_H5 | T_STABLE | valid_only | 32 | 20 | G_proj | sin modelo | 0,829 |  |
| A162 | E_H5 | T_STABLE | valid_only | 32 | 20 | G_eig | LOOCV | 0,787 |  |
| A163 | E_H5 | T_STABLE | valid_only | 32 | 20 | G_eig | sin modelo | 0,842 |  |

## Resúmenes inferenciales ya vistos

Calculados por Claude en `check2.py` sobre la cohorte histórica (E_H5, T_STABLE). El bootstrap remuestrea díadas sobre predicciones LOOCV fijas: 3000 réplicas, percentil, semilla 0. **No reajusta los modelos**, así que no incorpora la incertidumbre del ajuste.

| Comparación | Población | Delta AUC | IC 95 % bootstrap |
|---|---|--:|---|
| COMB − TRIO | all_dyads, n=45 | +0,124 | [−0,004; +0,265] |
| COMB − S | all_dyads, n=45 | +0,108 | [−0,013; +0,243] |
| G − S | all_dyads, n=45 | +0,052 | [−0,122; +0,229] |
| G − H | all_dyads, n=45 | +0,032 | [−0,101; +0,174] |
| COMB − TRIO | valid_only, n=32 | +0,037 | [−0,075; +0,156] |
| COMB − S | valid_only, n=32 | +0,012 | [−0,096; +0,127] |
| G − S | valid_only, n=32 | −0,021 | [−0,198; +0,163] |
| G − H | valid_only, n=32 | +0,008 | [−0,134; +0,152] |

Nulo por permutación de etiquetas con reajuste LOOCV, 300 permutaciones, predictor G: media 0,283 en all_dyads y 0,306 en valid_only. Ninguna permutación alcanzó la AUC observada. La media muy por debajo de 0,5 ilustra el sesgo de LOOCV con una sola variable y muestras pequeñas.

## Otros resultados predictivos vistos que no son AUC

- `data/results/present_performance_increment.csv` (F1): con n=29, añadir h_obs a las métricas oficiales tardías baja el R² LOOCV del Score en rondas presentes de 0,654 a 0,497; añadir η, a 0,575.
- `agent_outputs/datos/04_out.txt` (F5): correlaciones de Spearman entre predictores tempranos E_H5 y desempeño en 41–60. Por ejemplo, con Score presente y n=45: G 0,449; S 0,590; D 0,589; C 0,525.

## Cómo usar este registro

1. Antes de mirar un resultado nuevo, añadir una fila con ventana, outcome, población, predictores, validación y fecha, dejando la AUC vacía.
2. Después de calcularlo, completar la AUC sin borrar ni editar filas anteriores.
3. Al reportar cualquier AUC, citar su ID y el número total de combinaciones ya exploradas.
4. Las cifras sin script, tabla por díada y procedimiento quedan como "reportadas, no verificadas".

Tabla completa: `docs/explored_auc_register.csv`. Para regenerar ambos archivos: `python3 scripts/build_auc_register.py`.

**Límite de trazabilidad.** Los scripts y salidas de F3 a F6, y la copia del paquete de F2, están fuera de este repositorio, en `~/Downloads/sodcl-pro-review/claude_artifacts` del VPS de Thomas. F1 es la única familia cuyos artefactos están versionados aquí.
