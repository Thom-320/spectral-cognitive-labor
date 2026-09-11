#!/usr/bin/env python3
"""Genera el registro de AUC exploradas en Spectral/SODCL (CSV + Markdown).

Transcribe valores ya observados desde sus artefactos; no recalcula ni ajusta modelos.
Uso, desde la raíz: python3 scripts/build_auc_register.py
Escribe docs/explored_auc_register.csv y docs/EXPLORED_AUC_REGISTER.md. Solo usa la biblioteca estándar.
"""
import csv
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "docs"
ART = "~/Downloads/sodcl-pro-review/claude_artifacts"

FAMILIES = OrderedDict([
    ("F1", dict(title="Pipeline histórico del repositorio", who="Thomas (pipeline histórico)", date="2026-05-23 (commit 7271fa0; congelado en 3f3b70c)",
                source="data/results/early_prediction_summary.csv; src/08_early_prediction.py", status="DOC-REPO")),
    ("F2", dict(title="Paquete metodológico para Esteban", who="Sesión Codex/Claude previa (paquete para Esteban)", date="2026-08-21",
                source=f"docs/esteban_meeting/methodology_audit.csv y reproduction_log.md (sin trackear en main; copia en {ART}/esteban_packet_copy)", status="DOC-PKT")),
    ("F3", dict(title="Claude, sesión principal: recálculo y comparadores", who="Claude (sesión principal)", date="2026-09-10",
                source=f"{ART}/scratch_scripts_claude/check1.py, check2.py", status="CLAUDE")),
    ("F4", dict(title="Claude: ventana calendario y etiqueta por Category", who="Claude (sesión principal)", date="2026-09-10",
                source=f"{ART}/scratch_scripts_claude/check3.py, strict_external_contrast.csv", status="CLAUDE")),
    ("F5", dict(title="Agente auditor de datos: sensibilidad", who="Subagente 'datos' del workflow de Claude", date="2026-09-10",
                source=f"{ART}/agent_outputs/datos/04_circularity.py, 05_sensitivity.py y sus *_out.txt", status="AGENT")),
    ("F6", dict(title="Agente auditor de geometría: plantillas proyectadas", who="Subagente 'geometría' del workflow de Claude", date="2026-09-10",
                source=f"{ART}/agent_outputs/geometria/D_templates_projection.py, D_out.txt", status="AGENT")),
])

STATUS = OrderedDict([
    ("DOC-REPO", "Guardado en el repositorio; reproducido desde las features congeladas por Claude, el auditor de datos y la sesión de Codex."),
    ("DOC-PKT", "Documentado en el paquete para Esteban; no reejecutado en septiembre."),
    ("CLAUDE", "Cálculo exploratorio de Claude; script disponible; no revisado por una persona. Ajustó modelos pese a la regla de la fase uno."),
    ("AGENT", "Cálculo exploratorio de un subagente; script y salida disponibles; no verificado adversarialmente ni revisado por una persona."),
])

EARLY = OrderedDict([
    ("E_H5", "Primeras 5 rondas comunes de humans_only_absent.csv (filtro ausente→ausente). La 5ª llega a ronda ≥30 en 15/45 díadas."),
    ("E_Hk", "Primeras k rondas comunes de humans_only_absent.csv."),
    ("E_H≤r", "Rondas calendario ≤ r de humans_only_absent.csv."),
    ("E_H5-noov", "E_H5 excluyendo las 15 díadas cuya 5ª ronda es ≥30."),
    ("E_P5", "Primeras 5 oportunidades realmente ausentes de performances.csv; cortes en rondas 6–19. Es la cohorte reconstruida de TEMPORAL_REPAIR."),
    ("E_P≤r", "Todas las rondas ausentes con ronda ≤ r de performances.csv."),
])

TARGET = OrderedDict([
    ("T_STABLE", "stable_orientation de 06: mayoría de las ventanas anidadas 30–60, 40–60 y 50–60, con plantillas LR/TB sobre humans_only_absent. Positivo = LR o TB."),
    ("T_STABLE*", "Como T_STABLE, pero con umbrales de orientación o de tamaño de partición cambiados (ver notas)."),
    ("T_O4060", "Orientación por plantillas solo en la ventana 40–60 de humans_only_absent."),
    ("T_CAT40", "external_label_40_60 de 06: la unión de Category de ambos jugadores en rondas ≥40 contiene LEFT y RIGHT, o TOP y BOTTOM."),
    ("T_CAT41", "Igual que T_CAT40 con rondas ≥41 (check3.py, ext_label)."),
    ("T_CATFRAC", "Al menos la mitad de las filas de rondas ≥41 tienen Category axial (LEFT, RIGHT, TOP, BOTTOM)."),
    ("T_MODES", "Category modal de cada jugador en rondas ≥40: el par es LEFT/RIGHT o BOTTOM/TOP."),
    ("T_MODES_IO", "Como T_MODES, incluyendo el par IN/OUT."),
    ("T_SCORE", "Score medio en rondas presentes 41–60 de performances.csv mayor que la mediana."),
    ("T_ORIENT_H", "Orientación por plantillas en rondas 41–60 de humans_only_absent.csv; solo díadas con partición tardía de tamaño válido."),
    ("T_ORIENT_P", "Orientación por plantillas en todas las rondas ausentes 41–60 de performances.csv; solo díadas con partición tardía de tamaño válido."),
])

FEATURES = OrderedDict([
    ("G", "dominant_score: máxima |correlación| del margen temprano con plantillas lineales LR/TB. No usa el Laplaciano."),
    ("G_proj", "dominant_score con plantillas proyectadas al espacio de Fiedler."),
    ("G_eig", "||P_F m||/||m||: norma del margen proyectado al espacio de Fiedler, independiente de la base."),
    ("H", "early_h_obs: conductancia de la partición temprana."),
    ("S", "Similarity_early."), ("D", "DLIndex_early."), ("C", "Consistency_early."),
    ("TRIO", "D + S + C."), ("COMB", "G + D + S + C."), ("G+S", "G + S."),
])

# Columns: id, fam, early, target, pop, n, pos, features, kind, auc, note
R = []
def add(fam, early, target, pop, n, pos, feats, kind, auc, note=""):
    R.append(dict(fam=fam, early=early, target=target, pop=pop, n=n, pos=pos, features=feats, kind=kind, auc=auc, note=note))

# F1 historical
for pop, n, pos in [("all_dyads (13 INVALID=0)", 45, 20), ("valid_only", 32, 20)]:
    vals = {"all_dyads (13 INVALID=0)": [(0.834, 0.804), (0.808, 0.736), (0.890, 0.860)],
            "valid_only": [(0.825, 0.7792), (0.8583, 0.775), (0.9083, 0.8125)]}[pop]
    for feats, (ins, loo) in zip(["G", "TRIO", "COMB"], vals):
        add("F1", "E_H5", "T_STABLE", pop, n, pos, feats, "LOOCV", loo)
        add("F1", "E_H5", "T_STABLE", pop, n, pos, feats, "in-sample", ins)

# F2 Esteban packet
add("F2", "E_H≤20", "T_ORIENT_H", "partición tardía válida", 29, 21, "G", "LOOCV", 0.744, "Ventanas calendario 1–20 y 41–60, sin solapamiento.")
add("F2", "E_P≤20", "T_ORIENT_P", "partición tardía válida", 28, 21, "G", "LOOCV", 0.789, "Todas las ausentes ≤20 frente a 41–60.")
add("F2", "E_P5", "T_ORIENT_P", "partición tardía válida", 28, 21, "G", "LOOCV", 0.361,
    "Misma cohorte temprana que TEMPORAL_REPAIR. Un AUC LOOCV <0,5 con una sola variable indica ausencia de señal, no señal inversa.")
add("F2", "E_H5", "T_STABLE", "valid_only", 32, 20, "G_proj", "LOOCV", 0.7875, "reproduction_log.md; plantillas proyectadas.")

# F3 Claude check1/check2 (historical features)
for pop, n, vals in [("all_dyads (13 INVALID=0)", 45, dict(G=0.834, S=0.780, D=0.753, C=0.770)),
                     ("valid_only", 32, dict(G=0.825, S=0.8375, D=0.7667, C=0.8083))]:
    for f, v in vals.items():
        add("F3", "E_H5", "T_STABLE", pop, n, 20, f, "sin modelo", v)
for pop, n, vals in [("all_dyads (13 INVALID=0)", 45, dict(S=0.752, **{"G+S": 0.818}, H=0.772)),
                     ("valid_only", 32, dict(S=0.800, **{"G+S": 0.787}, H=0.771))]:
    for f, v in vals.items():
        add("F3", "E_H5", "T_STABLE", pop, n, 20, f, "LOOCV", v)

# F4 Claude check3
for feats, v, e in [("G", 0.853, "E_P≤20"), ("G", 0.622, "E_P≤10"), ("G", 0.836, "E_H5"), ("S", 0.793, "E_P≤20"), ("D", 0.700, "E_P≤20"),
                    ("C", 0.780, "E_P≤20"), ("TRIO", 0.784, "E_P≤20"), ("COMB", 0.880, "E_P≤20"), ("G+S", 0.871, "E_P≤20")]:
    add("F4", e, "T_CAT41", "todas", 45, 15, feats, "LOOCV", v,
        "Métricas oficiales tempranas de humans_only_absent ≤20." if feats in ("S", "D", "C", "TRIO", "COMB", "G+S") else "")
for feats, v in [("G", 0.853), ("S", 0.781), ("TRIO", 0.825), ("COMB", 0.899)]:
    add("F4", "E_P≤20", "T_CATFRAC", "todas", 45, 13, feats, "LOOCV", v)

# F5 data auditor: k sweep
ksweep = {1: (0.790, 0.820, 0.654, 0.742, 0), 2: (0.642, 0.698, 0.529, 0.654, 0), 3: (0.658, 0.710, 0.592, 0.675, 1),
          4: (0.686, 0.736, 0.646, 0.713, 6), 5: (0.804, 0.834, 0.779, 0.825, 15), 6: (0.844, 0.862, 0.825, 0.858, 18),
          8: (0.926, 0.936, 0.925, 0.937, 28), 10: (0.958, 0.974, 0.958, 0.979, 37), 15: (0.994, 0.994, 1.000, 1.000, 45),
          20: (0.996, 0.996, 1.000, 1.000, 45)}
for k, (la, ra, lv, rv, ov) in ksweep.items():
    note = f"k={k}; la k-ésima ronda llega a ≥30 en {ov}/45 díadas."
    add("F5", f"E_Hk (k={k})", "T_STABLE", "all_dyads (13 INVALID=0)", 45, 20, "G", "LOOCV", la, note)
    add("F5", f"E_Hk (k={k})", "T_STABLE", "all_dyads (13 INVALID=0)", 45, 20, "G", "sin modelo", ra, note)
    add("F5", f"E_Hk (k={k})", "T_STABLE", "valid_only", 32, 20, "G", "LOOCV", lv, note)
    add("F5", f"E_Hk (k={k})", "T_STABLE", "valid_only", 32, 20, "G", "sin modelo", rv, note)
add("F5", "E_H5-noov", "T_STABLE", "all_dyads sin solapadas", 30, 14, "G", "LOOCV", 0.875)
add("F5", "E_H5-noov", "T_STABLE", "all_dyads sin solapadas", 30, 14, "G", "sin modelo", 0.906)
add("F5", "E_H5-noov", "T_STABLE", "valid_only sin solapadas", 21, 14, "G", "LOOCV", 0.867)
add("F5", "E_H5-noov", "T_STABLE", "valid_only sin solapadas", 21, 14, "G", "sin modelo", 0.929)
for r, n_all, n_val, la, ra, lv, rv in [(10, 37, 26, 0.653, 0.732, 0.497, 0.667), (20, 45, 32, 0.824, 0.848, 0.800, 0.825), (29, 45, 32, 0.838, 0.860, 0.829, 0.850)]:
    add("F5", f"E_H≤{r}", "T_STABLE", "all_dyads con rondas", n_all, None, "G", "LOOCV", la)
    add("F5", f"E_H≤{r}", "T_STABLE", "all_dyads con rondas", n_all, None, "G", "sin modelo", ra)
    add("F5", f"E_H≤{r}", "T_STABLE", "valid_only con rondas", n_val, None, "G", "LOOCV", lv)
    add("F5", f"E_H≤{r}", "T_STABLE", "valid_only con rondas", n_val, None, "G", "sin modelo", rv)
outcomes = [
    ("T_CAT40", "todas", 45, 15, dict(G=0.856, S=0.827, D=0.733, C=0.818), dict(G=0.836, TRIO=0.767, COMB=0.858)),
    ("T_CAT40", "valid_only", 32, 15, dict(G=0.824, S=0.863, D=0.725, C=0.847), dict(G=0.796, TRIO=0.804, COMB=0.820)),
    ("T_O4060", "todas", 45, 23, dict(G=0.834, S=0.759, D=0.659, C=0.741), dict(G=0.800, TRIO=0.692, COMB=0.775)),
    ("T_MODES", "todas", 45, 11, dict(G=0.837, S=0.818, D=0.845, C=0.818), dict(G=0.816, TRIO=0.818, COMB=0.834)),
    ("T_MODES_IO", "todas", 45, 12, dict(G=0.846, S=0.795, D=0.831, C=0.783), dict(G=0.823, TRIO=0.783, COMB=0.848)),
    ("T_SCORE", "todas", 45, 22, dict(G=0.765, S=0.864, D=0.827, C=0.824), dict(G=0.745, TRIO=0.856, COMB=0.905)),
]
for tgt, pop, n, pos, raw, loo in outcomes:
    for f, v in raw.items():
        add("F5", "E_H5", tgt, pop, n, pos, f, "sin modelo", v)
    for f, v in loo.items():
        add("F5", "E_H5", tgt, pop, n, pos, f, "LOOCV", v)
add("F5", "E_H5", "T_STABLE*", "all_dyads", 45, 21, "G", "LOOCV", 0.784, "ORIENT_MIN_SCORE=0,20 con cualquier ratio. Con 0,25 o 0,30 no cambia: 0,804.")
add("F5", "E_H5", "T_STABLE*", "valid_only", 32, 21, "G", "LOOCV", 0.753, "ORIENT_MIN_SCORE=0,20.")
for lo, hi, a, v, nv in [(1, 63, 0.817, 0.807, 39), (5, 59, 0.817, 0.800, 36), (8, 56, 0.817, 0.800, 36), (12, 52, 0.804, 0.775, 30), (16, 48, 0.813, 0.788, 30), (20, 44, 0.804, 0.433, 23)]:
    add("F5", "E_H5", "T_STABLE*", "all_dyads", 45, None, "G", "LOOCV", a, f"Límites de tamaño de partición [{lo},{hi}] en lugar de [10,54].")
    add("F5", "E_H5", "T_STABLE*", "valid_only", nv, None, "G", "LOOCV", v, f"Límites de tamaño de partición [{lo},{hi}].")

# F6 geometry auditor
for pop, n, vals in [("all_dyads (13 INVALID=0)", 45, [("G_proj", 0.8060, 0.8340), ("G_eig", 0.8080, 0.8420)]),
                     ("valid_only", 32, [("G_proj", 0.7875, 0.8292), ("G_eig", 0.7875, 0.8417)])]:
    for f, loo, raw in vals:
        add("F6", "E_H5", "T_STABLE", pop, n, 20, f, "LOOCV", loo)
        add("F6", "E_H5", "T_STABLE", pop, n, 20, f, "sin modelo", raw)

for i, r in enumerate(R, 1):
    r["id"] = f"A{i:03d}"
    r["status"] = FAMILIES[r["fam"]]["status"]
    r["who"] = FAMILIES[r["fam"]]["who"]
    r["date"] = FAMILIES[r["fam"]]["date"]
    r["source"] = FAMILIES[r["fam"]]["source"]

cols = ["id", "fam", "date", "who", "status", "early", "target", "pop", "n", "pos", "features", "kind", "auc", "note", "source"]
with open(HERE / "explored_auc_register.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in R:
        w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})

# ---------- Markdown ----------
def fmt(v):
    return "—" if v is None or v == "" else (f"{v:.3f}".replace(".", ",") if isinstance(v, float) else str(v))

configs = {(r["early"].split(" (")[0] if r["fam"] != "F5" or not r["early"].startswith("E_Hk") else r["early"], r["target"], r["pop"]) for r in R}
geo_loocv = [r["auc"] for r in R if r["kind"] == "LOOCV" and r["features"] in ("G", "G_proj", "G_eig")]
lines = []
L = lines.append
L("# Registro de AUC exploradas: Spectral/SODCL")
L("")
L("Generado el 2026-09-11 por `scripts/build_auc_register.py` a partir de artefactos existentes. **No recalcula ni ajusta modelos.** Cada fila es un valor que alguien del proyecto o de sus auditorías ya vio. Se incluye aunque no se haya reportado, porque haberlo visto condiciona cualquier análisis posterior.")
L("")
L("Experimento y datos: Andrade-Lotero y Goldstone (2021), PLOS ONE 16(7): e0254532. Reanálisis: Thomas Chisica. Todas las AUC son exploratorias y retrospectivas; ninguna es confirmatoria.")
L("")
L("## Resumen")
L("")
L(f"- **{len(R)} valores de AUC** en **{len(FAMILIES)} familias** y **{len(configs)} combinaciones** distintas de ventana temprana, outcome y población.")
L(f"- Las AUC LOOCV de la señal geométrica sola van de **{fmt(min(geo_loocv))}** a **{fmt(max(geo_loocv))}**. Los valores por encima de 0,92 usan ventanas tempranas que ya contienen parte del outcome.")
L("- **La cohorte reconstruida ya fue evaluada.** Con las primeras cinco oportunidades ausentes reales (E_P5) y orientación tardía por plantillas, el paquete para Esteban obtuvo AUC 0,361 con n=28 (fila {}). Cualquier análisis futuro sobre esa cohorte no es ciego.".format(next(r["id"] for r in R if r["early"] == "E_P5")))
L("- **La AUC depende de cuánto se solapa la ventana temprana con el outcome.** En all_dyads, con 2 a 4 rondas del archivo filtrado cae a 0,64–0,69; con 8 o más supera 0,92 porque invade las ventanas del target.")
L("- **Con un outcome de desempeño (T_SCORE), las métricas oficiales superan a la geometría**: 0,856 frente a 0,745 en LOOCV.")
L("- **Plantillas lineales, proyectadas o norma en el espacio de Fiedler dan prácticamente la misma AUC**: difieren como mucho 0,01 en LOOCV y 0,017 sin modelo.")
L("- Ninguna comparación tiene un intervalo de incertidumbre con reajuste por díada ni agrupación por sesión.")
L("")
L("## Estado de las fuentes")
L("")
L("| Código | Significado |")
L("|---|---|")
for k, v in STATUS.items():
    L(f"| {k} | {v} |")
L("")
L("## Códigos")
L("")
L("**Ventana temprana**")
L("")
L("| Código | Definición |")
L("|---|---|")
for k, v in EARLY.items():
    L(f"| {k} | {v} |")
L("")
L("**Outcome tardío**")
L("")
L("| Código | Definición |")
L("|---|---|")
for k, v in TARGET.items():
    L(f"| {k} | {v} |")
L("")
L("**Predictores**")
L("")
L("| Código | Definición |")
L("|---|---|")
for k, v in FEATURES.items():
    L(f"| {k} | {v} |")
L("")
L("**Tipo de AUC.** LOOCV: regresión logística con imputación por mediana y estandarización, ajustada dejando fuera una díada cada vez. In-sample: el mismo modelo evaluado sobre sus datos de ajuste. Sin modelo: el predictor usado directamente como puntuación.")
L("")
for fam, meta in FAMILIES.items():
    rows = [r for r in R if r["fam"] == fam]
    L(f"## {fam}. {meta['title']}")
    L("")
    L(f"Autor: {meta['who']}. Fecha: {meta['date']}. Estado: {meta['status']}. Fuente: `{meta['source']}`.")
    L("")
    L("| ID | Ventana | Outcome | Población | n | Pos. | Predictores | Tipo | AUC | Nota |")
    L("|---|---|---|---|--:|--:|---|---|--:|---|")
    for r in rows:
        L(f"| {r['id']} | {r['early']} | {r['target']} | {r['pop']} | {fmt(r['n'])} | {fmt(r['pos'])} | {r['features']} | {r['kind']} | {fmt(r['auc'])} | {r['note']} |")
    L("")
L("## Resúmenes inferenciales ya vistos")
L("")
L("Calculados por Claude en `check2.py` sobre la cohorte histórica (E_H5, T_STABLE). El bootstrap remuestrea díadas sobre predicciones LOOCV fijas: 3000 réplicas, percentil, semilla 0. **No reajusta los modelos**, así que no incorpora la incertidumbre del ajuste.")
L("")
L("| Comparación | Población | Delta AUC | IC 95 % bootstrap |")
L("|---|---|--:|---|")
for comp, pop, d, ci in [("COMB − TRIO", "all_dyads, n=45", "+0,124", "[−0,004; +0,265]"), ("COMB − S", "all_dyads, n=45", "+0,108", "[−0,013; +0,243]"),
                         ("G − S", "all_dyads, n=45", "+0,052", "[−0,122; +0,229]"), ("G − H", "all_dyads, n=45", "+0,032", "[−0,101; +0,174]"),
                         ("COMB − TRIO", "valid_only, n=32", "+0,037", "[−0,075; +0,156]"), ("COMB − S", "valid_only, n=32", "+0,012", "[−0,096; +0,127]"),
                         ("G − S", "valid_only, n=32", "−0,021", "[−0,198; +0,163]"), ("G − H", "valid_only, n=32", "+0,008", "[−0,134; +0,152]")]:
    L(f"| {comp} | {pop} | {d} | {ci} |")
L("")
L("Nulo por permutación de etiquetas con reajuste LOOCV, 300 permutaciones, predictor G: media 0,283 en all_dyads y 0,306 en valid_only. Ninguna permutación alcanzó la AUC observada. La media muy por debajo de 0,5 ilustra el sesgo de LOOCV con una sola variable y muestras pequeñas.")
L("")
L("## Otros resultados predictivos vistos que no son AUC")
L("")
L("- `data/results/present_performance_increment.csv` (F1): con n=29, añadir h_obs a las métricas oficiales tardías baja el R² LOOCV del Score en rondas presentes de 0,654 a 0,497; añadir η, a 0,575.")
L("- `agent_outputs/datos/04_out.txt` (F5): correlaciones de Spearman entre predictores tempranos E_H5 y desempeño en 41–60. Por ejemplo, con Score presente y n=45: G 0,449; S 0,590; D 0,589; C 0,525.")
L("")
L("## Cómo usar este registro")
L("")
L("1. Antes de mirar un resultado nuevo, añadir una fila con ventana, outcome, población, predictores, validación y fecha, dejando la AUC vacía.")
L("2. Después de calcularlo, completar la AUC sin borrar ni editar filas anteriores.")
L("3. Al reportar cualquier AUC, citar su ID y el número total de combinaciones ya exploradas.")
L("4. Las cifras sin script, tabla por díada y procedimiento quedan como \"reportadas, no verificadas\".")
L("")
L("Tabla completa: `docs/explored_auc_register.csv`. Para regenerar ambos archivos: `python3 scripts/build_auc_register.py`.")
L("")
L(f"**Límite de trazabilidad.** Los scripts y salidas de F3 a F6, y la copia del paquete de F2, están fuera de este repositorio, en `{ART}` del VPS de Thomas. F1 es la única familia cuyos artefactos están versionados aquí.")
(HERE / "EXPLORED_AUC_REGISTER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(R)} filas; {len(configs)} configuraciones; geometría LOOCV {min(geo_loocv)}–{max(geo_loocv)}")
