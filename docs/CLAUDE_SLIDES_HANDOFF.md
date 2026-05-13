# Handoff para Claude: slides finales Beamer

## Objetivo

Construir y comparar **3 versiones de la presentación final** del proyecto de Teoría de Grafos, todas en Beamer o Beamer-compatible, sin inventar resultados y sin depender de paquetes que no existen en este TeX Live.

El objetivo práctico es escoger el mejor deck para la exposición en equipo. No rehacer el paper, no tocar datos, no cambiar claims científicos.

## Directorio de trabajo

Trabajar en:

```bash
/Users/thom/Desktop/grafos_proyecto
```

No trabajar en `~/Downloads` como fuente principal. Si generas notas temporales en Downloads, copia el resultado final al repo.

## Fuente de verdad

Usar estos archivos como canon:

- Paper final Codex: `paper/entrega_final_codex.tex`
- PDF final Codex: `dist/entrega_final_codex_documento.pdf`
- Slides actuales Codex, referencia sobria: `paper/presentacion_final_codex.tex`
- Slides actuales Claude, referencia visual: `paper/presentacion_final_claude.tex`
- Póster final Codex, referencia de jerarquía y claims: `paper/poster_final_codex.tex`

Figuras disponibles:

- `figures/fiedler_grid.png`
- `figures/fiedler_analysis.png`
- `figures/spectral_comparison_summary.png`
- `figures/temporal_dynamics.png`
- `figures/counterexample_P6xP8.png`
- `figures/early_prediction_summary.png`
- `figures/andrade_summary.png`
- `figures/partition_robustness_summary.png`
- `figures/entropy_analysis.png`

## Restricciones del entorno

Ya se verificó que este entorno no tiene:

- `latexmk`
- tema Beamer `metropolis`
- Fira Sans vía TeX

Por tanto:

- No dependas de `\usetheme{metropolis}`.
- No dependas de `latexmk`.
- No dependas de `fontspec` salvo que decidas compilar explícitamente con XeLaTeX y hayas probado que compila.
- Preferir una versión robusta con `pdflatex` y paquetes estándar.

Comandos de compilación aceptados:

```bash
mkdir -p build/final
cd paper
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/final presentacion_final_A_metropolis_custom.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/final presentacion_final_A_metropolis_custom.tex
```

Si usas XeLaTeX:

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory=../build/final archivo.tex
```

Después de compilar, copiar PDFs finales a `dist/`.

## Entregables esperados

Crear tres variantes, no una sola:

1. `paper/presentacion_final_A_metropolis_custom.tex`
   - Look tipo Metropolis, pero implementado manualmente.
   - Progress bar, frametitles limpios, section pages, mucho aire.
   - Debe compilar sin el paquete `metropolis`.

2. `paper/presentacion_final_B_ur_sober.tex`
   - Dirección URosario sobria.
   - Paleta crema/blanco, azul oscuro `#1A3A6E`, burgundy solo para alerta.
   - Más institucional y académico.

3. `paper/presentacion_final_C_dense_math.tex`
   - Variante más matemática/densa.
   - Menos “pitch deck”, más pizarra/paper.
   - Debe preservar legibilidad y no llenar cada slide de texto.

Exportar:

- `dist/presentacion_final_A_metropolis_custom.pdf`
- `dist/presentacion_final_B_ur_sober.pdf`
- `dist/presentacion_final_C_dense_math.pdf`

Crear también:

- `docs/SLIDES_COMPARISON.md`

Ese markdown debe comparar las 3 versiones con una tabla:

- Legibilidad en aula
- Rigor percibido
- Calidad visual
- Riesgo de compilación
- Fidelidad científica
- Recomendación final

## Estructura obligatoria del deck

12 slides, duración objetivo 10-12 minutos, exposición en equipo de 4 personas.

### Reparto sugerido

- Thomas: slides 1-3
- Juan Sebastián Mora: slides 4-6
- Ángel Amaya: slides 7-9
- Sara Figueredo Laserna: slides 10-12

### Slides

1. Portada
   - Título: `Ruptura de simetría y predicción temprana de especialización axial en Seeking the Unicorn`
   - Subtítulo: `Un reanálisis espectral del paradigma SODCL sobre el producto fuerte P_8 \boxtimes P_8`
   - Equipo: Thomas Chisica, Juan Sebastián Mora, Ángel Amaya, Sara Figueredo Laserna
   - Curso: Teoría de Grafos 2026-I, Universidad del Rosario
   - Profesor: Daniel Alfonso Bojacá Torres

2. Paradigma experimental
   - SODCL / Seeking the Unicorn
   - 45 díadas, 60 rondas
   - Dos jugadores exploran una grilla 8x8 sin comunicación explícita

3. Modelo de grafo
   - `G = P_8 \boxtimes P_8`
   - `|V| = 64`, `|E| = 210`
   - conectividad de rey
   - Figura sugerida: `figures/fiedler_grid.png`

4. Problema espectral
   - `lambda_2 = lambda_3 approx 0.4164`
   - El eigenspace de Fiedler es bidimensional
   - Un único vector de Fiedler no es baseline canónico

5. Baseline correcto
   - `u_theta = cos(theta) v_2 + sin(theta) v_3`
   - `S_theta` como familia de cortes
   - Fiedler ingenuo: `h = 28/210 approx 0.1333`
   - Axis-aligned: `h = 22/210 approx 0.1048`
   - Cuidado: no afirmar óptimo global de conductancia si el paper no lo prueba

6. Señal geométrica temprana
   - margen de visitas `m_d(v) = f_1(v) - f_2(v)`
   - `g_d = max{|<m~, a_LR>|, |<m~, a_TB>|}`
   - primeras 5 rondas ausentes comunes

7. Resultado 1: axial vs mixta
   - Figura: `figures/spectral_comparison_summary.png`
   - Tabla mínima:
     - Axial n=21, mixta n=8
     - `h(S_obs)`: 0.142 vs 0.714
     - `eta`: 1.120 vs 0.196
     - `DLIndex`: 0.919 vs 0.514
     - MI: 0.839 vs 0.260
     - JSD: 0.842 vs 0.284
   - Mann-Whitney para conductancia: `p = 3.07e-5`

8. Resultado 2: dinámica/robustez
   - Figura: `figures/temporal_dynamics.png` o `figures/partition_robustness_summary.png`
   - Ventana 40-60:
     - `h(S_obs)`: axial 0.119, mixta 0.701
     - especialización suave: axial 0.906, mixta 0.492
     - Jaccard inter-ventana: axial 0.934, mixta 0.651

9. Resultado 3: contraejemplo rectangular
   - Figura: `figures/counterexample_P6xP8.png`
   - Cuadrado: `lambda_2 = lambda_3 approx 0.4164`
   - Rectangular `P_6 \boxtimes P_8`: `lambda_2 = 0.4041`, `lambda_3 = 0.7288`, gap `0.3247`
   - Claim: la degeneración es estructural, no artefacto numérico

10. Resultado 4: predicción temprana
    - Figura: `figures/early_prediction_summary.png`
    - LOOCV AUC:
      - geometría `g_d`: 0.804
      - métricas oficiales tempranas: 0.736
      - combinado: 0.860
    - Subset válido n=32: combinado 0.8125

11. Transferencia y límites
    - Axiales en rondas presentes: accuracy aprox. 0.936, score aprox. 24.43
    - Mixtas en rondas presentes: accuracy aprox. 0.688, score aprox. -6.14
    - Unidad experimental: díada; transferencia descriptiva, no causal
    - Novelty gate: `Spearman rho(eta, DLIndex) = 0.894`, `p = 6.23e-11`
    - Claim correcto: la geometría temprana predice cristalización; conductancia tardía no reemplaza métricas oficiales

12. Conclusiones y preguntas
    - Degeneración obliga a baseline symmetry-aware
    - `g_d` anticipa especialización axial estable
    - Contraejemplo rectangular muestra que cambiar geometría cambia el problema espectral
    - Siguiente paso: tableros rectangulares u obstáculos para romper simetría experimentalmente

## Reglas científicas duras

No escribir:

- “los humanos vencen a Fiedler”
- “los humanos optimizan espectralmente”
- “esto explica todas las estrategias”
- “causalidad” a partir de datos observacionales
- p-values por ronda como si las rondas fueran sujetos independientes

Sí escribir:

- “reanálisis espectral”
- “baseline sensible a degeneración”
- “especialización axial”
- “señal temprana interpretable”
- “transferencia descriptiva”
- “limitación: modelo bipartito axial no cubre ALL/NOTHING/IN/OUT igual de bien”

## Reglas visuales

- No estética startup.
- No gradientes fuertes.
- No iconos decorativos.
- No stock illustrations.
- Figuras grandes, texto corto.
- Máximo 3 bullets por slide, salvo tablas.
- Si una figura es central, debe ocupar al menos 55-65% del slide.
- Cada slide debe tener una frase interpretativa, no solo una gráfica.

Paleta recomendada:

- Fondo claro: `#FAFAF8` o blanco
- Tinta: `#16161A`
- Gris: `#5A5A5E`
- Azul: `#1A3A6E`
- Burgundy de alerta: `#7A1F2B`, usar poco

## Comparación final esperada

Después de crear las tres versiones, no decidas solo por gusto visual. Renderiza a PNG o revisa PDFs y evalúa:

- ¿Se lee desde el fondo del salón?
- ¿Cada slide tiene un único punto?
- ¿Las figuras no quedan comprimidas?
- ¿El deck defiende la rúbrica del curso?
- ¿La historia científica evita claims inflados?
- ¿Compila en este Mac sin instalar paquetes nuevos?

La recomendación final debe ser pragmática: escoger una versión principal y, si vale la pena, tomar 1-2 slides de otra.
