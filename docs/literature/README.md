# Biblioteca de lectura del proyecto

Índice de los 24 artículos que sostienen la dirección científica. **Los PDF no se redistribuyen en este repositorio**: son copias de lectura personal que viven fuera de él. Cada entrada enlaza a su DOI o a la fuente abierta desde la que se descargó.

La procedencia detallada, con URL, SHA-256, páginas y fecha, está en [`paper_catalog.json`](paper_catalog.json). Para comprobar una copia local:

```bash
python scripts/verify_literature_library.py --dir RUTA_DE_LA_CARPETA
```

Informes que usan esta biblioteca: [revisión inicial](../LITERATURE_REVIEW_2026-09-10.md) y [ampliación con evaluación de auditorías externas](../LITERATURE_EXPANSION_AND_AUDIT_REVIEW.md). La dirección actual está en [`MECHANISM_PROPOSAL.md`](../MECHANISM_PROPOSAL.md).

## Prioridad

**S0** es la mesa de trabajo: define el fenómeno, los mecanismos rivales, el estado latente bajo observación parcial y cómo comparar teorías. **S1** acota interpretaciones y aporta antecedentes directos. **S2** es fundamento matemático y control de novedad para la parte espectral, que ya no es el centro de la pregunta.


## S0 · mesa de trabajo

| Referencia | Artículo | Páginas | Para qué |
|---|---|--:|---|
| Andrade-Lotero y Goldstone, 2021 | [Self-organized division of cognitive labor](https://doi.org/10.1371/journal.pone.0254532) | 22 | Fuente primaria: tarea, regiones focales, modelos MBiases, WSLS y FRA, shaky hand y comparación por AIC. Todo claim nuevo se compara contra esto. |
| Goldstone et al., 2024 (en línea 2023) | [The Emergence of Specialized Roles Within Groups](https://doi.org/10.1111/tops.12644) | 25 | Marco CARMI de los propios autores: la especialización es complementar lo que se espera del otro, no solo ocupar regiones distintas. |
| Wahn y Schmitz, 2023 (en línea 2022) | [Labor division in collaborative visual search: a review](https://doi.org/10.1007/s00426-022-01767-8) | 11 | Mapa del subcampo: reparto espacial, información compartida, costes de coordinación y problemas abiertos. |
| Niehorster et al., 2019 (en línea 2018) | [Searching with and against each other: Spatiotemporal coordination of visual search behavior in collaborative and competitive settings](https://doi.org/10.3758/s13414-018-01640-0) | 18 | Un mapa espacial agregado no identifica la política que lo generó; usa solapamiento y predictibilidad temporal. |
| Hawkins y Goldstone, 2016 | [The Formation of Social Conventions in Real-Time Environments](https://doi.org/10.1371/journal.pone.0151670) | 14 | Estabilidad, equidad y eficiencia de una convención no van juntas; relevante para definir qué persiste. |
| De Vicariis et al., 2024 | [Computational joint action: Dynamical models to understand the development of joint coordination](https://doi.org/10.1371/journal.pcbi.1011948) | 27 | Modelo probabilístico del compañero bajo observación imperfecta: separa acción observada de creencia sobre el compañero. |
| Li, Henning y Camerer, 2023 | [Estimating Hidden Markov Models (HMMs) of the cognitive process in strategic thinking using eye-tracking](https://doi.org/10.3389/frbhe.2023.1225856) | 18 | Cadena oculta con observaciones indirectas y predicción cuando el proceso se trunca por presión temporal. |
| Deffner et al., 2024 | [Collective incentives reduce over-exploitation of social information in unconstrained human groups](https://doi.org/10.1038/s41467-024-47010-3) | 13 | Modelo de decisión social con estados ocultos sobre trayectorias espaciales en búsqueda colectiva. |
| Correa et al., 2023 | [Humans decompose tasks by trading off utility and computational cost](https://doi.org/10.1371/journal.pcbi.1011087) | 31 | Cómo comparar teorías donde de verdad discrepan; incluye una alternativa espectral y comparadores simples. |

## S1 · antecedentes y límites

| Referencia | Artículo | Páginas | Para qué |
|---|---|--:|---|
| Wahn y Kingstone, 2020 | [Labor division in joint tasks: Humans maximize use of their individual attentional capacities](https://doi.org/10.3758/s13414-020-02012-3) | 11 | La orientación preferida depende de la pantalla y de la capacidad atencional; limita interpretaciones espectrales de LR y TB. |
| Roberts y Goldstone, 2011 | [Adaptive Group Coordination and Role Differentiation](https://doi.org/10.1371/journal.pone.0022377) | 8 | Diferenciación funcional de roles sin territorios espaciales. |
| Malcolmson et al., 2007 | [Collaboration during visual search](https://doi.org/10.3758/BF03196825) | 6 | Antecedente directo: 19 de 24 parejas parten la pantalla en izquierda y derecha. |
| Guennouni y Speekenbrink, 2022 | [Transfer of Learned Opponent Models in Zero Sum Games](https://doi.org/10.1007/s42113-022-00133-6) | 17 | Cadenas ocultas para cambio de estrategia y modelos del oponente; útil para el diseño y la recuperación de modelos. |
| Wahn et al., 2020 | [Dyadic and triadic search: Benefits, costs, and predictors of group performance](https://doi.org/10.3758/s13414-019-01915-0) | 19 | Cobertura y solapamiento como medidas de reparto; costes de coordinación. |

## S2 · fundamento y control de novedad

| Referencia | Artículo | Páginas | Para qué |
|---|---|--:|---|
| Naito et al., 2022 | [Insights about the common generative rule underlying an information foraging task can be facilitated via collective search](https://doi.org/10.1038/s41598-022-12126-3) | 12 | Búsqueda colectiva e inferencia de la regla generadora. |
| Wu et al., 2021 (CogSci) | [Specialization and selective social attention establishes the balance between individual and social learning](https://charleywu.github.io/downloads/wu2021specialization.pdf) | 7 | Especialización y atención social selectiva en búsqueda colectiva. |
| Wu et al., 2021 (en línea 2020) | [Inference and Search on Graph-Structured Spaces](https://doi.org/10.1007/s42113-020-00091-x) | 23 | Inferencia y búsqueda humana con kernel de difusión sobre un grafo. |
| Mahadevan y Maggioni, 2007 | [Proto-value Functions: A Laplacian Framework for Learning Representation and Control in Markov Decision Processes](https://www.jmlr.org/papers/volume8/mahadevan07a/mahadevan07a.pdf) | 63 | El grafo útil viene de la estructura de transición, no de la proximidad en pantalla. |
| Stachenfeld et al., 2017 | [The hippocampus as a predictive map](https://doi.org/10.1038/nn.4650) | 13 | La representación depende de las transiciones y de la política, no solo del espacio. |
| Tomov et al., 2020 | [Discovery of hierarchical representations for efficient planning](https://doi.org/10.1371/journal.pcbi.1007594) | 42 | Topología, tarea y recompensas cambian la jerarquía inferida. |
| Solway et al., 2014 | [Optimal Behavioral Hierarchy](https://doi.org/10.1371/journal.pcbi.1003779) | 10 | Jerarquías óptimas por economía de representación. |
| Şimşek et al., 2005 | [Identifying useful subgoals in reinforcement learning by local graph partitioning](https://doi.org/10.1145/1102351.1102454) | 8 | Subobjetivos por partición de grafos construidos con transiciones. |
| Vargas Bernal, Porter y Tien, 2024 | [Adapting InfoMap to Absorbing Random Walks Using Absorption-Scaled Graphs](https://doi.org/10.1137/21M1466803) | 36 | Comunidades bajo caminatas absorbentes: la dinámica declarada cambia la partición detectada. |
| Cazenille et al., 2024 (preprint) | [Hearing the shape of an arena with spectral swarm robotics](https://arxiv.org/pdf/2403.17147) | 43 | Laplaciano y colectivos no es combinación inédita; control de novedad para la parte espectral. |

## Pendiente

**Brennan et al. (2008), Coordinating cognition: the costs and benefits of shared gaze during collaborative search**, [doi:10.1016/j.cognition.2007.05.012](https://doi.org/10.1016/j.cognition.2007.05.012). Antecedente clásico sobre qué información del compañero permite coordinar la búsqueda. Es de acceso cerrado y OpenAlex no registra copia abierta; requiere acceso institucional.

## Límites

La selección no es exhaustiva ni sistemática, y no establece novedad por ausencia de coincidencias. Cazenille y colaboradores es un preprint de arXiv. El trabajo de Wu y colaboradores sobre especialización es de actas de congreso, no una publicación independiente de su preprint. Goldstone y colaboradores, Niehorster y colaboradores, Wahn y Schmitz, y Wu y colaboradores sobre inferencia en grafos tienen año de volumen distinto del de publicación en línea; la tabla indica ambos.

