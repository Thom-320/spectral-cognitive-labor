# Biblioteca de lectura del proyecto

Índice de los 20 artículos consultados para decidir la dirección científica. **Los PDF no se redistribuyen en este repositorio**: son copias de lectura personal que viven fuera de él. Cada entrada enlaza al DOI o a la fuente abierta desde la que se descargó.

La procedencia detallada, con URL, SHA-256, páginas y fecha de descarga, está en [`paper_catalog.json`](paper_catalog.json): 20 entradas verificadas el 2026-09-10. Para comprobar una copia local contra ese catálogo:

```bash
python scripts/verify_literature_library.py --dir RUTA_DE_LA_CARPETA
```

Los informes que usan esta biblioteca son la [revisión inicial](../LITERATURE_REVIEW_2026-09-10.md) y la [ampliación con evaluación de auditorías externas](../LITERATURE_EXPANSION_AND_AUDIT_REVIEW.md).

## Lectura prioritaria

Goldstone y colaboradores para definir el fenómeno, Correa y colaboradores para justificar la comparación de teorías, Niehorster y colaboradores para limitar inferencias mecanísticas, y Vargas Bernal y colaboradores para dinámica y comunidades.

## Índice

| Referencia | Artículo | Páginas | Papel en el proyecto |
|---|---|--:|---|
| Andrade-Lotero y Goldstone, 2021 | [Self-organized division of cognitive labor](https://doi.org/10.1371/journal.pone.0254532) | 22 | Estudio original: define la tarea, las regiones focales y los modelos MBiases, WSLS y FRA. |
| Goldstone et al., 2024 (en línea 2023) | [The Emergence of Specialized Roles Within Groups](https://doi.org/10.1111/tops.12644) | 25 | Marco posterior sobre emergencia de roles; sitúa esta tarea junto a otros paradigmas. |
| Roberts y Goldstone, 2011 | [Adaptive Group Coordination and Role Differentiation](https://doi.org/10.1371/journal.pone.0022377) | 8 | Diferenciación de roles sin territorios espaciales. |
| Hawkins y Goldstone, 2016 | [The Formation of Social Conventions in Real-Time Environments](https://doi.org/10.1371/journal.pone.0151670) | 14 | Convenciones: eficiencia, equidad y estabilidad no van necesariamente juntas. |
| Niehorster et al., 2019 (en línea 2018) | [Searching with and against each other: Spatiotemporal coordination of visual search behavior in collaborative and competitive settings](https://doi.org/10.3758/s13414-018-01640-0) | 18 | Territorios estables y barridos desde extremos opuestos producen mapas parecidos. |
| Wahn et al., 2020 | [Dyadic and triadic search: Benefits, costs, and predictors of group performance](https://doi.org/10.3758/s13414-019-01915-0) | 19 | Cobertura y solapamiento como medidas de reparto; costes de coordinación. |
| Wahn y Schmitz, 2023 (en línea 2022) | [Labor division in collaborative visual search: a review](https://doi.org/10.1007/s00426-022-01767-8) | 11 | Revisión del reparto en búsqueda visual conjunta; ruta hacia estudios primarios. |
| Correa et al., 2023 | [Humans decompose tasks by trading off utility and computational cost](https://doi.org/10.1371/journal.pcbi.1011087) | 31 | Comparación de teorías de descomposición, incluida una espectral; referencia metodológica para diseñar desacuerdos. |
| Solway et al., 2014 | [Optimal Behavioral Hierarchy](https://doi.org/10.1371/journal.pcbi.1003779) | 10 | Jerarquías óptimas por economía de representación. |
| Şimşek et al., 2005 | [Identifying useful subgoals in reinforcement learning by local graph partitioning](https://doi.org/10.1145/1102351.1102454) | 8 | Subobjetivos por partición de grafos construidos con transiciones. |
| Mahadevan y Maggioni, 2007 | [Proto-value Functions: A Laplacian Framework for Learning Representation and Control in Markov Decision Processes](https://www.jmlr.org/papers/volume8/mahadevan07a/mahadevan07a.pdf) | 63 | Bases laplacianas para representación y control. |
| Stachenfeld et al., 2017 | [The hippocampus as a predictive map](https://doi.org/10.1038/nn.4650) | 13 | Representación sucesora: la representación depende del proceso, no solo del espacio. |
| Vargas Bernal, Porter y Tien, 2024 | [Adapting InfoMap to Absorbing Random Walks Using Absorption-Scaled Graphs](https://doi.org/10.1137/21M1466803) | 36 | Comunidades bajo caminatas absorbentes; la dinámica declarada cambia la partición detectada. |
| Naito et al., 2022 | [Insights about the common generative rule underlying an information foraging task can be facilitated via collective search](https://doi.org/10.1038/s41598-022-12126-3) | 12 | Búsqueda colectiva e inferencia de la regla generadora. |
| Wahn y Kingstone, 2020 | [Labor division in joint tasks: Humans maximize use of their individual attentional capacities](https://doi.org/10.3758/s13414-020-02012-3) | 11 | La orientación del reparto depende de la pantalla y de la capacidad atencional; limita una manipulación de geometría. |
| Malcolmson et al., 2007 | [Collaboration during visual search](https://doi.org/10.3758/BF03196825) | 6 | Sensibilidad frente a criterio de respuesta en parejas colaborativas. |
| Tomov et al., 2020 | [Discovery of hierarchical representations for efficient planning](https://doi.org/10.1371/journal.pcbi.1007594) | 42 | La topología, la tarea y las recompensas determinan la jerarquía descubierta. |
| Wu et al., 2021 (en línea 2020) | [Inference and Search on Graph-Structured Spaces](https://doi.org/10.1007/s42113-020-00091-x) | 23 | Inferencia y búsqueda humana con kernel de difusión sobre un grafo. |
| Wu et al., 2021 (CogSci) | [Specialization and selective social attention establishes the balance between individual and social learning](https://charleywu.github.io/downloads/wu2021specialization.pdf) | 7 | Especialización y atención social selectiva en búsqueda colectiva. |
| Cazenille et al., 2024 (preprint) | [Hearing the shape of an arena with spectral swarm robotics](https://arxiv.org/pdf/2403.17147) | 43 | Antecedente explícitamente espectral en colectivos robóticos. Preprint de arXiv. |

## Límites

La selección no es exhaustiva ni sistemática, y no establece novedad por ausencia de coincidencias. Cazenille y colaboradores es un preprint de arXiv. El trabajo de Wu y colaboradores sobre especialización es de actas de congreso, no una publicación independiente de su preprint. Goldstone y colaboradores conserva la paginación de la versión en línea de 2023 de un artículo incluido en el volumen de 2024.

Quedaron fuera del paquete, con DOI registrado en los informes: Brennan y colaboradores 2008, Schapiro y colaboradores 2013, Richardson y colaboradores 2015, el estudio de doubles-pong de 2018 y el trabajo de 2021 sobre coordinación y pupila.

