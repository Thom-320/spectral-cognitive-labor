# Ampliación de literatura y revisión de afirmaciones externas

2026-09-10. Revisión dirigida adicional, no revisión sistemática. No se entrenaron modelos nuevos. Se preservan datos, resultados y scripts históricos.

## Corrección de alcance

La selección inicial de 13 artículos no justificaba hablar de escasez. La búsqueda ampliada encuentra una literatura sustancial en búsqueda visual conjunta, seguimiento de objetos, coordinación complementaria, aprendizaje social y representaciones de grafos. Es más difícil localizar una coincidencia exacta con energía de Fiedler frente a coordenadas en SODCL, pero esa ausencia no establece novedad. Hay antecedentes que limitan propuestas experimentales aparentemente nuevas.

## Nuevos antecedentes prioritarios

| Fuente primaria | Qué añade | Lectura y límite |
| --- | --- | --- |
| Wahn y Kingstone (2020), [Labor division in joint tasks](https://doi.org/10.3758/s13414-020-02012-3) | Manipula disposición portrait/landscape y posición de participantes; la preferencia y ventaja de un reparto TB cambian con la pantalla. Una propuesta de variar geometría debe contrastarse con este antecedente. | PDF del editor descargado, resumen y secciones pertinentes consultados. Es seguimiento de objetos, no SODCL. |
| Malcolmson, Reynolds y Smilek (2007), [Collaboration during visual search](https://doi.org/10.3758/BF03196825) | Distingue sensibilidad perceptiva y criterio de respuesta en parejas colaborativas frente a nominales. Rendimiento no se reduce a rapidez ni axialidad. | PDF de seis páginas verificado; resumen consultado. No confundir con la tesis de 2006 que aparece bajo el mismo título en algunos buscadores. |
| [Division of labor… doubles-pong](https://doi.org/10.1016/j.humov.2017.11.012), 2018 | Variar posiciones iniciales permite contrastar cuentas predictivas y coordinación emergente del reparto espacial. | Resumen del editor consultado. CRAI Crossref/OpenAlex identifica copias en RUG/HAL; texto no descargado en este paquete. No adjudicar mecanismo de SODCL por analogía. |
| Richardson et al. (2015), [Self-organized complementary joint action](https://doi.org/10.1037/xhp0000041) | Complementariedad en evitación de colisiones y dinámica interpersonal, más allá de sincronía. | Resumen y metadatos localizados; acceso PMC dio desafío. Texto completo pendiente. |
| Wu, Schulz y Gershman (2021; online 2020), [Inference and Search on Graph-Structured Spaces](https://doi.org/10.1007/s42113-020-00091-x) | Modelo de inferencia/búsqueda con kernel de difusión K=exp(-alpha L). Conecta un operador laplaciano con predicciones conductuales explícitas. | PDF de autor descargado; formulación y secciones pertinentes consultadas. Cognición individual, no especialización diádica; no es la misma variable que E_F. |
| Tomov et al. (2020), [Discovery of hierarchical representations for efficient planning](https://doi.org/10.1371/journal.pcbi.1007594) | Jerarquías influidas por topología, tareas y recompensas. Permite cuestionar que la geometría sola determine una partición relevante. | Texto/PDF abierto; formulación y resultados generales consultados, no reproducción de ocho experimentos. |
| Wu et al. (2021), [Specialization and selective social attention](https://charleywu.github.io/downloads/wu2021specialization.pdf) | Especialización entre explorar e imitar en búsqueda colectiva, con atención y trayectorias. | Artículo de CogSci, siete páginas; no contar su preprint como estudio independiente. No afirmar transferencia de datos a SODCL. |
| Cazenille et al., [Hearing the shape of an arena with spectral swarm robotics](https://arxiv.org/abs/2403.17147) | Antecedente explícitamente espectral en colectivos robóticos. | PDF de arXiv descargado; resumen consultado. Clasificación de forma de arena no demuestra división cognitiva humana. Versión descargada rotulada como preprint; fecha de primera publicación 2024. |
| Wahn et al. (2021), [Coordination effort in joint action is reflected in pupil size](https://doi.org/10.1016/j.actpsy.2021.103291) | Diferencia coordinar un reparto de recibir uno predeterminado; coste cognitivo potencialmente medible. | Resumen primario en PubMed; no PDF en el paquete. No justifica añadir pupila a datos que no la contienen. |

También se añadió al paquete la revisión [Wahn y Schmitz](https://doi.org/10.1007/s00426-022-01767-8) y el artículo de [Vargas Bernal, Porter y Tien](https://doi.org/10.1137/21M1466803). La revisión sirve para rastrear estudios primarios; el artículo de Esteban fundamenta su pertinencia matemática, no una obligación de usar InfoMap.

**Implicación propuesta:** mantener el contraste pequeño como evaluación de representación. Para una futura manipulación, especificar qué predice una hipótesis espectral que no predigan capacidades atencionales, coordenadas o estrategias de barrido. Cambiar la forma del tablero sin esa disociación sería una extensión de antecedentes conocidos, no evidencia automática de mecanismo nuevo.

## Evaluación del texto de auditoría pegado

| Afirmación | Estado en esta revisión |
| --- | --- |
| Filtrado absent→absent; solapamiento en 15 díadas; métricas y proyectores | Coherente con los artefactos existentes. Se ejecutaron nuevamente seis pruebas de integridad y se comprobaron los 46 hashes históricos, contenidos del archivo y seis salidas antes de editar documentación. No se recalcularon todas las AUC. |
| Dependencia de base del corte por un autovector de Fiedler | Fundamento matemático correcto y ya reconocido en la reparación. El intervalo exacto 22–28 aristas del barrido angular citado no se reprodujo aquí. No extender sin revisión la conclusión a todo indicador del repositorio. |
| n=32 como «muestra limpia» | Etiqueta insuficiente: se necesita lista de díadas/exclusiones. Eliminar INVALID no demuestra separación temporal ni independencia; puede seleccionar la población. |
| Nuevos IC bootstrap y AUC 0.853/0.784/0.880 | Reportados por otro flujo, no verificados: no se localizaron script, etiquetas, particiones, predicciones OOF y procedimiento exacto. No incorporados a data/results. |
| Rondas <=20 frente a resultado 41–60 | Puede evitar solapamiento directo si TODAS las variables usan únicamente ese prefijo. Cambia el estimando: tiempo total fijo y distinto número de oportunidades, frente a cinco oportunidades ausentes. No reemplaza silenciosamente el protocolo anterior. |
| «Outcome externo del paper» | Una etiqueta sobre las mismas personas no es validación externa. Comprobar si se construye realmente con 41–60 o si la clasificación publicada agrega otras rondas, incluida la ventana temprana. Tampoco un mapa agregado demuestra persistencia. |
| «Sin tocar el repo» / «solo lectura» | Puede describir preservación de archivos, pero ajustar modelos en scratch sigue siendo realizar análisis nuevos y decisiones analíticas. No equivale a no entrenar modelos. |
| CITATION.cff y STATUS.md incorrectos | Confirmado. Se corrigieron autor/año/DOI del estudio y el estado científico, retirando atribuciones y planes de transferencia no verificados. |
| Número de agentes, tokens y colores del workflow | La captura muestra actividad; no prueba independencia, finalización ni corrección científica. Los informes y artefactos finales siguen siendo necesarios. No se ejecutó ese workflow desde esta revisión. |

Antes de aceptar las cifras nuevas solicitar: script y comando exactos; versión Git/dependencias/semillas; tabla por díada con predictores, outcome, ventana y exclusiones; predicciones fuera de muestra; preprocesamiento dentro de cada fold; bootstrap pareado con unidad de díada y aclaración de si reajusta modelos. Documentar elecciones posteriores a ver resultados. No volver a entrenar para adivinar una receta que reproduzca números redondeados.

## Rutas de búsqueda y archivos

Búsqueda web por familias, títulos y autores; referencias de Wahn/Schmitz; fuentes editoriales Springer/PLOS, páginas de autores Wu/Hawkins/Gershman/Porter, JMLR, ICML y arXiv. Consultas adicionales: `human collective search spontaneous division labor coordination spatial complementary roles experiment`; `joint action symmetry breaking complementary coordination Richardson 2015`; `Wahn Kingstone 2020 division screen`; `Inference and search on graph-structured spaces`; `spectral division of labor cognitive human`.

CRAI: búsqueda institucional Springer `joint action division labor` devolvió cinco resultados poco específicos; no se interpretó como censo de literatura. Su resolvedor OA para 10.1016/j.humov.2017.11.012 confirmó Crossref/OpenAlex y rutas RUG/HAL; Unpaywall no configurado. Las aperturas web de esas rutas fallaron. No se concluye inaccesibilidad definitiva. No se enviaron solicitudes, correos ni credenciales.

Los PDFs están en Downloads/Spectral_SODCL_papers, fuera de Git, para lectura personal. Cada descarga tiene URL, hash, número de páginas y una comprobación textual en download_manifest.json. Que un PDF sea válido no implica lectura exhaustiva ni reproducción de su estudio. El índice bibliográfico y manifiesto resumido sí pueden versionarse; no se redistribuyen PDFs de editoriales en el repositorio.

Las búsquedas más amplias aún no son sistemáticas ni exhaustivas. El prompt independiente pide ampliar por citas y familias, no completar una cuota de referencias ni afirmar ausencia de antecedentes a partir de esta muestra.
