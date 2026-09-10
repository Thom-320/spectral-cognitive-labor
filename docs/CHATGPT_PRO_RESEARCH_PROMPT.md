# Prompt para una investigación independiente en ChatGPT Pro

Copia el texto siguiente en una conversación con búsqueda/investigación disponible. Adjunta el repositorio como ZIP si ChatGPT no puede abrir sus archivos en GitHub, y los informes indicados al final. No presupongas que tiene acceso a rutas de mi Mac, CRAI o conversaciones anteriores.

---

Quiero una investigación crítica, amplia y basada en fuentes primarias para decidir el siguiente trabajo científico de mi proyecto Spectral/SODCL. Investiga; no te limites a aprobar el plan ni a redactar un roadmap. Responde en español.

Soy Thomas Chisica. Mi contribución es el reanálisis espectral y computacional del experimento de Edgar Andrade-Lotero y Robert L. Goldstone, “Self-organized division of cognitive labor”, PLOS ONE (2021), DOI 10.1371/journal.pone.0254532. No me atribuyas diseño experimental, recolección ni propiedad de los datos. Verifica atribución y procedencia en las fuentes originales.

Repositorio público: https://github.com/Thom-320/spectral-cognitive-labor

## Objetivo

¿Qué aporta una representación espectral respecto de las medidas conductuales originales y de geometría sencilla? Distingue descripción matemática útil, predicción incremental y explicación mecanística con predicciones diferenciables. No uses evidencia de un nivel como prueba del siguiente.

La investigación anterior seleccionó pocos artículos y NO demuestra escasez de literatura. Tu trabajo incluye intentar refutar la novedad propuesta mediante búsqueda amplia, sin inflar la bibliografía con coincidencias de palabras como espectro de EEG o división funcional del cerebro.

## Estado documentado que debes auditar, no aceptar ciegamente

- El conjunto fuente tiene 45 díadas. El análisis descriptivo de 29 díadas no es necesariamente la población de las AUC.
- La selección histórica usa un archivo de 1.244 filas que coincide con rondas ausentes seguidas de otra ausente, frente a 2.644 filas ausentes en performances.csv. En 15/45 díadas las cinco oportunidades seleccionadas históricamente llegan a ronda >=30.
- Las AUC históricas 0.804/0.860 no deben presentarse como predicción temprana validada. El predictor dominant_score usa coordenadas, no autovectores.
- Una reconstrucción separada usa las primeras cinco oportunidades realmente ausentes; cortes entre rondas 6 y 19. Reconstruye DLIndex, Similarity y Consistency siguiendo las definiciones originales; Consistency se refiere a la oportunidad ausente anterior. Hay seis pruebas de integridad, no una validación científica independiente completa.
- Grafo espacial 8x8 de ocho vecinos, Laplaciano combinatorio D-A, 64 vértices y 210 aristas. Es hipótesis de proximidad; no se debe identificar sin argumento con restricciones de movimiento o clics.
- El espacio de Fiedler es de dimensión dos. Un corte por signo de un solo autovector puede depender de la base. Una referencia axial favorable dentro de una familia no es óptimo global.
- Se comparan proyectores ortogonales de rango dos P_F y P_xy en el mismo espacio euclídeo R^64. E(m)=m^T P m/(m^T m). El margen cero necesita una convención explícita.
- Ángulos principales documentados: aproximadamente 6.59046 grados. Correlación de energías en las 45 díadas: 0.999486. Son distintas pero muy próximas; esto no prueba equivalencia predictiva. Energía alta no significa concentración en un solo eje.
- No existe todavía un outcome acordado con Andrade ni una nueva evaluación predictiva aprobada. Propuesta pendiente: axialidad sostenida en rondas 40–60, predicha tras cinco oportunidades ausentes. Distinguir eje persistente, territorio individual persistente y alternancia predecible; no axial válido frente a no clasificable.

## Otra auditoría: afirmaciones pendientes, no hechos confirmados

Recibí un texto que afirma haber calculado en scratch:
1. Delta combinado–conductual +0.037, IC bootstrap [-0.075,+0.156], n=32; y otro IC [-0.004,+0.265], n=45.
2. Usando TODAS las rondas ausentes <=20 y categorías supuestamente del paper en 41–60: AUC LOOCV 0.853 geometría, 0.784 conducta y 0.880 combinado, n=45.
No contamos aquí con el script, predicciones por díada, etiquetas ni procedimiento de bootstrap de esos cálculos. Clasifícalos como reportados sin verificar. No los combines con el análisis de cinco oportunidades. “Outcome externo del paper” sobre las mismas díadas no significa validación externa ni garantiza una etiqueta independiente de la ventana temprana. Verifica exactamente de dónde salen esas categorías y qué periodos utiliza su construcción.

La captura de un workflow multiagente muestra actividad y tokens; no certifica validez. No asumas que verificadores que comparten datos, funciones o premisas producen comprobaciones independientes. No entrenes modelos ni selecciones umbrales para reproducir esas cifras: esta solicitud es de investigación y propuesta científica.

## Búsqueda bibliográfica

Combina buscadores académicos/web, DOI/Crossref, OpenAlex, PubMed, bibliografías hacia atrás y citas hacia adelante, páginas de autores y repositorios. Usa CRAI solo si realmente tienes herramientas y sesión disponibles; en caso contrario registra la limitación y proporciona DOI/ruta para solicitar el texto. Nunca finjas leer un paper bloqueado. Distingue texto completo/secciones, resumen, preprint, artículo publicado y revisión. Deduplica congreso/preprint/artículo; verifica novedades hasta la fecha real de consulta.

Investiga al menos estas familias, adaptando términos a lo que descubras:
- collaborative visual search, shared gaze, joint multiple-object tracking, spatial division of labor;
- emergent/complementary roles, joint action, symmetry breaking, conventions, focal points, turn taking;
- distributed/interactive team cognition, collective foraging, specialization and selective social learning;
- graph-based task decomposition, hierarchical planning, subgoal discovery, diffusion kernels, proto-value functions, successor representations;
- spectral partitions, degeneracy, invariant subspaces, diffusion/flow-based communities y comparación con geometría euclídea.

Semillas, no lista cerrada:
- Goldstone, Andrade-Lotero, Hawkins y Roberts: The Emergence of Specialized Roles Within Groups, 10.1111/tops.12644.
- Roberts y Goldstone 2011: 10.1371/journal.pone.0022377.
- Hawkins y Goldstone 2016: 10.1371/journal.pone.0151670.
- Brennan et al. 2008: 10.1016/j.cognition.2007.05.012.
- Niehorster et al. 2019: 10.3758/s13414-018-01640-0.
- Wahn y Schmitz, revisión: 10.1007/s00426-022-01767-8; rastrea sus estudios primarios.
- Wahn y Kingstone 2020: 10.3758/s13414-020-02012-3, relevante para orientación del reparto y forma de pantalla.
- Doubles-pong: 10.1016/j.humov.2017.11.012.
- Richardson et al. 2015: 10.1037/xhp0000041.
- Correa et al. 2023: 10.1371/journal.pcbi.1011087.
- Wu, Schulz y Gershman: Inference and Search on Graph-Structured Spaces, 10.1007/s42113-020-00091-x.
- Tomov et al. 2020: 10.1371/journal.pcbi.1007594.
- Solway et al. 2014: 10.1371/journal.pcbi.1003779.
- Şimşek et al. 2005: 10.1145/1102351.1102454; Mahadevan y Maggioni 2007, JMLR 8, 2169–2231.
- Stachenfeld et al. 2017: 10.1038/nn.4650; Schapiro et al. 2013: 10.1038/nn.3331.
- Naito et al. 2022: 10.1038/s41598-022-12126-3.
- Wu et al., Specialization and selective social attention, CogSci 2021 (distinguir la versión de congreso de su preprint).

Para cada trabajo central registra pregunta, población/unidad, tarea, información compartida, representación/grafo, outcome, diseño de evaluación y limitación de transferencia a SODCL. Prioriza artículos que puedan cambiar nuestra conclusión. Una manipulación de rectángulos no es novedosa solo porque aquí pueda reinterpretarse espectralmente.

## Colaboradores

Andrade aporta contexto del experimento y validación conductual. Esteban Vargas Bernal tiene experiencia documentada en ciencia de redes, cadenas de Markov y comunidades; no asumas psicología experimental ni disponibilidad:
https://search.asu.edu/profile/4827825
https://sites.google.com/view/estebanvargasbernal/publicaciones
Artículo con Porter y Tien: 10.1137/21M1466803.
No confundas Esteban con Sebastián; no hay identidad/experiencia suficiente documentada de este último. No asignes autoría por asesoría. Propón encargos concretos sujetos a acuerdo.

## Entregables

1. Veredicto sobre lo conocido, lo replicado y la novedad aún no establecida.
2. Mapa razonado de literatura y tabla de evidencia con DOI/enlaces verificables; registro de búsquedas, límites y textos no abiertos. Sin cuota arbitraria de artículos.
3. Revisión adversarial de las afirmaciones del proyecto y de la auditoría pegada; lista de artefactos faltantes para verificarla.
4. Máximo tres siguientes preguntas, con una recomendada. Para cada una: hipótesis rivales, contraste discriminante, comparador justo, datos/permisos, resultado negativo informativo, riesgo de invalidez/novedad, horas/cómputo y criterio de continuar o cerrar.
5. Mantén como candidato el contraste conductuales / +E_xy / +E_F, pero evalúa si la literatura justifica otra pregunta. C–B prueba sustitución, no información condicional ni mecanismo. No agregues arquitecturas por defecto.
6. Una página para reunión con Andrade y Esteban: avances defendibles, decisiones y encargos. Plan de dos semanas a 6–8 horas semanales.

No conviertas el reanálisis retrospectivo en confirmatorio por escribir hoy un protocolo. No presupongas transferencia a Coordinator-and-Foragers/PsyNet. Prefiero una conclusión negativa o inconclusa defendible a salvar una hipótesis.

Archivos de apoyo que adjunto, si están disponibles: docs/TEMPORAL_REPAIR.md, docs/LITERATURE_REVIEW_2026-09-10.md, docs/LITERATURE_EXPANSION_AND_AUDIT_REVIEW.md, audit/temporal_repair_v1/summary.json, script y pruebas de reconstrucción. Declara qué archivos pudiste abrir realmente.
