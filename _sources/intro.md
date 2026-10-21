# Clasificación de la cobertura forestal a partir de variables cartográficas

**Curso:** Machine Learning  
**Entrega:** Entregable 3  
**Autoras:** Ana Galván, Valeria Oñate y María José Ruiz  
**Fecha:** 21 de octubre de 2026

## Resumen

La cartografía de la cobertura forestal es un insumo básico para el manejo de bosques, la
evaluación del riesgo de incendio y la conservación de hábitats, pero su levantamiento en
campo es costoso y, en áreas protegidas o remotas, difícil de mantener actualizado. Este
estudio evalúa en qué medida variables cartográficas disponibles de forma sistemática
(topografía, hidrografía, red vial, área administrativa y tipo de suelo) permiten predecir
el tipo de cobertura dominante en celdas de 30 × 30 m del Roosevelt National Forest
(Colorado, Estados Unidos), a partir del conjunto *Covertype* (581.012 celdas, 54
predictoras y 7 clases con una razón de desbalance de 103:1)
{cite}`uci_covertype,blackard2000`.

El trabajo se organiza en dos partes. La **Parte I** documenta el análisis exploratorio, una
auditoría de fuga de información que distingue la fuga del proxy geográfico, el
preprocesamiento y un modelo base de regresión logística multinomial, que alcanza un F1
macro de 0,558 (IC 95 %: 0,551–0,565) en la prueba reservada y cuya limitación es de
sesgo. La **Parte II** construye un diseño factorial de **112 combinaciones** (siete
algoritmos × cuatro estrategias de balanceo × cuatro métodos de optimización de
hiperparámetros), evaluadas mediante validación cruzada anidada; compara los métodos de
optimización en calidad, costo y convergencia; contrasta los modelos con una secuencia
jerárquica de pruebas estadísticas (Friedman, Nemenyi, DeLong, *bootstrap* y delta de
Cliff); interpreta el modelo finalista con SHAP y LIME, y analiza su eficiencia
computacional.

El modelo finalista, **XGBoost sin balanceo de clases**, alcanza en la prueba reservada un
**F1 macro de 0,927** (IC 95 %: 0,923–0,931), una *accuracy* de 0,940 y un AUC
uno-contra-el-resto de 0,997, con un tamaño de efecto máximo frente a la regresión
logística (δ de Cliff = 1). Los métodos de conjunto basados en árboles dominan a las demás
familias de forma consistente en todos los pliegues, y las estrategias de balanceo no
mejoran a los mejores algoritmos. La optimización bayesiana solo aventaja a la búsqueda en
rejilla cuando el espacio de búsqueda tiene varias dimensiones y el presupuesto supera la
fase de exploración inicial. La elevación es la variable dominante en todos los modelos. Al
predecir un área silvestre completa no incluida en el entrenamiento, el finalista cae a
una *accuracy* de 0,52, por debajo de la regresión logística (0,56), lo que delimita su
dominio de validez a las áreas muestreadas.

**Palabras clave:** clasificación multiclase, cobertura forestal, desbalance de clases,
validación cruzada anidada, optimización de hiperparámetros, XGBoost, SHAP.

## Pregunta de investigación

¿En qué medida las características topográficas, hidrológicas y edáficas de un sitio
permiten identificar el tipo de cobertura forestal que lo ocupa, y qué combinación de
algoritmo, tratamiento del desbalance y estrategia de optimización ofrece el mejor
equilibrio entre desempeño, costo computacional e interpretabilidad?

## Contribuciones

1. Un **análisis exploratorio y una auditoría de fuga** que separan la fuga de información
   del proxy geográfico y cuantifican, mediante validación por área silvestre, cuánto del
   desempeño depende de las zonas muestreadas (Parte I).
2. Un **benchmark factorial de 112 combinaciones** con validación cruzada anidada y
   registro estructurado de cada corrida, que identifica a los métodos de conjunto basados
   en árboles como la familia dominante (capítulo 5).
3. Una **comparación de cuatro métodos de optimización** que combina la evidencia del
   benchmark con un experimento controlado de presupuesto idéntico y cinco semillas, con
   curvas de desempeño en cualquier momento y una evaluación de *Successive Halving*
   (capítulo 6).
4. Una **comparación estadística jerárquica** que documenta la falta de potencia de la
   prueba de Nemenyi con tres pliegues y la complementa con un análisis por familias,
   pruebas de DeLong y tamaños de efecto (capítulo 7).
5. Un **análisis de interpretabilidad y de eficiencia** que contrasta SHAP con LIME,
   compara implementaciones estándar y optimizadas de cada algoritmo y estima
   empíricamente sus exponentes de escalamiento (capítulos 8 y 9).

## Ruta metodológica

1. Reserva del conjunto de prueba (20 %, estratificado por clase) antes de cualquier
   decisión basada en los datos.
2. Análisis exploratorio sobre el conjunto de entrenamiento: variable objetivo, análisis
   unidimensional, bidimensional y multivariado.
3. Auditoría de fuga de información y análisis de la estructura espacial.
4. Preprocesamiento encapsulado en un `Pipeline` de scikit-learn {cite}`pedregosa2011`,
   ajustado solo con datos de entrenamiento.
5. Modelo base (regresión logística multinomial) frente a clasificadores triviales.
6. Benchmark factorial con validación cruzada anidada y reentrenamiento del finalista con
   el conjunto de entrenamiento completo.
7. Comparación de métodos de optimización, comparación estadística, interpretabilidad y
   análisis de eficiencia computacional.

## Organización del documento

| Parte | Capítulo | Contenido |
|---|---|---|
| I | 1. Base de datos | Problema, justificación, fuente y licencia, diccionario de variables, estructura, tamaño de muestra, calidad de los datos y consideraciones éticas |
| I | 2. Análisis exploratorio | Variable objetivo; análisis unidimensional, bidimensional y multivariado |
| I | 2.5–2.8 Fuga y estructura | Auditoría de fuga de información y componentes temporal, espacial y espacio-temporal |
| I | 2.9 Preprocesamiento | Decisiones de preprocesamiento y hallazgos del análisis exploratorio que las motivan |
| I | 3. Modelo base | Línea base trivial, regresión logística, validación cruzada, métricas, calibración, validación por área y curva de aprendizaje |
| I | 4. Conclusiones de la Parte I | Hallazgos principales y limitaciones del modelo base |
| II | 5. Benchmark combinatorio | Diseño factorial de 112 combinaciones, validación anidada y evaluación del finalista |
| II | 6. Optimización de hiperparámetros | Grid, Random, Bayesiana (TPE) y Genética; desempeño en cualquier momento y *Successive Halving* |
| II | 7. Comparación estadística | Friedman, Nemenyi, DeLong, *bootstrap* y tamaño del efecto |
| II | 8. Interpretabilidad | SHAP global y local, y comparación con LIME |
| II | 9. Eficiencia, reproducibilidad y conclusiones | Complejidad algorítmica, implementaciones optimizadas, escalamiento, paralelización, revisión crítica y conclusiones generales |
| — | Apéndice A | Archivos, entorno, semillas y código auxiliar |
