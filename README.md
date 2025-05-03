
# Predicción de Actividades de Trabajo No Remunerado en Ecuador (2012–2019)

Este repositorio contiene el proyecto final del curso de Fundamentos de Ciencia de Datos de la Maestría en Ciencia de Datos, enfocado en el análisis y predicción de actividades de trabajo no remunerado (ATNR) en el Ecuador, a partir de datos de encuestas de uso del tiempo y otras fuentes sociodemográficas.

## Descripción General

El objetivo principal del proyecto es predecir la probabilidad y la intensidad (horas semanales) de participación en actividades de cocina dentro del hogar (actividad `ut15`) entre personas que realizan trabajo no remunerado (`ut01`), utilizando modelos de machine learning basados en Random Forest, calibración de umbral de decisión y evaluación de resultados diferenciados por sexo.

El trabajo se enfoca particularmente en estimar estos indicadores para el año 2018, un año sin información directa, a partir de los datos observados entre 2012 y 2024.

## 🛠️ Tecnologías Utilizadas

- Python 3.10
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- VS Code / Jupyter Notebook

## ⚙️ Flujo de Trabajo

### Parte 1: Predicción de `ut01` (personas que realizan trabajo no remunerado)
- Entrenamiento de modelo Random Forest para hombres y mujeres por separado.
- Ajuste de umbral de decisión (`0.40` para mujeres, `0.50` para hombres).
- Aplicación del modelo al año 2018 y generación de la variable predicha `ut01_pred`.

### Parte 2: Predicción de `ut15` (personas que cocinan entre quienes hacen ATNR)
- Filtrado a personas con `ut01 == 1`.
- Entrenamiento de nuevos modelos Random Forest con calibración de umbral.
- Aplicación de predicción de `ut15` para 2018, generando `ut15_pred`.

### Parte 3: Predicción de `ut15_sem` (horas semanales dedicadas a cocinar)
- Modelos de regresión Random Forest Regressor, separados por sexo.
- Entrenamiento con datos observados 2015, 2017, 2023 y 2024.
- Aplicación al subconjunto de personas con `ut01 == 1` y `ut15 == 1`.
- Generación de `ut15_semt_pred`.

### Análisis complementarios:
- Evaluación del impacto del uso de SMOTE vs. calibración de umbral.
- Visualización de la distribución de horas de cocina por año (`kdeplot`, histogramas).
- Generación de interacciones relevantes como:
  - `ut15_semt * p23` (cocina x trabajo remunerado)
  - `edad_grupo * instrucción`
  - `p24 * zona` (horas trabajadas x área)
  - `p24 * etnia_grupo`

## 🧪 Evaluación de Modelos

- Clasificación: precisión, recall, F1-score, matriz de confusión.
- Regresión: MAE, MSE, RMSE, R².
- Métricas diferenciadas por sexo y año.
- Comparación entre modelos balanceados con SMOTE y calibración de umbral.
- Selección del mejor enfoque para cada caso.

## 📊 Resultados Finales

- Consolidación en la base `base_final2` de los resultados predichos para el año 2018.
- Inclusión de identificador único (`ID`) para trazabilidad de registros.
- Visualización de importancias de variables por modelo.

## 👩‍💻 Autora

**Magaly Cecilia Aguiar Baño**  
Estudiante de Maestría en Ciencia de Datos  
Universidad Yachay Tech

## 📫 Contacto

- GitHub: [mcaguiar234](https://github.com/mcaguiar234)
- Correo: *[tu_correo_institucional]*
