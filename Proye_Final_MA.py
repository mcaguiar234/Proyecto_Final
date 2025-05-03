# -*- coding: utf-8 -*-
"""
MAESTRÍA CIENCIA DE DATOS
FUNDAMENTOS DE CIENCIA DE DATOS
UNIVERSIDAD YACHAY TECH
FECHA: 02/05/2025

@author: Magaly_Aguiar
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_curve, classification_report,
    confusion_matrix, roc_auc_score, f1_score)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. CARGA DE DATOS ----------
directorio = "C:/Users/aguia/OneDrive/MAESTRIA/FUNDAMENTOS CIENCIA DE DATOS/Trab_Final/BTA18.xlsx"
#directorio = "R:/CGTPE/DECON/AS/Carp_Tempor/Magaly Aguiar/2018/BTA18_edit_MA.xlsx"
base = pd.read_excel(directorio)
base_original = base.copy()
base_original['ID'] = base_original.index

tabla_frecuencia = base_original.groupby(['ejercicio', 'p02']).size().reset_index(name='frecuencia')
tabla_frecuencia1 = base_original.groupby(['p02']).size().reset_index(name='frecuencia')

base.reset_index(drop=True, inplace=True)  # Asegura orden limpio
base['ID'] = base.index                    # Asigna ID único a cada fila

# ---------- 2. FILTRADO Y LIMPIEZA ----------
base = base[base['p03'] >= 12]
base = base[(base['ejercicio'] <= 2019) & (base['ejercicio'] != 2018)]
base = base[base['ut01'] <= 2]
base = base[base['p04']<= 9]
print(base["p04"].dtype)

tabla_frecuencia = base.groupby(['ejercicio', 'p02']).size().reset_index(name='frecuencia')
tabla_frecuencia1 = base.groupby(['p02']).size().reset_index(name='frecuencia')


print(base['ejercicio'].unique())
print(base['p02'].unique()) #variable sexo
print(base['p04'].unique()) #variable relación parentesco
print(base['p06'].unique()) #variable estado civil
print(base['p12a'].unique()) #variable nivel de instrucción
print(base['p08'].unique()) #variable de étnia
print(base['p05a'].unique()) #variable seguridad social

# ----------- Tratamiento previo --------------
categ_p04 = {1:1,2:1,3:1,4:3,5:2,6:1,7:3,8:1,9:3}
base['p04_grup'] = base['p04'].map(categ_p04)
print(base)

categ_p06 = {1:1,5:1,6:2,2:2,3:2,4:2}
base['p06_grup'] = base['p06'].map(categ_p06)
print(base)

# Crear variable categórica para edad
def clasificar_edad(p03):
    if p03 <= 21:
        return 1
    elif p03 >= 60:
        return 4
    elif p03 >= 40:
        return 3
    else:  # entre 22 y 39
        return 2

base['edad_grupo'] = base['p03'].apply(clasificar_edad)

# Revisión final
print(base[['p03', 'edad_grupo']].head())

# Nivel de instrucción
grupo_instruccion = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,   # Básica o menos
    6: 2, 7: 2,                     # Secundaria
    8: 3, 9: 3, 10: 3, 11: 3        # Educación superior o postgrado
}
base['instruccion_grup'] = base['p12a'].map(grupo_instruccion)

# Etnia 1 = indígena, 2 = afroecuatoriano, 3 = mestizo, 4 = blanco, otros agrupados
grupo_etnia = {
    1: 1,        # Indígena
    2: 2,        # Afroecuatoriano
    3: 3,        # Mestizo
    4: 4,        # Blanco
    5: 5, 6: 5, 7: 5  # Otros
}
base['etnia_grup'] = base['p08'].map(grupo_etnia)

# ---------- 3. FUNCIONES AUXILIARES ----------
def preparar_datos(df, incluir_ut01=True):
    df = df.copy()
    id_col = df['ID']  # guarda temporalmente
    df['p24'] = df['p24'].fillna(0)
    columnas_dummies = ['p04_grup', 'p06_grup', 'edad_grupo', 'instruccion_grup', 'etnia_grup']
    df = pd.get_dummies(df, columns=columnas_dummies, drop_first=True)
    df['ID'] = id_col  # restaura ID
    if incluir_ut01 and 'ut01' in df.columns:
        df['ut01'] = df['ut01'].map({1: 1, 2: 0})
    return df

def entrenar_rf_calibrado(df,umbral=0.5):
    X = df.drop(columns=['ut01'])
    y = df['ut01']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    modelo_base = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    modelo_cal = CalibratedClassifierCV(modelo_base, method = 'sigmoid', cv=5)
    modelo_cal.fit(X_resampled, y_resampled)
    y_probs = modelo_cal.predict_proba(X_test)[:,1]
    y_pred = (y_probs >= umbral).astype(int)
    print("Clasificación (RF Calibrado, umbral = {umbral}):\n", classification_report (y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_probs))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    return modelo_cal

#****************************************
#* ENTRENAMIENTO
#****************************************
# --- 1. Filtrar y preparar datos ---
#----- HOMBRES -------------------
print("\n----- MODELO: HOMBRES -----")
base_hombres = base[base['p02'] == 1]

print('ID' in base_hombres.columns)  # debe dar True

# Asegúrate de que 'edad_grupo', 'p06_grup' y 'p04_grup' ya están creadas antes de esta parte
seleccion = ['ID', 'area', 'edad_grupo', 'p06_grup', 'ut01', 'p24', 'p04_grup', 'instruccion_grup', 'etnia_grup']
base_h0 = preparar_datos(base_hombres[seleccion], incluir_ut01=True)

print("\n✅ Variables usadas en el modelo - Hombres:")
print(base_h0.columns.tolist())

# --- 2. Entrenar modelo y mostrar coeficientes ---
modelo_rf_h = entrenar_rf_calibrado(base_h0)

# Acceder al estimador RandomForest dentro del calibrador (primer pliegue)
modelo_base_rf = modelo_rf_h.calibrated_classifiers_[0].estimator

# Obtener importancias
importancias = modelo_base_rf.feature_importances_
columnas = base_h0.drop(columns=['ut01']).columns

# Ordenar e imprimir
importancia_ordenada = sorted(zip(columnas, importancias), key=lambda x: x[1], reverse=True)

print("\n📊 Importancia de variables (Random Forest Calibrado):")
for var, imp in importancia_ordenada:
    print(f"{var:25} : {imp:.4f}")

import matplotlib.pyplot as plt

# Convertir a listas separadas
variables = [x[0] for x in importancia_ordenada]
importancias_valores = [x[1] for x in importancia_ordenada]

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.barh(variables[::-1], importancias_valores[::-1])  # De abajo hacia arriba
plt.xlabel("Importancia")
plt.title("Importancia de Variables - Random Forest Calibrado")
plt.tight_layout()
plt.show()

# ---------- 5. MUJERES ----------
print("\n----- MODELO: MUJERES -----")
# Filtrar datos de mujeres
base_mujeres = base[base['p02'] == 2]

# Variables seleccionadas (sin 'ejercicio')
seleccion = [ 'ID', 'area', 'edad_grupo', 'p06_grup', 'ut01', 'p24',
    'p04_grup', 'instruccion_grup', 'etnia_grup'
]

# Preparar datos
base_m0 = preparar_datos(base_mujeres[seleccion], incluir_ut01=True)

# Entrenar modelo calibrado con RF (umbral estándar)
modelo_rf_m = entrenar_rf_calibrado(base_m0)
modelo_rf_m1 = entrenar_rf_calibrado(base_m0, umbral=0.40)


# Acceder al estimador Random Forest dentro del calibrador (primer pliegue)
modelo_base_rf_m = modelo_rf_m.calibrated_classifiers_[0].estimator

importancias_m = modelo_base_rf_m.feature_importances_
columnas_m = base_m0.drop(columns=['ut01']).columns

importancia_ordenada_m = sorted(zip(columnas_m, importancias_m), key=lambda x: x[1], reverse=True)

print("\n📊 Importancia de variables (Mujeres - RF Calibrado):")
for var, imp in importancia_ordenada_m:
    print(f"{var:25} : {imp:.4f}")
    
#***************************************************
# PREDICCIÓN CON RANDOM FOREST Y EXPORTACIÓN 
#***************************************************

# Filtrar datos del 2018
base_2018 = base_original [base_original ['ejercicio'] == 2018]
base_2018 = base_2018[base_2018['p03'] >= 12]
print('ID' in base_2018.columns)  
# Clasificaciones agrupadas
base_2018['p04_grup'] = base_2018['p04'].map(categ_p04)
base_2018['p06_grup'] = base_2018['p06'].map(categ_p06)
base_2018['edad_grupo'] = base_2018['p03'].apply(clasificar_edad)
base_2018['instruccion_grup'] = base_2018['p12a'].map(grupo_instruccion)
base_2018['etnia_grup'] = base_2018['p08'].map(grupo_etnia)

# ---------- HOMBRES ----------
# Filtrar hombres
base_2018_h = base_2018[base_2018['p02'] == 1]

# Variables predictoras usadas en entrenamiento de hombres
seleccion_h = ['ID', 'area', 'p24', 'p06_grup', 'p04_grup', 'edad_grupo', 'instruccion_grup', 'etnia_grup']
base_2018_h = preparar_datos(base_2018_h[seleccion_h], incluir_ut01=False)

# Alinear columnas al modelo de entrenamiento (base_h0)
for col in base_h0.drop(columns=['ut01']).columns:
    if col not in base_2018_h.columns:
        base_2018_h[col] = 0

# Reordenar columnas
base_2018_h = base_2018_h[base_h0.drop(columns=['ut01']).columns]

# ----------- 3. Predecir con modelo Random Forest Calibrado -----------

probs_2018_h = modelo_rf_h.predict_proba(base_2018_h)[:, 1]
preds_2018_h = (probs_2018_h >= 0.5).astype(int) 

# Agregar resultados
base_2018_h['prob_ut01'] = probs_2018_h
base_2018_h['pred_ut01'] = preds_2018_h 

# ---------- MUJERES ----------
# ----------- 1. Filtrar y crear variables previas -----------
base_2018 = base_original[base_original['ejercicio'] == 2018]
base_2018 = base_2018[base_2018['p03'] >= 12]  # edad válida

# Clasificaciones agrupadas
base_2018['p04_grup'] = base_2018['p04'].map(categ_p04)
base_2018['p06_grup'] = base_2018['p06'].map(categ_p06)
base_2018['edad_grupo'] = base_2018['p03'].apply(clasificar_edad)
base_2018['instruccion_grup'] = base_2018['p12a'].map(grupo_instruccion)
base_2018['etnia_grup'] = base_2018['p08'].map(grupo_etnia)

# ----------- 2. Filtrar mujeres y preparar variables -----------

# Seleccionar solo mujeres
base_2018_m = base_2018[base_2018['p02'] == 2]

# Selección de variables predictoras (sin ut01)
seleccion_m = ['ID', 'area', 'p24', 'p06_grup', 'p04_grup', 'edad_grupo', 'instruccion_grup', 'etnia_grup']
base_2018_m = preparar_datos(base_2018_m[seleccion_m], incluir_ut01=False)

# Alinear columnas con base de entrenamiento (base_m0)
for col in base_m0.drop(columns=['ut01']).columns:
    if col not in base_2018_m.columns:
        base_2018_m[col] = 0

# Reordenar columnas
base_2018_m = base_2018_m[base_m0.drop(columns=['ut01']).columns]

# ----------- 3. Predecir con el modelo RF calibrado (umbral 0.40) -----------

probs_2018_m = modelo_rf_m.predict_proba(base_2018_m)[:, 1]
preds_2018_m = (probs_2018_m >= 0.40).astype(int)

# Agregar resultados
base_2018_m['prob_ut01'] = probs_2018_m
base_2018_m['pred_ut01'] = preds_2018_m

#************************************************************
#* Incorporación de los datos obtenido en la bdd original
#************************************************************
base_2018_h['ejercicio'] = 2018
base_2018_m['ejercicio'] = 2018

# Unir ambos conjuntos
base_2018_pred = pd.concat([base_2018_h, base_2018_m], ignore_index=True)
base_2018_pred = base_2018_pred[['ID', 'pred_ut01']]
base_2018_pred['ut01'] = base_2018_pred['pred_ut01'].map({1: 1, 0: 2})
base_2018_pred = base_2018_pred.drop(columns='pred_ut01')

#Obtner la base 2018
base2018a = base_original[base_original['ejercicio'] == 2018].copy()
base2018a.columns
#Eliminar la columna original ut01
base2018a = base2018a.drop(columns=['ut01'], errors='ignore')
#Hacer un merge de los dos archivos, donde solo se pegue la variable proyectada mediante ID
base_2018_con_pred = pd.merge(base2018a, base_2018_pred, on='ID', how='left')
#Unir con el resto de la base original (excluyendo 2018)
base_sin_2018 = base_original[base_original['ejercicio'] != 2018].copy()
base_final = pd.concat([base_sin_2018, base_2018_con_pred], ignore_index=True)

base_final = base_final[base_final['ut01'].isin([1, 2])]

tabla_frecuencia_ut = base.groupby(['ejercicio', 'p02', 'ut01']).size().reset_index(name='frecuencia')
tabla_frecuencia1 = base_final.groupby(['ejercicio', 'ut01']).size().reset_index(name='frecuencia')

#----------------- PARTE 2 -------------------------------------------

# Filtrar solo los años válidos para entrenamiento y valores válidos en ut15
base_ut15 = base_final.copy()
base_ut15 = base_ut15[
    (base_ut15['p03'] >= 12) &
    (base_ut15['ejercicio'].isin([2012, 2015, 2017, 2019])) &
    (base_ut15['ut01'] == 1) &              # 👈 filtro clave
    (base_ut15['ut15'].isin([1, 2])) &
    (base_ut15['p04'] <= 9)
]
base_ut15['ID'] = base_ut15.index
tabla_frecuencia15 = base_ut15.groupby(['p02', 'ut01']).size().reset_index(name='frecuencia')
tabla_frecuencia15 = base_original.groupby(['ejercicio', 'ut01']).size().reset_index(name='frecuencia')


# Clasificaciones agrupadas
base_ut15['p04_grup'] = base_ut15['p04'].map(categ_p04)
base_ut15['p06_grup'] = base_ut15['p06'].map(categ_p06)
base_ut15['edad_grupo'] = base_ut15['p03'].apply(clasificar_edad)
base_ut15['instruccion_grup'] = base_ut15['p12a'].map(grupo_instruccion)
base_ut15['etnia_grup'] = base_ut15['p08'].map(grupo_etnia)

base_ut15.columns
# Separar hombres y mujeres y preparar datos
# Variables a usar
seleccion_ut15 = ['ID', 'area', 'p24', 'p06_grup', 'p04_grup',
                  'edad_grupo', 'instruccion_grup', 'etnia_grup', 'ut15']

# HOMBRES
base_hombres_ut15 = base_ut15[base_ut15['p02'] == 1].copy()
base_hombres_ut15['ut15'] = base_hombres_ut15['ut15'].map({1: 1, 2: 0})
base_h0_ut15 = preparar_datos(base_hombres_ut15[seleccion_ut15], incluir_ut01=False)

# MUJERES
base_mujeres_ut15 = base_ut15[base_ut15['p02'] == 2].copy()
base_mujeres_ut15['ut15'] = base_mujeres_ut15['ut15'].map({1: 1, 2: 0})
base_m0_ut15 = preparar_datos(base_mujeres_ut15[seleccion_ut15], incluir_ut01=False)

#Diseño de modelo
def entrenar_rf_ut15(df):
    X = df.drop(columns=['ut15'])
    y = df['ut15']
    # Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Balanceo de clases
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    # Entrenar modelo Random Forest
    modelo = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    modelo.fit(X_resampled, y_resampled)
    # Evaluar en test
    y_pred = modelo.predict(X_test)
    y_probs = modelo.predict_proba(X_test)[:, 1]
    print("🔍 Clasificación (Random Forest - ut15):\n", classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_probs))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    return modelo

# Predicción hombres
modelo_rf_h_ut15 = entrenar_rf_ut15(base_h0_ut15)

#Ajuste del umbral
# Ajustar umbral para hombres

X = base_h0_ut15.drop(columns=['ut15'])
y = base_h0_ut15['ut15']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_probs = modelo_rf_h_ut15.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_idx = f1_scores.argmax()
best_threshold_h = thresholds[best_idx]

# Resultado ajustado
y_pred_h = (y_probs >= best_threshold_h).astype(int)
print(f"\n🔧 Mejor umbral para HOMBRES (F1): {best_threshold_h:.3f}")
print(classification_report(y_test, y_pred_h))

def entrenar_rf_ut15_mujeres(df):
    X = df.drop(columns=['ut15'])
    y = df['ut15']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    # Aquí damos más peso a clase 0 (por ejemplo, 2 a clase 0 y 1 a clase 1)
    modelo = RandomForestClassifier(n_estimators=100, class_weight={0: 2, 1: 1}, random_state=42)
    modelo.fit(X_resampled, y_resampled)
    y_pred = modelo.predict(X_test)
    y_probs = modelo.predict_proba(X_test)[:, 1]
    print("🔍 Clasificación (RF - Mujeres, ponderado):\n", classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_probs))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    return modelo, y_test, y_pred

#Entrenar modelos
modelo_rf_m_ut15 = entrenar_rf_ut15_mujeres(base_m0_ut15)

#--------Predicción base 2018 ----------------------------------------
#Filtro inicial
base_2018_ut15 = base_final[
    (base_final['ejercicio'] == 2018) &
    (base_final['ut01'] == 1) &   # Solo quienes sí hacen trabajo no remunerado
    (base_final['p03'] >= 12) &
    (base_final['p04'] <= 9)
].copy()

#Crear variables agrupadas
base_2018_ut15['p04_grup'] = base_2018_ut15['p04'].map(categ_p04)
base_2018_ut15['p06_grup'] = base_2018_ut15['p06'].map(categ_p06)
base_2018_ut15['edad_grupo'] = base_2018_ut15['p03'].apply(clasificar_edad)
base_2018_ut15['instruccion_grup'] = base_2018_ut15['p12a'].map(grupo_instruccion)
base_2018_ut15['etnia_grup'] = base_2018_ut15['p08'].map(grupo_etnia)

#Aplicar modelos para hombres
base_2018_h_ut15 = base_2018_ut15[base_2018_ut15['p02'] == 1].copy()
base_2018_h_ut15 = preparar_datos(base_2018_h_ut15[seleccion_ut15[:-1]], incluir_ut01=False)

# Alinear columnas
for col in base_h0_ut15.drop(columns=['ut15']).columns:
    if col not in base_2018_h_ut15.columns:
        base_2018_h_ut15[col] = 0
base_2018_h_ut15 = base_2018_h_ut15[base_h0_ut15.drop(columns=['ut15']).columns]

# Predicción
probs_2018_h = modelo_rf_h_ut15.predict_proba(base_2018_h_ut15)[:, 1]
preds_2018_h = (probs_2018_h >= 0.5).astype(int)

# Guardar resultados
base_2018_h_ut15['pred_ut15'] = preds_2018_h
base_2018_h_ut15['prob_ut15'] = probs_2018_h

#Aplicar modelos para mujeres
base_2018_m_ut15 = base_2018_ut15[base_2018_ut15['p02'] == 2].copy()
base_2018_m_ut15 = preparar_datos(base_2018_m_ut15[seleccion_ut15[:-1]], incluir_ut01=False)

# Alinear columnas
for col in base_m0_ut15.drop(columns=['ut15']).columns:
    if col not in base_2018_m_ut15.columns:
        base_2018_m_ut15[col] = 0
base_2018_m_ut15 = base_2018_m_ut15[base_m0_ut15.drop(columns=['ut15']).columns]

# Predicción
probs_2018_m = modelo_rf_m_ut15.predict_proba(base_2018_m_ut15)[:, 1]
preds_2018_m = (probs_2018_m >= 0.4).astype(int)

# Guardar resultados
base_2018_m_ut15['pred_ut15'] = preds_2018_m
base_2018_m_ut15['prob_ut15'] = probs_2018_m

#Unir y preparar un merge
# Unir hombres y mujeres
base_2018_pred_ut15 = pd.concat([
    base_2018_h_ut15[['ID', 'pred_ut15']],
    base_2018_m_ut15[['ID', 'pred_ut15']]
], ignore_index=True)

base_2018_pred_ut15['ut15'] = base_2018_pred_ut15['pred_ut15'].map({1: 1, 0: 2})
base_2018_pred_ut15 = base_2018_pred_ut15.drop(columns='pred_ut15')


# Casos con ut15 == 2 (no cocinó)
base2018b = base_final[
    (base_final['ejercicio'] == 2018) &
    (base_final['ut01'] == 2)
][['ID', 'ut15']]

base_2018_union = pd.concat([base_2018_pred_ut15, base2018b], ignore_index=True)

#Obtner la base 2018
base2018c = base_final[base_final['ejercicio'] == 2018].copy()
base2018c.columns
#Eliminar la columna original ut01
base2018c = base2018c.drop(columns=['ut15'], errors='ignore')
#Hacer un merge de los dos archivos, donde solo se pegue la variable proyectada mediante ID
base_2018_con_ut15 = pd.merge(base2018c, base_2018_union, on='ID', how='left')
#Unir con el resto de la base original (excluyendo 2018)
base_sin_2018_ut15 = base_final[base_final['ejercicio'] != 2018].copy()
base_final2 = pd.concat([base_sin_2018_ut15, base_2018_con_ut15], ignore_index=True)

tabla_frecuencia_ut15 = base_final2.groupby(['ejercicio', 'p02', 'ut15']).size().reset_index(name='frecuencia')
tabla_frecuencia15 = base_final2.groupby(['ejercicio', 'ut15']).size().reset_index(name='frecuencia')

#---------------------------- PARTE 3 ---------------------------------------

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# =====================
# 1. BASE DE DATOS INICIAL
# =====================
df = base_final2.copy()

# =====================
# 2. LIMPIEZA Y VARIABLES DERIVADAS
# =====================
df = df[(df['ut01'] == 1) & (df['ut15'] == 1)].copy()
df['p24'] = df['p24'].fillna(0)
df['ut15_semt'] = df['ut15_semt'].fillna(0)
df = df.dropna(subset=['p03'])
df['p24_x_area'] = df['p24'] * df['area']
df['p24_x_edad'] = df['p24'] * df['p03']

# =====================
# 3. SEPARAR ENTRENAMIENTO Y PROYECCIÓN 2018
# =====================

# Años válidos para entrenamiento
anios_entrenamiento = [2015, 2017, 2023, 2024]
df_entreno = df[df['ejercicio'].isin(anios_entrenamiento)].copy()
df_2018 = df[df['ejercicio'] == 2018].copy()

# Eliminar ut15_semt de df_2018 si existe
if 'ut15_semt' in df_2018.columns:
    df_2018 = df_2018.drop(columns=['ut15_semt'])


# =====================
# 4. FUNCION DE ENTRENAMIENTO DE MODELO
# =====================
def entrenar_modelo(df, sexo):
    df = df[df['p02'] == sexo].copy()
    X = df[['p03', 'p06', 'p12a', 'p24', 'area', 'p24_x_area', 'p24_x_edad']]
    y = df['ut15_semt']

    cat_cols = ['p03', 'p06', 'p12a', 'area']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough')

    model = Pipeline([
        ('prep', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n--- Métricas de Regresión para {'Hombres' if sexo == 1 else 'Mujeres'} ---")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")
    print(f"Promedio real: {np.mean(y_test):.2f} | Promedio predicho: {np.mean(y_pred):.2f}")

    return model, y_test, y_pred

# =====================
# 5. ENTRENAMIENTO
# =====================
modelo_hombres = entrenar_modelo(df_entreno, sexo=1)
modelo_mujeres = entrenar_modelo(df_entreno, sexo=2)

# =====================
# 6. PREDICCION 2018
# =====================
def proyectar_2018(df_2018, modelo, sexo):
    df = df_2018[df_2018['p02'] == sexo].copy()
    X = df[['p03', 'p06', 'p12a', 'p24', 'area', 'p24_x_area', 'p24_x_edad']].dropna()
    df = df.loc[X.index].copy()
    df['ut15_semt_pred'] = modelo.predict(X)
    return df

df_2018_hombres = proyectar_2018(df_2018, modelo_hombres, sexo=1)
df_2018_mujeres = proyectar_2018(df_2018, modelo_mujeres, sexo=2)

# =====================
# 7. RESULTADOS
# =====================
print("\n=== PROMEDIOS PREDICHOS UT15_SEMT - 2018 ===")
print(f"Hombres: {df_2018_hombres['ut15_semt_pred'].mean():.2f} horas/semana")
print(f"Mujeres: {df_2018_mujeres['ut15_semt_pred'].mean():.2f} horas/semana")

# =====================
# 8. COMPARACION CON PROMEDIOS HISTORICOS
# =====================
promedios_historicos = df_entreno.groupby(['p02', 'ejercicio'])['ut15_semt'].mean().reset_index()
promedios_historicos['grupo'] = promedios_historicos['p02'].map({1: 'Hombres', 2: 'Mujeres'})

# Agregar fila con proyección 2018
promedios_2018 = pd.DataFrame({
    'p02': [1, 2],
    'ejercicio': [2018, 2018],
    'ut15_semt': [df_2018_hombres['ut15_semt_pred'].mean(), df_2018_mujeres['ut15_semt_pred'].mean()],
    'grupo': ['Hombres', 'Mujeres']
})

promedios_comparacion = pd.concat([promedios_historicos, promedios_2018], ignore_index=True)

# =====================
# 9. VISUALIZACION
# =====================
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.lineplot(data=promedios_comparacion, x='ejercicio', y='ut15_semt', hue='grupo', marker='o')
plt.title('Promedios Históricos y Predichos de Horas Semanales de Cocina (ut15_semt)')
plt.xlabel('Año')
plt.ylabel('Promedio de horas')
plt.grid(True)
plt.show()

# =====================
# 10. UNIR PREDICCIONES A BASE ORIGINAL
# =====================

# Concatenar predicciones de hombres y mujeres
predicciones_2018 = pd.concat([df_2018_hombres[['ID', 'ut15_semt_pred']],
                               df_2018_mujeres[['ID', 'ut15_semt_pred']]])


# Eliminar columna original 'ut15_semt' solo en 2018
base_2018_temp = base_final2[base_final2['ejercicio'] == 2018].copy()
base_2018_temp = base_2018_temp.drop(columns=['ut15_semt'], errors='ignore')

# Unir las predicciones por ID
base_2018_con_pred = pd.merge(base_2018_temp, predicciones_2018, on='ID', how='left')

# Renombrar predicción como 'ut15_semt'
base_2018_con_pred.rename(columns={'ut15_semt_pred': 'ut15_semt'}, inplace=True)

# Reconstruir base completa
base_sin_2018 = base_final2[base_final2['ejercicio'] != 2018].copy()
base_final2 = pd.concat([base_sin_2018, base_2018_con_pred], ignore_index=True)
