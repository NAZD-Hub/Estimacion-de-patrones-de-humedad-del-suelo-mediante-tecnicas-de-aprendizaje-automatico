import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
# Leer el conjunto de datos desde el archivo CSV
archivo_csv = 'data/dataset_humedad_suelo_y_clima_final_2011-2015_filtrado.csv'
# Imprimir el nombre de la primera columna

df = pd.read_csv(archivo_csv)

# Visualizar las primeras filas del dataframe
print(df.head())
df['fecha'] = pd.to_datetime(df['DateTime'])  # Asegúrate de que la columna de fecha esté en formato de fecha y hora

df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['dia'] = df['fecha'].dt.day

# Preparar datos para el entrenamiento
# Definir las características y la variable objetivo
X = df[['temp', 'dwpt', 'prcp','rhum_fil','mes']]  # Características
y = df['VW_30cm']                           # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calcular el error absoluto medio (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Visualización de la importancia de las características
feature_importance = model.feature_importances_
plt.barh(['temp', 'dwpt', 'prcp','rhum_fil','mes'], feature_importance)
plt.xlabel('Importancia de la Característica')
plt.ylabel('Características')
plt.title('Importancia de las Características en la Predicción de la Humedad del Suelo')
plt.show()


# Leer el conjunto de datos desde el archivo CSV
archivo_csv = 'data/dataset_humedad_suelo_y_clima_final_2016_filtrado.csv'
# Imprimir el nombre de la primera columna

df_prueba = pd.read_csv(archivo_csv)

# Convertir la columna de fecha a datetime
df_prueba['fecha'] = pd.to_datetime(df_prueba['DateTime'])

# Extraer el mes de la fecha
df_prueba['mes'] = df_prueba['fecha'].dt.month

# Seleccionar las características y la variable objetivo para el conjunto de prueba
X_prueba = df_prueba[['temp', 'dwpt', 'prcp', 'rhum_fil', 'mes']]
y_prueba = df_prueba['VW_30cm']

# Hacer predicciones sobre el conjunto de prueba
y_pred_prueba = model.predict(X_prueba)
df_prueba['predicciones'] = y_pred_prueba

# Calcular el error cuadrático medio (MSE) en los datos de prueba
mse_prueba = mean_squared_error(y_prueba, y_pred_prueba)
print(f'Mean Squared Error (prueba): {mse_prueba}')

# Calcular el error absoluto medio (MAE)
mae_prueba = mean_absolute_error(y_prueba, y_pred_prueba)
print(f'Mean Absolute Error: {mae_prueba}')

#-------------------------------------------------------
# Asumiendo que ya has cargado tu DataFrame y se llama df
df_prueba['DateTime'] = pd.to_datetime(df['DateTime'])

# Establecer la columna 'DateTime' como el índice del DataFrame
df_prueba.set_index('DateTime', inplace=True)
df_prueba = df_prueba.rename(columns={'VW_30cm': 'H_30cm'})

# Agrupar los datos por mes y calcular el valor promedio mensual
datos_mensuales = df_prueba.resample('M').mean()

# 1. Filtro de media móvil
window_size = 24
df_prueba['predicciones_fil'] = df_prueba['predicciones'].rolling(window=window_size).mean()

print(df_prueba)

# Reemplazar NaN por 1.5 en la columna 'predicciones_fil'
valor_reemplazo = 0.145365
df_prueba['predicciones_fil'].fillna(valor_reemplazo, inplace=True)

y_pred_fill_prueba = df_prueba['predicciones_fil']
y_pred_fill_prueba = y_pred_fill_prueba.tolist()


# Calcular el error cuadrático medio (MSE) en los datos de prueba
mse_prueba = mean_squared_error(y_prueba, y_pred_fill_prueba)
print(f'Mean Squared Error (prueba): {mse_prueba}')

# Calcular el error absoluto medio (MAE)
mae_prueba = mean_absolute_error(y_prueba, y_pred_fill_prueba)
print(f'Mean Absolute Error: {mae_prueba}')

columnas_a_graficar = ['H_30cm','predicciones','predicciones_fil']

# Graficar los valores mensuales
df_prueba[columnas_a_graficar].plot()
plt.xlabel('Date')
plt.ylabel('Monthly Average Value')
# plt.title('Gráfico de Valores Promedio Mensuales')
plt.legend(loc='upper right')  # Ajusta la ubicación de la leyenda
# Cambiar las leyendas (etiquetas)
plt.legend(labels=['H_30cm', 'Predictions', 'Moving Average Predictions'])
plt.show()
