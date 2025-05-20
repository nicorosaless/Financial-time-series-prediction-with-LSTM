import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_financial_data(filepath):
    """
    Carga un dataset de datos financieros desde un archivo CSV.

    Args:
        filepath (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    try:
        data = pd.read_csv(filepath)
        print("Dataset cargado exitosamente.")
        return data
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None

# --- Template para LSTM ---

# 1. Cargar los datos
# Asegúrate de que la ruta al archivo CSV sea correcta.
DATASET_PATH = "/Users/nicolasrosales/Desktop/upc/Q2/PA2/practica/Financial-time-series-prediction-with-LSTM/backend/data/dataset.csv"
data = load_financial_data(DATASET_PATH)

if data is not None:
    print("Primeras 5 filas del dataset:")
    print(data.head())

    # 2. Preprocesamiento de datos
    #   a. Seleccionar la columna relevante para la predicción (ej. 'Price').
    if 'Price' not in data.columns:
        print("Error: La columna 'Price' no se encuentra en el dataset. Por favor, verifica el nombre de la columna.")
        exit()
        
    price_values = data['Price'].values.reshape(-1,1)

    #   b. Normalizar los datos (ej. usando MinMaxScaler de sklearn).
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(price_values)

    #   c. Crear secuencias de datos para LSTM (X_train, y_train).
    training_data_len = int(np.ceil( len(scaled_data) * .80 )) # 80% para entrenamiento, 20% para prueba

    train_data = scaled_data[0:int(training_data_len), :]

    x_train = []
    y_train = []
    time_step = 60 # Número de pasos de tiempo anteriores para predecir el siguiente

    for i in range(time_step, len(train_data)):
        x_train.append(train_data[i-time_step:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # Reshape para LSTM [samples, time_steps, features]


    # 3. Construir el modelo LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))


    # 4. Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')


    # 5. Entrenar el modelo
    print("\nEntrenando el modelo LSTM...")
    # Ajusta batch_size y epochs según sea necesario y la capacidad de tu máquina.
    # Para un aprendizaje rápido inicial, se usan valores bajos.
    model.fit(x_train, y_train, batch_size=32, epochs=10) 


    # 6. Preparar datos de prueba y hacer predicciones
    test_data = scaled_data[training_data_len - time_step:, :]
    x_test = []
    y_test = price_values[training_data_len:, :] # Valores reales para comparar (sin escalar)

    for i in range(time_step, len(test_data)):
        x_test.append(test_data[i-time_step:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    print("\nHaciendo predicciones...")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions) # Des-normalizar


    # 7. Evaluar el modelo (opcional pero recomendado)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    print(f"\nRoot Mean Squared Error (RMSE): {rmse}")


    # 8. Visualizar los resultados (opcional)
    print("\nGenerando gráfico de predicciones...")
    plt.figure(figsize=(16,8))
    plt.title('Predicción del Modelo LSTM vs Datos Reales')
    plt.xlabel('Tiempo', fontsize=18)
    plt.ylabel('Precio', fontsize=18) # Modificado de 'Precio de Cierre' a 'Precio'
    
    # Asegúrate de que 'data' tenga un índice de tipo fecha si quieres graficar fechas en el eje X.
    # Si no, se usará un índice numérico.
    plot_data = data.iloc[training_data_len:].copy() # Usar .copy() para evitar SettingWithCopyWarning
    plot_data['Predictions'] = predictions

    plt.plot(data['Price'][:training_data_len], label='Datos de Entrenamiento') # Modificado de data['Close']
    plt.plot(plot_data['Price'], label='Datos Reales (Test)') # Modificado de plot_data['Close']
    plt.plot(plot_data['Predictions'], label='Predicciones (Test)')
    
    plt.legend(loc='lower right')
    plt.show()

    print("\n--- Fin del Script LSTM ---")
    print("El modelo ha sido entrenado y las predicciones se han visualizado.")
    print("Puedes ajustar los hiperparámetros (time_step, epochs, batch_size, capas del modelo) para mejorar el rendimiento.")

else:
    print("No se pudo cargar el dataset. Por favor, verifica la ruta y el archivo.")