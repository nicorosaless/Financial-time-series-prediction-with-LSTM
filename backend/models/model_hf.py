from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset if it's private or requires authentication.
print("Intentando cargar el dataset 'WinkingFace/CryptoLM-Ripple-XRP-USDT' desde Hugging Face...")

try:
    ds = load_dataset("WinkingFace/CryptoLM-Ripple-XRP-USDT")
    print("\nDataset cargado exitosamente!")
    print("Estructura del dataset:")
    print(ds)

    # --- LSTM Model Implementation ---
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    import math
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    # 1. Extraer y Preprocesar los datos
    # Asumimos que el dataset tiene un split 'train' y una columna 'close' para el precio.
    # Esto podría necesitar ajuste basado en la estructura real del dataset.
    if 'train' in ds and 'close' in ds['train'].features:
        price_data_hf = np.array(ds['train']['close']).reshape(-1, 1)
        print(f"\nForma de los datos de precios extraídos: {price_data_hf.shape}")

        #   a. Normalizar los datos
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data_hf = scaler.fit_transform(price_data_hf)

        #   b. Crear secuencias de datos para LSTM
        training_data_len_hf = int(np.ceil(len(scaled_data_hf) * .80)) # 80% para entrenamiento

        train_data_hf = scaled_data_hf[0:int(training_data_len_hf), :]

        x_train_hf = []
        y_train_hf = []
        time_step_hf = 60 # Número de pasos de tiempo anteriores

        for i in range(time_step_hf, len(train_data_hf)):
            x_train_hf.append(train_data_hf[i-time_step_hf:i, 0])
            y_train_hf.append(train_data_hf[i, 0])
        
        x_train_hf, y_train_hf = np.array(x_train_hf), np.array(y_train_hf)
        x_train_hf = np.reshape(x_train_hf, (x_train_hf.shape[0], x_train_hf.shape[1], 1))

        # 2. Construir el modelo LSTM
        model_hf = Sequential()
        model_hf.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_hf.shape[1], 1)))
        model_hf.add(Dropout(0.2))
        model_hf.add(LSTM(units=50, return_sequences=False))
        model_hf.add(Dropout(0.2))
        model_hf.add(Dense(units=25))
        model_hf.add(Dense(units=1))

        # 3. Compilar el modelo
        model_hf.compile(optimizer='adam', loss='mean_squared_error')

        # 4. Entrenar el modelo
        print("\nEntrenando el modelo LSTM con datos de Hugging Face...")
        model_hf.fit(x_train_hf, y_train_hf, batch_size=32, epochs=10) # Ajusta batch_size y epochs

        # 5. Preparar datos de prueba y hacer predicciones
        test_data_hf = scaled_data_hf[training_data_len_hf - time_step_hf:, :]
        x_test_hf = []
        y_test_hf_actual = price_data_hf[training_data_len_hf:, :] # Valores reales sin escalar

        for i in range(time_step_hf, len(test_data_hf)):
            x_test_hf.append(test_data_hf[i-time_step_hf:i, 0])

        x_test_hf = np.array(x_test_hf)
        x_test_hf = np.reshape(x_test_hf, (x_test_hf.shape[0], x_test_hf.shape[1], 1 ))

        print("\nHaciendo predicciones...")
        predictions_hf = model_hf.predict(x_test_hf)
        predictions_hf = scaler.inverse_transform(predictions_hf) # Des-normalizar

        # 6. Evaluar el modelo
        if len(y_test_hf_actual) == len(predictions_hf):
            rmse_hf = math.sqrt(mean_squared_error(y_test_hf_actual, predictions_hf))
            print(f"\nRoot Mean Squared Error (RMSE) para Hugging Face data: {rmse_hf}")
        else:
            print("\nNo se pudo calcular RMSE: las longitudes de y_test_hf_actual y predictions_hf no coinciden.")
            print(f"Longitud y_test_hf_actual: {len(y_test_hf_actual)}, Longitud predictions_hf: {len(predictions_hf)}")


        # 7. Visualizar los resultados
        print("\nGenerando gráfico de predicciones para Hugging Face data...")
        
        # Crear un DataFrame para facilitar el ploteo, similar a model.py
        # Necesitamos un índice para el eje X. Si no hay fechas, usamos un rango numérico.
        # La longitud total de los datos originales es len(price_data_hf)
        
        # Datos de entrenamiento (80% de los datos originales)
        training_plot_len = training_data_len_hf 
        
        # Datos de validación/prueba (los restantes 20%)
        # El índice para los datos de prueba comienza después de los datos de entrenamiento
        validation_indices = np.arange(training_plot_len, len(price_data_hf))

        plt.figure(figsize=(16,8))
        plt.title('Predicción del Modelo LSTM (Datos de Hugging Face)')
        plt.xlabel('Índice de Tiempo', fontsize=18)
        plt.ylabel('Precio (Columna \'close\')', fontsize=18)

        # Plotear datos de entrenamiento (primeros 80%)
        plt.plot(np.arange(training_plot_len), price_data_hf[:training_plot_len, 0], label='Datos de Entrenamiento (80%)')

        # Plotear datos reales de prueba
        # Asegurarse de que y_test_hf_actual y predictions_hf tengan la misma longitud para el ploteo
        # y que los índices coincidan.
        # La longitud de y_test_hf_actual es len(price_data_hf) - training_data_len_hf
        # La longitud de predictions_hf debería ser la misma.
        
        if len(y_test_hf_actual) == len(predictions_hf):
            plt.plot(validation_indices, y_test_hf_actual[:, 0], label='Datos Reales (Test)')
            plt.plot(validation_indices, predictions_hf[:, 0], label='Predicciones (Test)')
        else:
            print("No se pueden plotear los datos de prueba y predicciones debido a longitudes diferentes.")

        plt.legend(loc='lower right')
        plt.show()

    else:
        print("\nNo se pudo encontrar el split 'train' o la columna 'close' en el dataset de Hugging Face.")
        print("Por favor, verifica la estructura del dataset y ajusta el código si es necesario.")

except Exception as e:
    print(f"\nError al cargar el dataset: {e}")
    print("Asegúrate de haber iniciado sesión con 'huggingface-cli login' si el dataset lo requiere.")
    print("También verifica que la biblioteca 'datasets' esté instalada (pip install datasets).")

print("\n--- Fin del script model_hf.py ---")
