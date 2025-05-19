import pandas as pd

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