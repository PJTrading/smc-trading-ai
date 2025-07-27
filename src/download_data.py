# import yfinance as yf
# import pandas as pd
# from pathlib import Path # Para manejar rutas de archivos de forma robusta

# # --- PARÁMETROS ---
# # Definimos los parámetros principales en un solo lugar para fácil modificación.
# TICKER = 'ES=F'
# START_DATE = '2019-01-01'
# END_DATE = '2023-12-31'
# INTERVAL = '1h'

# # Creamos una ruta al directorio de datos 'raw'
# # Path.cwd() es la carpeta actual (la raíz de nuestro proyecto)
# OUTPUT_PATH = Path.cwd() / "data" / "raw"
# OUTPUT_FILE = OUTPUT_PATH / f"{TICKER}_data_{INTERVAL}.csv"

# # --- LÓGICA PRINCIPAL ---
# def main():
#     """
#     Función principal para descargar y guardar los datos históricos.
#     """
#     # Asegurarnos de que el directorio de salida exista
#     OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

#     print(f"Descargando datos para {TICKER}...")
#     print(f"Periodo: {START_DATE} a {END_DATE} con intervalo {INTERVAL}")

#     try:
#         # Descargar los datos usando yfinance
#         data = yf.download(
#             tickers=TICKER,
#             start=START_DATE,
#             end=END_DATE,
#             interval=INTERVAL
#         )

#         # Comprobar si se descargaron datos
#         if data.empty:
#             print(f"Error: No se encontraron datos para el ticker {TICKER}.")
#             print("Verifica el ticker, el rango de fechas y tu conexión a internet.")
#             return

#         # Guardar los datos en el archivo CSV
#         data.to_csv(OUTPUT_FILE)

#         print("-" * 50)
#         print(f"¡Éxito! Datos guardados en: {OUTPUT_FILE}")
#         print(f"Total de registros descargados: {len(data)}")
#         print("Primeras 5 filas de los datos:")
#         print(data.head())
#         print("-" * 50)

#     except Exception as e:
#         print(f"Ocurrió un error inesperado durante la descarga: {e}")

# # Este bloque asegura que la función main() solo se ejecute
# # cuando el script es llamado directamente.
# if __name__ == "__main__":
#     main()

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

# --- PARÁMETROS ---
TICKER = 'ES=F'
INTERVAL = '1h'
DAYS_HISTORY = 729 # Yahoo permite 730 días, usamos 729 por seguridad

# La fecha de inicio se calculará automáticamente
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=DAYS_HISTORY)

# Formatear a texto para la API
start_str = START_DATE.strftime('%Y-%m-%d')
end_str = END_DATE.strftime('%Y-%m-%d')

OUTPUT_PATH = Path.cwd() / "data" / "raw"
OUTPUT_FILE = OUTPUT_PATH / f"{TICKER}_data_{INTERVAL}.csv"

# --- LÓGICA PRINCIPAL ---
def main():
    """
    Función principal para descargar los datos históricos permitidos por Yahoo.
    """
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"Descargando datos para {TICKER} ({INTERVAL})...")
    print(f"Periodo solicitado: {start_str} a {end_str} ({DAYS_HISTORY} días)")
    
    try:
        data = yf.download(
            tickers=TICKER,
            start=start_str,
            end=end_str,
            interval=INTERVAL,
            auto_adjust=True
        )

        if data.empty:
            print("Error: No se encontraron datos.")
            return

        data.to_csv(OUTPUT_FILE)
        
        print("-" * 50)
        print(f"¡Éxito! Datos guardados en: {OUTPUT_FILE}")
        print(f"Total de registros descargados: {len(data)}")
        print("Primeras 5 filas de los datos:")
        print(data.head())
        print("-" * 50)

    except Exception as e:
        print(f"Ocurrió un error inesperado durante la descarga: {e}")

if __name__ == "__main__":
    main()