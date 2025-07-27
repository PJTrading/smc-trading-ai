import pandas as pd
from pathlib import Path
from smartmoneyconcepts import smc

# --- PARÁMETROS ---
# Rutas de entrada y salida
INPUT_PATH = Path.cwd() / "data" / "raw" / "ES=F_data_1h.csv"
OUTPUT_PATH = Path.cwd() / "data" / "processed" / "ES=F_data_1h_features.csv"

# Parámetros para el Etiquetado de Objetivo
# Objetivo: predecir si el precio subirá un 1.5% antes de bajar un 1%
RISK_REWARD_RATIO = 1.5
STOP_LOSS_PCT = 0.01  # 1%
HOLD_PERIOD = 48      # ¿En las próximas 48 horas?

def main():
    """
    Función principal para cargar los datos, añadir características SMC y etiquetar el objetivo.
    """
    print("Iniciando Fase 2: Ingeniería de Características y Etiquetado...")
    
    # --- 1. Cargar Datos Limpios ---
    print(f"Cargando datos desde {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(0) # Simplificar encabezado
    df.index.name = 'Datetime'
    
    # La librería SMC espera nombres de columna en minúsculas. ¡Paso crucial!
    df.columns = [col.lower() for col in df.columns]

    print("Columnas del DataFrame limpias y listas para SMC:", df.columns.tolist())
    print("DataFrame:", df.head())

    # --- 2. Ingeniería de Características SMC ---
    print("Generando características de Smart Money Concepts...")
    
    # Detectar Swings (necesario para OB, BoS/ChoCH)
    swing_data = smc.swing_highs_lows(df, swing_length=20)
    df = pd.concat([df, swing_data], axis=1)

    # Detectar Order Blocks (OB)
    ob_data = smc.ob(df, swing_highs_lows=swing_data, close_mitigation=False)
    df = pd.concat([df, ob_data], axis=1)
    
    # Detectar Fair Value Gaps (FVG)
    fvg_data = smc.fvg(df)
    df = pd.concat([df, fvg_data], axis=1)

    # Detectar Break of Structure (BoS) y Change of Character (ChoCH)
    bos_choch_data = smc.bos_choch(df, swing_highs_lows=swing_data)
    df = pd.concat([df, bos_choch_data], axis=1)
    
    print("Características SMC generadas.")
    
    # --- 3. Etiquetado de Objetivo (Target Labeling) ---
    print(f"Creando la etiqueta objetivo (target) con R:R={RISK_REWARD_RATIO}...")
    
    target = []
    df_len = len(df)
    
    for i in range(df_len - HOLD_PERIOD):
        entry_price = df['close'].iloc[i]
        sl_price = entry_price * (1 - STOP_LOSS_PCT)
        tp_price = entry_price * (1 + (STOP_LOSS_PCT * RISK_REWARD_RATIO))
        
        outcome = None # Usaremos None en lugar de NaN para más claridad
        
        # Mirar en el futuro dentro del periodo de mantenimiento
        for j in range(1, HOLD_PERIOD + 1):
            future_low = df['low'].iloc[i+j]
            future_high = df['high'].iloc[i+j]
            
            # Comprobar si se toca el Stop Loss
            if future_low <= sl_price:
                outcome = 0 # Fracaso
                break
            
            # Comprobar si se toca el Take Profit
            if future_high >= tp_price:
                outcome = 1 # Éxito
                break
        
        target.append(outcome)

    # Añadir padding al final para que la longitud coincida con el DataFrame
    target.extend([None] * HOLD_PERIOD)
    df['target'] = target
    
    print("Etiquetado de objetivo completado.")
    
    # --- 4. Guardar el DataFrame final ---
    # Asegurarse de que el directorio de salida exista
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    
    print("-" * 50)
    print(f"¡Éxito! DataFrame enriquecido guardado en: {OUTPUT_PATH}")
    print(f"Nuevas dimensiones del DataFrame: {df.shape}")
    print("Distribución del target (1=Éxito, 0=Fracaso):")
    print(df['target'].value_counts(dropna=True))
    print("-" * 50)


if __name__ == "__main__":
    main()