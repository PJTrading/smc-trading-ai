# src/feature_engineering_v2.py
import pandas as pd
from pathlib import Path
from ta.volatility import average_true_range
from ta.momentum import rsi
from smartmoneyconcepts import smc

# --- PARÁMETROS ---
INPUT_PATH = Path.cwd() / "data" / "raw" / "ES=F_data_1h.csv"
OUTPUT_PATH = Path.cwd() / "data" / "processed" / "ES=F_data_1h_features_v2.csv"

# Parámetros para el Etiquetado
RISK_REWARD_RATIO = 1.5
STOP_LOSS_PCT = 0.01
HOLD_PERIOD = 48

def main():
    print("Iniciando Feature Engineering v2...")
    df = pd.read_csv(INPUT_PATH, header=[0, 1], index_col=0, parse_dates=True)
    # df.columns = df.columns.get_level_values(1).str.lower()
    # df.index.name = 'Datetime'
    df.columns = df.columns.get_level_values(0).str.lower() # Simplificar encabezado
    df.index.name = 'Datetime'
    
    # --- 1. Indicadores Técnicos Clásicos (Sin Lookahead) ---
    print("Calculando indicadores técnicos...")
    df['atr'] = average_true_range(df['high'], df['low'], df['close'], window=14)
    df['rsi'] = rsi(df['close'], window=14)
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['distance_to_ema_20'] = df['close'] - df['ema_20']

    # --- 2. Cuantificar SMC como "Estado Actual" ---
    print("Cuantificando conceptos SMC...")
    # Detectar FVG y OBs como antes
    fvg_data = smc.fvg(df)
    swing_data = smc.swing_highs_lows(df)
    ob_data = smc.ob(df, swing_highs_lows=swing_data)

    # Crear características relativas
    # Rellenamos hacia adelante para saber "cuál fue el último nivel conocido"
    df['last_fvg_top'] = fvg_data['Top'].ffill()
    df['last_ob_top'] = ob_data['Top'].ffill()
    
    # Distancia al último nivel SMC conocido
    df['distance_to_fvg'] = df['close'] - df['last_fvg_top']
    df['distance_to_ob'] = df['close'] - df['last_ob_top']
    
    # --- 3. Etiquetado del Objetivo (Sin cambios) ---
    print("Creando la etiqueta objetivo...")
    # [El código de etiquetado es idéntico al anterior, lo omito por brevedad pero debe estar aquí]
    target = []
    df_len = len(df)
    for i in range(df_len - HOLD_PERIOD):
        entry_price = df['close'].iloc[i]
        sl_price = entry_price * (1 - STOP_LOSS_PCT)
        tp_price = entry_price * (1 + (STOP_LOSS_PCT * RISK_REWARD_RATIO))
        outcome = None
        for j in range(1, HOLD_PERIOD + 1):
            if df['low'].iloc[i+j] <= sl_price:
                outcome = 0; break
            if df['high'].iloc[i+j] >= tp_price:
                outcome = 1; break
        target.append(outcome)
    target.extend([None] * HOLD_PERIOD)
    df['target'] = target

    # --- 4. Limpieza y Guardado ---
    # Seleccionar solo las características finales que usaremos
    final_features = [
        'open', 'high', 'low', 'close', 'volume', 'target',
        'atr', 'rsi', 'distance_to_ema_20', 'distance_to_fvg', 'distance_to_ob'
    ]
    df_final = df[final_features].copy()
    
    # Guardar
    df_final.to_csv(OUTPUT_PATH)
    print(f"¡Éxito! Nuevo DataFrame v2 guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()