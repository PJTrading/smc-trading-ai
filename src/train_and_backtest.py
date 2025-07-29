import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import vectorbt as vbt

# --- PARÁMETROS ---
INPUT_FILE = Path.cwd() / "data" / "processed" / "ES=F_data_1h_features_v2.csv"
PROBABILITY_THRESHOLD = 0.52

def main():
    print("Iniciando Fase 3: Entrenamiento y Backtesting (Versión con limpieza corregida)...")

    # --- 1. Cargar Datos ---
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    df.index.name = 'Datetime'
    
    # --- 2. Preparar Características y Objetivo ---
    leaky_features = [col for col in df.columns if 'MitigatedIndex' in col or 'BrokenIndex' in col or 'Swept' in col]
    features = df.columns.drop(['open', 'high', 'low', 'close', 'volume', 'target'] + leaky_features)
    
    X = df[features].copy()
    y = df['target'].copy()

    # --- 3. Limpieza y Alineación CORRECTA ---
    # Paso 3.1: Primero, eliminar las filas donde el OBJETIVO es nulo.
    y_cleaned = y.dropna()
    
    # Paso 3.2: Alinear X para que tenga las mismas filas que y_cleaned.
    X_aligned = X.loc[y_cleaned.index]

    # Paso 3.3: AHORA, aplicar el shift a las características alineadas para evitar el look-ahead.
    # X_shifted = X_aligned.shift(1)
    
    # Paso 3.4: Rellenar los NaNs que queden en las características (del shift y de la propia librería).
    # Usamos fillna(0) asumiendo que un NaN en una característica significa que el patrón no está presente.
    # X_final = X_shifted.fillna(0)
    X_final = X_aligned.fillna(0)

    # Paso 3.5: Finalmente, eliminar la primera fila que ahora es NaN en 'y' debido al shift de X.
    # Al hacer esto, X e y quedan perfectamente sincronizados.
    y_final = y_cleaned.loc[X_final.index].dropna()
    X_final = X_final.loc[y_final.index]

    # Reemplazar infinitos si los hubiera
    X_final.replace([np.inf, -np.inf], 0, inplace=True)

    print(f"Datos limpios y alineados. {len(X_final)} muestras para entrenar y probar.")
    
    # --- 4. División de Datos Cronológica ---
    train_size = int(len(X_final) * 0.8)
    X_train, X_test = X_final.iloc[:train_size], X_final.iloc[train_size:]
    y_train, y_test = y_final.iloc[:train_size], y_final.iloc[train_size:]

    # --- 5. Escalar y Entrenar ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Entrenando el modelo RandomForest (con datos honestos y retrasados)...")
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # --- 6. Evaluación Honesta ---
    print("\n--- Reporte de Clasificación (Honesto) ---")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, zero_division=0)) # Añadido zero_division=0 para evitar warnings
    
    # --- 7. Backtesting ---
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    
    price_data_for_backtest = df.loc[X_test.index]
    price_data_for_backtest['signal_prob'] = probabilities
    
    print("\nEjecutando backtest con vectorbt...")
    price = price_data_for_backtest['close']
    entries = price_data_for_backtest['signal_prob'] > PROBABILITY_THRESHOLD
    exits = price_data_for_backtest['signal_prob'] < (1 - PROBABILITY_THRESHOLD)

    pf = vbt.Portfolio.from_signals(
        price, entries, exits, freq='H', sl_stop=0.01, tp_stop=0.015, fees=0.0006, slippage=0.0002
    )
    
    print("\n--- Resultados del Backtest ---")
    print(pf.stats())
    # pf.plot().show()
    print("¡Fase 3 completada!")

if __name__ == "__main__":
    main()


