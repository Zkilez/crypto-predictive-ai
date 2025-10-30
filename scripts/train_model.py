import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, brier_score_loss, balanced_accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import pickle
from datetime import datetime
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import requests
from textblob import TextBlob

# --- Métrica de calibración: Expected Calibration Error (ECE) ---
def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        in_bin = (proba >= bin_lower) & (proba < bin_upper)
        if np.any(in_bin):
            acc_bin = np.mean(y_true[in_bin] == (proba[in_bin] >= 0.5).astype(int))
            conf_bin = np.mean(proba[in_bin])
            weight = np.mean(in_bin)
            ece += weight * abs(acc_bin - conf_bin)
    return float(ece)

def calculate_technical_indicators(df):
    """
    Calcula indicadores técnicos para el modelo
    """
    # RSI (Relative Strength Index)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['ma_7'] = df['price'].rolling(window=7).mean()
    df['ma_21'] = df['price'].rolling(window=21).mean()
    df['ma_50'] = df['price'].rolling(window=50).mean()
    
    # EMAs y MACD
    df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volatilidad
    df['volatility'] = df['price'].rolling(window=7).std()
    
    # Análisis de tendencia de mercado
    # Tendencia a corto plazo (7 días)
    df['trend_short'] = (df['ma_7'] > df['ma_7'].shift(3)).astype(int)
    
    # Tendencia a medio plazo (21 días)
    df['trend_medium'] = (df['ma_21'] > df['ma_21'].shift(7)).astype(int)
    
    # Tendencia a largo plazo (50 días)
    df['trend_long'] = (df['ma_50'] > df['ma_50'].shift(14)).astype(int)
    
    # Fuerza de la tendencia (combinación de tendencias)
    df['trend_strength'] = df['trend_short'] + df['trend_medium'] + df['trend_long']
    df['volatility_14'] = df['price'].rolling(window=14).std()
    
    # Cambios de precio y retornos pasados
    df['price_change'] = df['price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['return_1d'] = df['price'].pct_change(periods=1)
    df['return_3d'] = df['price'].pct_change(periods=3)
    df['return_7d'] = df['price'].pct_change(periods=7)
    
    # Momentum
    df['momentum'] = df['price'] - df['price'].shift(4)
    df['roc_7'] = df['price'].pct_change(periods=7) * 100  # Rate of Change (7 días)
    df['roc_14'] = df['price'].pct_change(periods=14) * 100  # Rate of Change (14 días)
    
    # Bollinger Bands
    df['bb_middle'] = df['price'].rolling(window=20).mean()
    df['bb_std'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Análisis de sentimiento de mercado (simulado con datos técnicos)
    df['market_sentiment'] = 0
    # Sentimiento positivo cuando RSI > 50 y tendencias son positivas
    df.loc[(df['rsi'] > 50) & (df['trend_short'] > 0) & (df['trend_medium'] > 0), 'market_sentiment'] = 1
    # Sentimiento negativo cuando RSI < 50 y tendencias son negativas
    df.loc[(df['rsi'] < 50) & (df['trend_short'] < 0) & (df['trend_medium'] < 0), 'market_sentiment'] = -1
    
    return df

def prepare_features(df):
    """
    Prepara las características para el modelo
    """
    df = calculate_technical_indicators(df)
    
    # Target: 1 si el precio sube en los próximos 3 días, 0 si baja
    df['future_price'] = df['price'].shift(-3)
    df['target'] = (df['future_price'] > df['price']).astype(int)
    
    # Eliminar filas con NaN
    df = df.dropna()
    
    # Seleccionar características
    features = [
        'rsi', 'ma_7', 'ma_21', 'ma_50', 'ema_12', 'ema_26', 'macd',
        'volatility', 'volatility_14',
        'price_change', 'volume_change', 'return_1d', 'return_3d', 'return_7d',
        'momentum',
        'bb_upper', 'bb_lower', 'volume', 'market_cap'
    ]
    
    X = df[features]
    y = df['target']
    
    return X, y, features

def train_model(crypto_data):
    """
    Entrena el modelo de predicción con modelos separados por símbolo
    """
    print(f"\n{'='*60}")
    print("ENTRENANDO MODELO DE PREDICCIÓN DE CRIPTOMONEDAS")
    print(f"{'='*60}\n")

    # Diccionarios para almacenar modelos y métricas por símbolo
    models_per_symbol = {}
    scalers_per_symbol = {}
    metrics_per_symbol = {}
    calibrators_per_symbol = {}
    thresholds_per_symbol = {}
    
    features = None
    all_features_importances = {}

    # Procesar datos y entrenar modelos separados por símbolo
    for symbol, data in crypto_data.items():
        print(f"\n{'='*60}")
        print(f"PROCESANDO {symbol}")
        print(f"{'='*60}\n")
        
        df = pd.DataFrame(data)
        df['price'] = pd.to_numeric(df['price'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['market_cap'] = pd.to_numeric(df['market_cap'])

        X, y, features_local = prepare_features(df)
        if features is None:
            features = features_local

        n = len(X)
        train_end = int(n * 0.7)
        calib_end = int(n * 0.8)
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_calib = X.iloc[train_end:calib_end]
        y_calib = y.iloc[train_end:calib_end]
        X_test = X.iloc[calib_end:]
        y_test = y.iloc[calib_end:]
        
        print(f"Datos para {symbol}: {len(X_train)} train / {len(X_calib)} calibración / {len(X_test)} test muestras")
        print(f"Distribución de clases (train): Subida {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%) / Bajada {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")

        # Normalizar características específicas para este símbolo
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_calib_scaled = scaler.transform(X_calib)
        X_test_scaled = scaler.transform(X_test)
        
        # Guardar el scaler específico para este símbolo
        scalers_per_symbol[symbol] = scaler
        
        print(f"\nEntrenando modelos para {symbol}...")
        
        # Optimización de hiperparámetros para este símbolo
        print(f"Optimizando Gradient Boosting para {symbol} con GridSearchCV...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'min_samples_split': [5, 10]
        }
        
        # Crear el modelo base
        gb_base = GradientBoostingClassifier(random_state=42)
        
        # Configurar validación cruzada temporal más robusta (10 splits con gap)
        tscv = TimeSeriesSplit(n_splits=10, gap=3)
        
        # Crear GridSearchCV
        gb_grid = GridSearchCV(
            estimator=gb_base,
            param_grid=param_grid,
            cv=tscv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Ajustar con pesos de clase para manejar desbalance
        gb_sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        gb_grid.fit(X_train_scaled, y_train, sample_weight=gb_sample_weight)
        
        # Mostrar los mejores parámetros
        print(f"\nMejores parámetros para {symbol}: {gb_grid.best_params_}")
        print(f"Mejor balanced_accuracy en CV: {gb_grid.best_score_:.3f}")
        
        # Usar los mejores parámetros para el modelo final
        best_params = gb_grid.best_params_
        gb_model = GradientBoostingClassifier(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            subsample=best_params['subsample'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42
        )
        
        # Crear modelos adicionales para el ensemble
        rf_model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=7,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        svm_model = SVC(
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        
        # Crear sistema de consenso (Stacking) en lugar de votación simple
        stacking_model = StackingClassifier(
            estimators=[
                ('gb', gb_model),
                ('rf', rf_model),
                ('svm', svm_model)
            ],
            final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
            cv=5,
            n_jobs=-1
        )
        
        # Entrenar también un Random Forest como comparación
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Entrenar todos los modelos
        stacking_model.fit(X_train_scaled, y_train)
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train, sample_weight=gb_sample_weight)
        
        # Evaluar en test
        stacking_pred = stacking_model.predict(X_test_scaled)
        stacking_accuracy = accuracy_score(y_test, stacking_pred)
        stacking_bal_acc = balanced_accuracy_score(y_test, stacking_pred)
        stacking_f1 = f1_score(y_test, stacking_pred, average='binary')
        
        rf_pred = rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_bal_acc = balanced_accuracy_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred, average='binary')
        
        gb_pred = gb_model.predict(X_test_scaled)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        gb_bal_acc = balanced_accuracy_score(y_test, gb_pred)
        gb_f1 = f1_score(y_test, gb_pred, average='binary')
        
        print(f"\nStacking para {symbol}:")
        print(f"  Accuracy: {stacking_accuracy*100:.2f}% | BalAcc: {stacking_bal_acc:.3f} | F1: {stacking_f1:.3f}")
        
        print(f"Random Forest para {symbol}:")
        print(f"  Accuracy: {rf_accuracy*100:.2f}% | BalAcc: {rf_bal_acc:.3f} | F1: {rf_f1:.3f}")
        
        print(f"Gradient Boosting para {symbol}:")
        print(f"  Accuracy: {gb_accuracy*100:.2f}% | BalAcc: {gb_bal_acc:.3f} | F1: {gb_f1:.3f}")
        
        # Seleccionar el mejor modelo para este símbolo basado en F1
        models = {
            "Stacking": (stacking_model, stacking_f1, stacking_bal_acc),
            "Gradient Boosting": (gb_model, gb_f1, gb_bal_acc),
            "Random Forest": (rf_model, rf_f1, rf_bal_acc)
        }
        
        best_name = max(models.items(), key=lambda x: x[1][1])[0]
        best_model = models[best_name][0]
        best_f1 = models[best_name][1]
        best_bal_acc = models[best_name][2]
            
        print(f"\nMejor modelo para {symbol}: {best_name} (F1: {best_f1:.3f})")
        
        # Guardar importancia de características
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            all_features_importances[symbol] = importances.tolist()
            
            # Mostrar top 5 características
            print(f"Top 5 características más importantes para {symbol}:")
            indices = np.argsort(importances)[::-1][:5]
            for i, idx in enumerate(indices, 1):
                print(f"  {i}. {features[idx]}: {importances[idx]:.4f}")
        
        # Calibrar el modelo
        print(f"\nCalibrando modelo para {symbol}...")
        best_calibrator = None
        best_method = None
        best_brier = float('inf')
        
        for method in ['sigmoid', 'isotonic']:
            try:
                cal = CalibratedClassifierCV(estimator=best_model, method=method, cv='prefit')
                cal.fit(X_calib_scaled, y_calib)
                proba_calib = cal.predict_proba(X_calib_scaled)[:, 1]
                brier = brier_score_loss(y_calib, proba_calib)
                
                if brier < best_brier:
                    best_brier = brier
                    best_method = method
                    best_calibrator = cal
            except Exception as e:
                print(f"  [!] Falló calibración {method} para {symbol}: {e}")
        
        if best_calibrator is not None:
            print(f"  Mejor método de calibración para {symbol}: {best_method} (Brier: {best_brier:.3f})")
            calibrators_per_symbol[symbol] = best_calibrator
            
            # Optimizar umbral para F1
            proba_calib = best_calibrator.predict_proba(X_calib_scaled)[:, 1]
            best_f1_score = -1.0
            best_threshold = 0.5
            
            for t in np.linspace(0.05, 0.95, 19):
                y_pred = (proba_calib >= t).astype(int)
                f1_t = f1_score(y_calib, y_pred, average='binary')
                if f1_t > best_f1_score:
                    best_f1_score = f1_t
                    best_threshold = float(t)
            
            print(f"  Umbral óptimo para {symbol}: {best_threshold:.2f} (F1: {best_f1_score:.3f})")
            
            # Evaluar en test con el umbral optimizado
            proba_test = best_calibrator.predict_proba(X_test_scaled)[:, 1]
            y_pred_thresh = (proba_test >= best_threshold).astype(int)
            acc_thresh = accuracy_score(y_test, y_pred_thresh)
            bal_acc_thresh = balanced_accuracy_score(y_test, y_pred_thresh)
            f1_thresh = f1_score(y_test, y_pred_thresh, average='binary')
            
            print(f"  Métricas finales para {symbol} (umbral {best_threshold:.2f}):")
            print(f"    Accuracy: {acc_thresh*100:.2f}% | BalAcc: {bal_acc_thresh:.3f} | F1: {f1_thresh:.3f}")
            
            # Guardar umbral y métricas
            thresholds_per_symbol[symbol] = {
                'f1': best_threshold,
                'balanced_accuracy': best_threshold  # Usamos el mismo por simplicidad
            }
            
            metrics_per_symbol[symbol] = {
                'accuracy': float(acc_thresh),
                'balanced_accuracy': float(bal_acc_thresh),
                'f1': float(f1_thresh),
                'brier': float(best_brier),
                'method': best_method,
                'threshold': float(best_threshold)
            }
            
            # Guardar el modelo calibrado
            models_per_symbol[symbol] = best_calibrator
        else:
            print(f"  [!] No se pudo calibrar el modelo para {symbol}, usando modelo sin calibrar")
            models_per_symbol[symbol] = best_model
            
            # Evaluar en test con umbral por defecto
            if hasattr(best_model, 'predict_proba'):
                proba_test = best_model.predict_proba(X_test_scaled)[:, 1]
                y_pred = (proba_test >= 0.5).astype(int)
                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='binary')
                
                print(f"  Métricas finales para {symbol} (umbral 0.5, sin calibrar):")
                print(f"    Accuracy: {acc*100:.2f}% | BalAcc: {bal_acc:.3f} | F1: {f1:.3f}")
                
                thresholds_per_symbol[symbol] = {
                    'f1': 0.5,
                    'balanced_accuracy': 0.5
                }
                
                metrics_per_symbol[symbol] = {
                    'accuracy': float(acc),
                    'balanced_accuracy': float(bal_acc),
                    'f1': float(f1),
                    'method': 'none',
                    'threshold': 0.5
                }
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN DE MODELOS POR SÍMBOLO")
    print(f"{'='*60}\n")
    
    for symbol, metrics in metrics_per_symbol.items():
        print(f"{symbol}:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        print(f"  Método de calibración: {metrics.get('method', 'none')}")
        print(f"  Umbral óptimo: {metrics['threshold']:.2f}")
        print()
    
    # Guardar todos los modelos y datos relacionados
    model_data = {
        'models_per_symbol': models_per_symbol,
        'scalers_per_symbol': scalers_per_symbol,
        'features': features,
        'trained_date': datetime.now().isoformat(),
        'feature_importances_per_symbol': all_features_importances,
        'calibrated': True,
        'split_strategy': 'per-symbol chronological 70/10/20 (prefit calibration)',
        'per_symbol_thresholds': {
            s: {'balanced_accuracy': float(t['balanced_accuracy']), 'f1': float(t['f1'])}
            for s, t in thresholds_per_symbol.items()
        },
        'per_symbol_metrics': metrics_per_symbol,
        'min_confidence_threshold': 0.80  # Umbral mínimo de confianza para filtrar predicciones poco fiables
    }

    with open('crypto_prediction_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✓ Modelos guardados en crypto_prediction_model.pkl")
    print(f"✓ Entrenamiento completado exitosamente!")

def main():
    # Cargar datos históricos
    try:
        with open('crypto_historical_data.json', 'r') as f:
            crypto_data = json.load(f)
        
        print(f"✓ Datos cargados: {len(crypto_data)} criptomonedas")
        
        # Entrenar modelo
        train_model(crypto_data)
        
    except FileNotFoundError:
        print("Error: No se encontró crypto_historical_data.json")
        print("Ejecuta primero: fetch_crypto_data.py")
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")

if __name__ == "__main__":
    main()
