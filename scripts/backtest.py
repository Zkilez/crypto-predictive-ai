import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import os

def calculate_technical_indicators(df):
    """
    Calcula indicadores técnicos (misma función que en train_model.py y predict.py)
    """
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['ma_7'] = df['price'].rolling(window=7).mean()
    df['ma_21'] = df['price'].rolling(window=21).mean()
    df['ma_50'] = df['price'].rolling(window=50).mean()
    
    # EMAs y MACD
    df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    
    # Volatilidad
    df['volatility'] = df['price'].rolling(window=7).std()
    df['volatility_14'] = df['price'].rolling(window=14).std()
    
    # Cambios de precio y retornos pasados
    df['price_change'] = df['price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['return_1d'] = df['price'].pct_change(periods=1)
    df['return_3d'] = df['price'].pct_change(periods=3)
    df['return_7d'] = df['price'].pct_change(periods=7)
    
    # Momentum
    df['momentum'] = df['price'] - df['price'].shift(4)
    df['roc_7'] = df['price'].pct_change(periods=7) * 100
    df['roc_14'] = df['price'].pct_change(periods=14) * 100
    
    # Bandas de Bollinger
    df['bb_middle'] = df['price'].rolling(window=20).mean()
    df['bb_std'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # Análisis de tendencia de mercado
    df['trend_short'] = (df['ma_7'] > df['ma_7'].shift(3)).astype(int)
    df['trend_medium'] = (df['ma_21'] > df['ma_21'].shift(7)).astype(int)
    df['trend_long'] = (df['ma_50'] > df['ma_50'].shift(14)).astype(int)
    df['trend_strength'] = df['trend_short'] + df['trend_medium'] + df['trend_long']
    
    # Análisis de volatilidad avanzado
    df['volatility_norm'] = df['price'].rolling(window=30).std() / df['price'].rolling(window=30).mean()
    df['volatility_relative'] = df['volatility'] / df['volatility'].rolling(window=90).mean()
    
    # Filtro de señales basado en volatilidad
    df['volatility_signal'] = 0  # Neutral por defecto
    df.loc[df['volatility_relative'] > 1.5, 'volatility_signal'] = -1  # Alta volatilidad
    df.loc[df['volatility_relative'] < 0.5, 'volatility_signal'] = 1  # Baja volatilidad
    
    # Análisis de sentimiento de mercado (simulado)
    df['market_sentiment'] = 0  # Neutral por defecto
    df.loc[(df['rsi'] > 60) & (df['trend_short'] > 0) & (df['trend_medium'] > 0), 'market_sentiment'] = 1
    df.loc[(df['rsi'] < 40) & (df['trend_short'] < 0) & (df['trend_medium'] < 0), 'market_sentiment'] = -1
    
    return df

def create_target(df, forward_days=1):
    """
    Crea la variable objetivo basada en el precio futuro
    """
    df['future_price'] = df['price'].shift(-forward_days)
    df['target'] = (df['future_price'] > df['price']).astype(int)
    return df

def run_backtest(start_date=None, end_date=None, test_period_days=30, threshold_metric='f1'):
    """
    Ejecuta un backtest del modelo en un período específico
    """
    print(f"\n{'='*60}")
    print("EJECUTANDO BACKTEST DEL MODELO")
    print(f"{'='*60}\n")
    
    # Cargar modelo
    try:
        with open('crypto_prediction_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Verificar si estamos usando modelos separados por símbolo
        models_per_symbol = model_data.get('models_per_symbol', {})
        scalers_per_symbol = model_data.get('scalers_per_symbol', {})
        features = model_data['features']
        per_symbol_thresholds = model_data.get('per_symbol_thresholds', {})
        min_confidence_threshold = model_data.get('min_confidence_threshold', 0.75)
        
        # Determinar si estamos usando modelos por símbolo
        using_per_symbol_models = len(models_per_symbol) > 0
        
        if using_per_symbol_models:
            print(f"✓ Modelos separados por símbolo cargados")
            print(f"✓ Umbral mínimo de confianza: {min_confidence_threshold:.2f}")
            print(f"✓ Métrica de umbral seleccionada: {threshold_metric}")
            print(f"✓ Entrenado el: {model_data['trained_date']}\n")
        else:
            print("Error: Este script de backtest requiere modelos separados por símbolo")
            return
        
    except FileNotFoundError:
        print("Error: No se encontró crypto_prediction_model.pkl")
        print("Ejecuta primero: train_model.py")
        return
    
    # Cargar datos históricos
    try:
        with open('crypto_historical_data.json', 'r') as f:
            crypto_data = json.load(f)
    except FileNotFoundError:
        print("Error: No se encontró crypto_historical_data.json")
        return
    
    # Configurar fechas de backtest
    if end_date is None:
        # Usar la fecha más reciente disponible
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    if start_date is None:
        # Por defecto, usar un período de test_period_days antes de end_date
        start_date = end_date - timedelta(days=test_period_days)
    else:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    print(f"Período de backtest: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
    
    # Resultados por símbolo
    backtest_results = {}
    
    # Crear directorio para gráficos si no existe
    os.makedirs('backtest_results', exist_ok=True)
    
    for symbol, data in crypto_data.items():
        if symbol not in models_per_symbol:
            print(f"Omitiendo {symbol}: No hay modelo entrenado para este símbolo")
            continue
            
        print(f"\nBacktesting {symbol}...")
        
        # Convertir datos a DataFrame
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Convertir columnas numéricas
        df['price'] = pd.to_numeric(df['price'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['market_cap'] = pd.to_numeric(df['market_cap'])
        
        # Filtrar por período de backtest
        df = df.loc[start_date:end_date]
        
        if len(df) < 50:
            print(f"  [!] Datos insuficientes para {symbol}, omitiendo...")
            continue
        
        # Calcular indicadores técnicos
        df = calculate_technical_indicators(df)
        
        # Crear variable objetivo
        df = create_target(df)
        
        # Eliminar filas con NaN
        df.dropna(inplace=True)
        
        if len(df) < 10:
            print(f"  [!] Datos insuficientes después de calcular indicadores para {symbol}, omitiendo...")
            continue
        
        # Obtener modelo y scaler para este símbolo
        symbol_model = models_per_symbol[symbol]
        symbol_scaler = scalers_per_symbol[symbol]
        
        # Obtener umbral específico para este símbolo
        symbol_threshold = 0.5
        if symbol in per_symbol_thresholds:
            symbol_thresholds = per_symbol_thresholds[symbol]
            if isinstance(symbol_thresholds, dict) and threshold_metric in symbol_thresholds:
                symbol_threshold = symbol_thresholds[threshold_metric]
            elif not isinstance(symbol_thresholds, dict):
                symbol_threshold = symbol_thresholds
        
        # Extraer características
        X = df[features]
        
        # Escalar características
        X_scaled = symbol_scaler.transform(X)
        
        # Hacer predicciones
        y_proba = symbol_model.predict_proba(X_scaled)[:, 1]
        y_pred_raw = (y_proba >= symbol_threshold).astype(int)
        
        # Aplicar filtros de confianza y volatilidad
        y_pred = y_pred_raw.copy()
        confidence = np.where(y_pred == 1, y_proba, 1 - y_proba)
        
        # Ajustar predicciones basadas en volatilidad y sentimiento
        for i in range(len(df)):
            # Ajustar por volatilidad
            if 'volatility_signal' in df.columns:
                volatility_signal = df['volatility_signal'].iloc[i]
                if volatility_signal == -1:  # Alta volatilidad
                    confidence[i] *= 0.8
                elif volatility_signal == 1:  # Baja volatilidad
                    confidence[i] *= 1.1
            
            # Ajustar por sentimiento de mercado
            if 'market_sentiment' in df.columns:
                market_sentiment = df['market_sentiment'].iloc[i]
                if (market_sentiment == -1 and y_pred[i] == 1) or (market_sentiment == 1 and y_pred[i] == 0):
                    confidence[i] *= 0.9
                elif (market_sentiment == 1 and y_pred[i] == 1) or (market_sentiment == -1 and y_pred[i] == 0):
                    confidence[i] *= 1.1
            
            # Limitar confianza a 1.0
            confidence[i] = min(confidence[i], 1.0)
            
            # Aplicar umbral de confianza mínima
            if confidence[i] < min_confidence_threshold:
                y_pred[i] = -1  # Marcar como "sin predicción" debido a baja confianza
        
        # Calcular métricas solo para predicciones con alta confianza
        high_confidence_mask = y_pred != -1
        if sum(high_confidence_mask) > 0:
            y_true_high_conf = df['target'].values[high_confidence_mask]
            y_pred_high_conf = y_pred[high_confidence_mask]
            
            accuracy = accuracy_score(y_true_high_conf, y_pred_high_conf)
            balanced_acc = balanced_accuracy_score(y_true_high_conf, y_pred_high_conf)
            f1 = f1_score(y_true_high_conf, y_pred_high_conf, zero_division=0)
            conf_matrix = confusion_matrix(y_true_high_conf, y_pred_high_conf)
            
            # Calcular retorno simulado
            df['prediction'] = y_pred
            df['confidence'] = confidence
            
            # Simular estrategia de trading
            df['position'] = 0  # 0: sin posición, 1: long, -1: short
            df.loc[df['prediction'] == 1, 'position'] = 1
            df.loc[df['prediction'] == 0, 'position'] = -1
            
            # Calcular retornos diarios
            df['daily_return'] = df['price'].pct_change()
            
            # Calcular retornos de la estrategia (posición del día anterior * retorno del día)
            df['strategy_return'] = df['position'].shift(1) * df['daily_return']
            
            # Calcular retorno acumulado
            df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
            df['strategy_cumulative_return'] = (1 + df['strategy_return']).cumprod() - 1
            
            # Calcular métricas de rendimiento
            total_return = df['strategy_cumulative_return'].iloc[-1]
            sharpe_ratio = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(252)
            
            # Calcular porcentaje de predicciones con alta confianza
            high_conf_pct = sum(high_confidence_mask) / len(df) * 100
            
            print(f"  Resultados para {symbol}:")
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    Balanced Accuracy: {balanced_acc:.4f}")
            print(f"    F1 Score: {f1:.4f}")
            print(f"    Predicciones con alta confianza: {high_conf_pct:.2f}%")
            print(f"    Retorno total: {total_return:.2%}")
            print(f"    Sharpe Ratio: {sharpe_ratio:.2f}")
            
            # Guardar resultados
            backtest_results[symbol] = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'f1_score': f1,
                'confusion_matrix': conf_matrix.tolist(),
                'high_confidence_pct': high_conf_pct,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'threshold': symbol_threshold,
                'min_confidence': min_confidence_threshold
            }
            
            # Generar gráfico
            plt.figure(figsize=(12, 8))
            
            # Gráfico de precio
            ax1 = plt.subplot(2, 1, 1)
            df['price'].plot(ax=ax1, color='blue', label='Precio')
            
            # Marcar predicciones
            buy_signals = df[df['prediction'] == 1].index
            sell_signals = df[df['prediction'] == 0].index
            no_signals = df[df['prediction'] == -1].index
            
            ax1.scatter(buy_signals, df.loc[buy_signals, 'price'], color='green', marker='^', s=100, label='Compra')
            ax1.scatter(sell_signals, df.loc[sell_signals, 'price'], color='red', marker='v', s=100, label='Venta')
            ax1.scatter(no_signals, df.loc[no_signals, 'price'], color='gray', marker='o', s=50, label='Baja Confianza')
            
            ax1.set_title(f'Backtest de {symbol} - Predicciones vs Precio')
            ax1.set_ylabel('Precio')
            ax1.legend()
            
            # Gráfico de retornos acumulados
            ax2 = plt.subplot(2, 1, 2)
            df['cumulative_return'].plot(ax=ax2, color='blue', label='Buy & Hold')
            df['strategy_cumulative_return'].plot(ax=ax2, color='green', label='Estrategia')
            
            ax2.set_title(f'Retornos Acumulados - Estrategia vs Buy & Hold')
            ax2.set_ylabel('Retorno Acumulado')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f'backtest_results/{symbol}_backtest.png')
            plt.close()
            
        else:
            print(f"  [!] No hay predicciones con alta confianza para {symbol}")
            backtest_results[symbol] = {
                'error': 'No hay predicciones con alta confianza'
            }
    
    # Guardar resultados del backtest
    with open('backtest_results/backtest_summary.json', 'w') as f:
        json.dump({
            'backtest_period': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            },
            'model_info': {
                'trained_date': model_data['trained_date'],
                'min_confidence_threshold': min_confidence_threshold,
                'threshold_metric': threshold_metric
            },
            'results': backtest_results
        }, f, indent=4)
    
    print(f"\nBacktest completado. Resultados guardados en backtest_results/")
    
    # Calcular métricas globales
    valid_results = [r for r in backtest_results.values() if 'accuracy' in r]
    if valid_results:
        avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
        avg_balanced_acc = sum(r['balanced_accuracy'] for r in valid_results) / len(valid_results)
        avg_f1 = sum(r['f1_score'] for r in valid_results) / len(valid_results)
        avg_return = sum(r['total_return'] for r in valid_results) / len(valid_results)
        avg_sharpe = sum(r['sharpe_ratio'] for r in valid_results) / len(valid_results)
        
        print(f"\nMétricas globales:")
        print(f"  Accuracy promedio: {avg_accuracy:.4f}")
        print(f"  Balanced Accuracy promedio: {avg_balanced_acc:.4f}")
        print(f"  F1 Score promedio: {avg_f1:.4f}")
        print(f"  Retorno promedio: {avg_return:.2%}")
        print(f"  Sharpe Ratio promedio: {avg_sharpe:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ejecutar backtest del modelo de predicción de criptomonedas')
    parser.add_argument('--start-date', type=str, help='Fecha de inicio del backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Fecha de fin del backtest (YYYY-MM-DD)')
    parser.add_argument('--test-period', type=int, default=30, help='Período de backtest en días (si no se especifica start-date)')
    parser.add_argument('--threshold-metric', choices=['balanced_accuracy', 'f1'], default='f1', help='Métrica para seleccionar el umbral de decisión')
    
    args = parser.parse_args()
    
    run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        test_period_days=args.test_period,
        threshold_metric=args.threshold_metric
    )