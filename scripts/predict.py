import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import os

def calculate_crypto_correlation(symbol, all_crypto_data, df):
    """
    Calcula la correlación entre la criptomoneda actual y otras principales
    para ajustar la confianza de la predicción.
    """
    # Lista de criptomonedas principales para correlación
    main_cryptos = ['BTC', 'ETH']
    
    # Si la criptomoneda actual es una de las principales, usar otras
    if symbol in main_cryptos:
        reference_cryptos = [c for c in ['BTC', 'ETH', 'BNB'] if c != symbol]
    else:
        reference_cryptos = main_cryptos
    
    correlation_data = {}
    correlation_signal = 0
    correlation_strength = 0
    
    # Calcular correlaciones con otras criptomonedas
    for ref_crypto in reference_cryptos:
        if ref_crypto in all_crypto_data:
            # Obtener datos de precio de la criptomoneda actual
            current_prices = df['price'].values
            
            # Obtener datos de la criptomoneda de referencia
            ref_data = all_crypto_data[ref_crypto]
            ref_prices = pd.to_numeric(ref_data['price']).values
            
            # Asegurar que ambas series tengan la misma longitud
            min_length = min(len(current_prices), len(ref_prices))
            if min_length > 30:  # Necesitamos suficientes datos para una correlación significativa
                # Usar los últimos N datos para ambas series
                current_prices = current_prices[-min_length:]
                ref_prices = ref_prices[-min_length:]
                
                # Calcular coeficiente de correlación
                correlation = np.corrcoef(current_prices, ref_prices)[0, 1]
                correlation_data[ref_crypto] = correlation
                
                # Determinar la fuerza de la correlación
                if abs(correlation) > 0.7:
                    correlation_strength = max(correlation_strength, abs(correlation))
                    
                    # Verificar si la tendencia de la referencia es alcista o bajista
                    ref_trend = 1 if ref_prices[-1] > ref_prices[-20] else -1
                    current_trend = 1 if current_prices[-1] > current_prices[-20] else -1
                    
                    # Si hay alta correlación y tendencias alineadas, reforzar señal
                    if correlation > 0 and ref_trend == current_trend:
                        correlation_signal += ref_trend
                    # Si hay alta correlación negativa y tendencias opuestas, reforzar señal
                    elif correlation < 0 and ref_trend != current_trend:
                        correlation_signal -= ref_trend
    
    # Normalizar la señal de correlación
    if correlation_data:
        correlation_signal = np.sign(correlation_signal)
        
    return {
        'correlation_data': correlation_data,
        'correlation_signal': correlation_signal,
        'correlation_strength': correlation_strength
    }

def calculate_technical_indicators(df):
    """
    Calcula indicadores técnicos (misma función que en train_model.py)
    """
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['ma_7'] = df['price'].rolling(window=7).mean()
    df['ma_21'] = df['price'].rolling(window=21).mean()
    df['ma_50'] = df['price'].rolling(window=50).mean()
    
    # EMAs y MACD (paridad con train_model.py)
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
    
    df['bb_middle'] = df['price'].rolling(window=20).mean()
    df['bb_std'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # Análisis de tendencia de mercado
    # Tendencia a corto plazo (7 días)
    df['trend_short'] = (df['ma_7'] > df['ma_7'].shift(3)).astype(int)
    
    # Tendencia a medio plazo (21 días)
    df['trend_medium'] = (df['ma_21'] > df['ma_21'].shift(7)).astype(int)
    
    # Tendencia a largo plazo (50 días)
    df['trend_long'] = (df['ma_50'] > df['ma_50'].shift(14)).astype(int)
    
    # Fuerza de la tendencia (combinación de tendencias)
    df['trend_strength'] = df['trend_short'] + df['trend_medium'] + df['trend_long']
    
    # Análisis de volatilidad avanzado
    # Volatilidad histórica (desviación estándar normalizada)
    df['volatility_norm'] = df['price'].rolling(window=30).std() / df['price'].rolling(window=30).mean()
    
    # Índice de volatilidad relativa (comparación con volatilidad histórica)
    df['volatility_relative'] = df['volatility'] / df['volatility'].rolling(window=90).mean()
    
    # Filtro de señales basado en volatilidad
    df['volatility_signal'] = 0  # Neutral por defecto
    # Alta volatilidad = señal de precaución
    df.loc[df['volatility_relative'] > 1.5, 'volatility_signal'] = -1
    # Baja volatilidad = señal favorable
    df.loc[df['volatility_relative'] < 0.5, 'volatility_signal'] = 1
    
    # Análisis de sentimiento de mercado (simulado)
    # En un sistema real, esto se conectaría a una API de noticias/sentimiento
    df['market_sentiment'] = 0  # Neutral por defecto
    # Sentimiento positivo cuando RSI > 60 y tendencias son positivas
    df.loc[(df['rsi'] > 60) & (df['trend_short'] > 0) & (df['trend_medium'] > 0), 'market_sentiment'] = 1
    # Sentimiento negativo cuando RSI < 40 y tendencias son negativas
    df.loc[(df['rsi'] < 40) & (df['trend_short'] < 0) & (df['trend_medium'] < 0), 'market_sentiment'] = -1
    
    # Sistema de consenso temporal
    # Análisis de tendencias en múltiples marcos temporales
    df['ema_5'] = df['price'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['price'].ewm(span=10, adjust=False).mean()
    df['ema_30'] = df['price'].ewm(span=30, adjust=False).mean()
    df['ema_60'] = df['price'].ewm(span=60, adjust=False).mean()
    
    # Señales de tendencia en diferentes marcos temporales
    df['trend_very_short'] = (df['ema_5'] > df['ema_10']).astype(int)  # Muy corto plazo
    df['trend_short_ema'] = (df['ema_10'] > df['ema_30']).astype(int)  # Corto plazo
    df['trend_medium_ema'] = (df['ema_30'] > df['ema_60']).astype(int)  # Medio plazo
    
    # Consenso temporal (suma de señales de tendencia)
    df['temporal_consensus'] = df['trend_very_short'] + df['trend_short_ema'] + df['trend_medium_ema']
    
    # Análisis de divergencias
    # Divergencia RSI-Precio (señal de posible reversión)
    df['price_higher_high'] = ((df['price'] > df['price'].shift(1)) & 
                              (df['price'].shift(1) > df['price'].shift(2))).astype(int)
    df['rsi_lower_high'] = ((df['rsi'] < df['rsi'].shift(1)) & 
                           (df['rsi'].shift(1) > df['rsi'].shift(2))).astype(int)
    
    # Divergencia bajista (precio sube pero RSI baja)
    df['bearish_divergence'] = (df['price_higher_high'] & df['rsi_lower_high']).astype(int)
    
    df['price_lower_low'] = ((df['price'] < df['price'].shift(1)) & 
                            (df['price'].shift(1) < df['price'].shift(2))).astype(int)
    df['rsi_higher_low'] = ((df['rsi'] > df['rsi'].shift(1)) & 
                           (df['rsi'].shift(1) < df['rsi'].shift(2))).astype(int)
    
    # Divergencia alcista (precio baja pero RSI sube)
    df['bullish_divergence'] = (df['price_lower_low'] & df['rsi_higher_low']).astype(int)
    
    # Sistema de alertas para condiciones extremas
    # Condiciones de sobrecompra/sobreventa extremas
    df['extreme_overbought'] = (df['rsi'] > 80).astype(int)
    df['extreme_oversold'] = (df['rsi'] < 20).astype(int)
    
    # Alerta de mercado extremo
    df['market_extreme_alert'] = 0
    df.loc[df['extreme_overbought'] == 1, 'market_extreme_alert'] = -1  # Posible caída
    df.loc[df['extreme_oversold'] == 1, 'market_extreme_alert'] = 1     # Posible subida
    
    # Sistema de filtrado adaptativo basado en condiciones de mercado
    # Identificar régimen de mercado (tendencia, rango, alta volatilidad)
    df['market_regime'] = 0  # Neutral/Rango por defecto
    
    # Mercado en tendencia alcista fuerte
    df.loc[(df['trend_strength'] >= 2) & (df['volatility_relative'] < 1.2) & 
           (df['price'] > df['ma_21']), 'market_regime'] = 1
    
    # Mercado en tendencia bajista fuerte
    df.loc[(df['trend_strength'] <= 1) & (df['volatility_relative'] < 1.2) & 
           (df['price'] < df['ma_21']), 'market_regime'] = -1
    
    # Mercado de alta volatilidad (posible cambio de régimen)
    df.loc[df['volatility_relative'] > 1.5, 'market_regime'] = 2
    
    # Ajustar umbrales de filtrado según régimen de mercado
    df['adaptive_threshold'] = 0.5  # Umbral neutral por defecto
    
    # En tendencia fuerte, reducir umbral para capturar más señales en dirección de tendencia
    df.loc[df['market_regime'] == 1, 'adaptive_threshold'] = 0.4  # Tendencia alcista
    df.loc[df['market_regime'] == -1, 'adaptive_threshold'] = 0.4  # Tendencia bajista
    
    # En alta volatilidad, aumentar umbral para filtrar señales falsas
    df.loc[df['market_regime'] == 2, 'adaptive_threshold'] = 0.7
    
    # Análisis de volumen relativo
    # Volumen relativo (comparado con promedio de 20 días)
    df['volume_relative'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Señal de confirmación por volumen
    df['volume_confirmation'] = 0  # Neutral por defecto
    
    # Alto volumen confirma movimiento de precio
    df.loc[(df['price_change'] > 0) & (df['volume_relative'] > 1.5), 'volume_confirmation'] = 1  # Confirma subida
    df.loc[(df['price_change'] < 0) & (df['volume_relative'] > 1.5), 'volume_confirmation'] = -1  # Confirma bajada
    
    # Bajo volumen sugiere debilidad en el movimiento
    df.loc[(df['price_change'].abs() > 0.01) & (df['volume_relative'] < 0.7), 'volume_confirmation'] = -2  # Señal débil
    
    # Detección de patrones de velas japonesas
    # Crear columnas para OHLC (simuladas a partir del precio)
    # En un sistema real, usaríamos datos OHLC reales
    df['open'] = df['price'].shift(1)
    df['high'] = df['price'] * 1.005  # Simulado: 0.5% más alto que el precio de cierre
    df['low'] = df['price'] * 0.995   # Simulado: 0.5% más bajo que el precio de cierre
    df['close'] = df['price']
    
    # Calcular tamaño del cuerpo y sombras
    df['body_size'] = (df['close'] - df['open']).abs()
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Detectar patrones de velas básicos
    # Martillo/Hombre colgado (potencial reversión)
    df['hammer'] = (
        (df['body_size'] < df['lower_shadow'] * 2) &  # Cuerpo pequeño
        (df['upper_shadow'] < df['body_size'] * 0.5) &  # Sombra superior pequeña
        (df['lower_shadow'] > df['body_size'] * 2)  # Sombra inferior larga
    ).astype(int)
    
    # Estrella fugaz/Estrella vespertina (potencial reversión bajista)
    df['shooting_star'] = (
        (df['body_size'] < df['upper_shadow'] * 2) &  # Cuerpo pequeño
        (df['lower_shadow'] < df['body_size'] * 0.5) &  # Sombra inferior pequeña
        (df['upper_shadow'] > df['body_size'] * 2)  # Sombra superior larga
    ).astype(int)
    
    # Doji (indecisión)
    df['doji'] = (
        (df['body_size'] < (df['high'] - df['low']) * 0.1)  # Cuerpo muy pequeño
    ).astype(int)
    
    # Señal combinada de patrones de velas
    df['candlestick_signal'] = 0  # Neutral por defecto
    
    # Señales alcistas
    df.loc[(df['hammer'] == 1) & (df['close'] < df['ma_21']), 'candlestick_signal'] = 1
    
    # Señales bajistas
    df.loc[(df['shooting_star'] == 1) & (df['close'] > df['ma_21']), 'candlestick_signal'] = -1
    
    # Señales de indecisión
    df.loc[df['doji'] == 1, 'candlestick_signal'] = 2
    
    # NUEVA MEJORA: Sistema de ponderación dinámica de indicadores
    # Calcular la efectividad histórica de cada indicador
    # En un sistema real, esto se calcularía con datos históricos de aciertos/fallos
    # Aquí simulamos la efectividad basada en la volatilidad y el régimen de mercado
    
    # Inicializar pesos de indicadores (valores base)
    df['weight_rsi'] = 1.0
    df['weight_macd'] = 1.0
    df['weight_ma_cross'] = 1.0
    df['weight_volume'] = 1.0
    df['weight_volatility'] = 1.0
    df['weight_candlestick'] = 1.0
    
    # Ajustar pesos según régimen de mercado
    # En tendencia alcista, dar más peso a indicadores de momentum
    df.loc[df['market_regime'] == 1, 'weight_macd'] = 1.5
    df.loc[df['market_regime'] == 1, 'weight_ma_cross'] = 1.3
    df.loc[df['market_regime'] == 1, 'weight_rsi'] = 0.8
    
    # En tendencia bajista, dar más peso a RSI y patrones de velas
    df.loc[df['market_regime'] == -1, 'weight_rsi'] = 1.5
    df.loc[df['market_regime'] == -1, 'weight_candlestick'] = 1.3
    df.loc[df['market_regime'] == -1, 'weight_macd'] = 0.8
    
    # En alta volatilidad, dar más peso a volumen y volatilidad
    df.loc[df['market_regime'] == 2, 'weight_volume'] = 1.8
    df.loc[df['market_regime'] == 2, 'weight_volatility'] = 1.5
    df.loc[df['market_regime'] == 2, 'weight_ma_cross'] = 0.7
    
    # Calcular señales ponderadas
    # RSI señal (>70 = -1, <30 = 1, otro = 0)
    df['rsi_signal'] = 0
    df.loc[df['rsi'] > 70, 'rsi_signal'] = -1
    df.loc[df['rsi'] < 30, 'rsi_signal'] = 1
    
    # MACD señal (MACD > 0 = 1, MACD < 0 = -1)
    df['macd_signal'] = np.sign(df['macd'])
    
    # MA Cross señal (MA7 > MA21 = 1, MA7 < MA21 = -1)
    df['ma_cross_signal'] = np.sign(df['ma_7'] - df['ma_21'])
    
    # Señal combinada ponderada
    df['weighted_signal'] = (
        df['rsi_signal'] * df['weight_rsi'] +
        df['macd_signal'] * df['weight_macd'] +
        df['ma_cross_signal'] * df['weight_ma_cross'] +
        df['volume_confirmation'] * df['weight_volume'] +
        df['volatility_signal'] * df['weight_volatility'] +
        df['candlestick_signal'] * df['weight_candlestick']
    )
    
    # Normalizar la señal ponderada
    df['weighted_signal_norm'] = df['weighted_signal'] / (
        df['weight_rsi'] + df['weight_macd'] + df['weight_ma_cross'] + 
        df['weight_volume'] + df['weight_volatility'] + df['weight_candlestick']
    )
    
    # Calcular la confianza basada en la señal ponderada
    df['weighted_confidence'] = df['weighted_signal_norm'].abs()
    
    return df

def analyze_social_sentiment(symbol):
    """
    Analiza el sentimiento de redes sociales para una criptomoneda.
    En un entorno real, esto se conectaría a APIs de Twitter, Reddit, etc.
    Para esta implementación, simulamos los resultados.
    """
    # Mapeo de símbolos a términos de búsqueda
    search_terms = {
        'BTC': ['bitcoin', 'btc', 'crypto'],
        'ETH': ['ethereum', 'eth', 'vitalik'],
        'SOL': ['solana', 'sol'],
        'BNB': ['binance', 'bnb']
    }
    
    # Simulación de datos de sentimiento basados en tendencias actuales
    # En un entorno real, esto vendría de APIs de redes sociales
    sentiment_simulation = {
        'BTC': {'positive': 0.65, 'negative': 0.25, 'neutral': 0.10},
        'ETH': {'positive': 0.55, 'negative': 0.30, 'neutral': 0.15},
        'SOL': {'positive': 0.40, 'negative': 0.45, 'neutral': 0.15},
        'BNB': {'positive': 0.50, 'negative': 0.30, 'neutral': 0.20}
    }
    
    if symbol in sentiment_simulation:
        sentiment_data = sentiment_simulation[symbol]
        
        # Calcular puntuación de sentimiento (-1 a 1)
        sentiment_score = sentiment_data['positive'] - sentiment_data['negative']
        
        # Determinar la fuerza del sentimiento
        sentiment_strength = abs(sentiment_score)
        
        # Determinar la señal de sentimiento
        if sentiment_score > 0.15:
            sentiment_signal = 1  # Positivo
        elif sentiment_score < -0.15:
            sentiment_signal = -1  # Negativo
        else:
            sentiment_signal = 0  # Neutral
            
        # Calcular volumen de menciones (simulado)
        mention_volume = {
            'BTC': 0.85,
            'ETH': 0.70,
            'SOL': 0.50,
            'BNB': 0.45
        }.get(symbol, 0.3)
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_strength': sentiment_strength,
            'sentiment_signal': sentiment_signal,
            'mention_volume': mention_volume
        }
    else:
        return {
            'sentiment_score': 0,
            'sentiment_strength': 0,
            'sentiment_signal': 0,
            'mention_volume': 0
        }

def predict_future_volatility(df, window=30):
    """
    Predice la volatilidad futura basada en patrones históricos y tendencias actuales.
    """
    # Calcular volatilidad histórica
    returns = df['price'].pct_change().dropna()
    hist_volatility = returns.rolling(window=window).std().dropna() * np.sqrt(365)
    
    if len(hist_volatility) < window:
        return {'volatility_prediction': 'normal', 'confidence': 0.5}
    
    # Últimos valores de volatilidad
    recent_volatility = hist_volatility.iloc[-window:]
    current_volatility = hist_volatility.iloc[-1]
    
    # Detectar tendencia de volatilidad
    volatility_trend = 1 if recent_volatility.iloc[-1] > recent_volatility.iloc[-window//2] else -1
    
    # Calcular percentiles para clasificar la volatilidad
    low_percentile = np.percentile(hist_volatility, 25)
    high_percentile = np.percentile(hist_volatility, 75)
    
    # Predecir volatilidad futura
    if current_volatility < low_percentile and volatility_trend < 0:
        prediction = 'baja'
        confidence = 0.8
    elif current_volatility > high_percentile and volatility_trend > 0:
        prediction = 'alta'
        confidence = 0.8
    elif current_volatility > high_percentile and volatility_trend < 0:
        prediction = 'decreciente'
        confidence = 0.7
    elif current_volatility < low_percentile and volatility_trend > 0:
        prediction = 'creciente'
        confidence = 0.7
    else:
        prediction = 'normal'
        confidence = 0.6
    
    return {
        'volatility_prediction': prediction,
        'confidence': confidence
    }

def make_predictions(threshold_metric='balanced_accuracy'):
    """
    Hace predicciones para todas las criptomonedas
    """
    print(f"\n{'='*60}")
    print("GENERANDO PREDICCIONES")
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
        per_symbol_metrics = model_data.get('per_symbol_metrics', {})
        min_confidence_threshold = model_data.get('min_confidence_threshold', 0.75)
        
        # Determinar si estamos usando el modelo antiguo o el nuevo con modelos por símbolo
        using_per_symbol_models = len(models_per_symbol) > 0
        
        if using_per_symbol_models:
            print(f"✓ Modelos separados por símbolo cargados")
            print(f"✓ Umbral mínimo de confianza: {min_confidence_threshold:.2f}")
            print(f"✓ Métrica de umbral seleccionada: {threshold_metric}")
            print(f"✓ Entrenado el: {model_data['trained_date']}\n")
            
            # Mostrar métricas por símbolo
            print("Métricas por símbolo:")
            for symbol, metrics in per_symbol_metrics.items():
                print(f"  {symbol}: BalAcc {metrics.get('balanced_accuracy', 0):.3f} | F1 {metrics.get('f1', 0):.3f}")
        else:
            # Modelo antiguo (un solo modelo para todos los símbolos)
            model = model_data['model']
            scaler = model_data['scaler']
            calibration_info = model_data.get('calibration', {})
            decision_threshold = calibration_info.get('decision_threshold', 0.5)
            decision_thresholds = calibration_info.get('decision_thresholds', {})
            
            print(f"✓ Modelo único cargado: {model_data['model_name']}")
            print(f"✓ Accuracy del modelo: {model_data['accuracy']*100:.2f}%")
            print(f"✓ Umbral global de decisión: {decision_threshold:.2f}")
            print(f"✓ Métrica de umbral seleccionada: {threshold_metric}")
            print(f"✓ Entrenado el: {model_data['trained_date']}\n")
        
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
    
    # Crear un diccionario para almacenar todos los datos de criptomonedas
    all_crypto_data = {}
    for symbol, data in crypto_data.items():
        all_crypto_data[symbol] = pd.DataFrame(data)
        all_crypto_data[symbol]['price'] = pd.to_numeric(all_crypto_data[symbol]['price'])
        all_crypto_data[symbol]['volume'] = pd.to_numeric(all_crypto_data[symbol]['volume'])
        if 'market_cap' in all_crypto_data[symbol].columns:
            all_crypto_data[symbol]['market_cap'] = pd.to_numeric(all_crypto_data[symbol]['market_cap'])
    
    predictions = {}
    
    for symbol, data in crypto_data.items():
        print(f"Prediciendo {symbol}...")
        
        df = pd.DataFrame(data)
        df['price'] = pd.to_numeric(df['price'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['market_cap'] = pd.to_numeric(df['market_cap'])
        
        # Calcular indicadores
        df = calculate_technical_indicators(df)
        
        # Calcular correlación con otras criptomonedas
        correlation_results = calculate_crypto_correlation(symbol, all_crypto_data, df)
        df['correlation_signal'] = correlation_results['correlation_signal']
        df['correlation_strength'] = correlation_results['correlation_strength']
        
        # Analizar sentimiento de redes sociales
        social_sentiment = analyze_social_sentiment(symbol)
        df['social_sentiment_signal'] = social_sentiment['sentiment_signal']
        df['social_sentiment_strength'] = social_sentiment['sentiment_strength']
        df['social_mention_volume'] = social_sentiment['mention_volume']
        
        # Predecir volatilidad futura
        volatility_prediction = predict_future_volatility(df)
        df['future_volatility_prediction'] = volatility_prediction['volatility_prediction']
        df['future_volatility_confidence'] = volatility_prediction['confidence']
        
        # Obtener últimos datos disponibles
        last_row = df.iloc[-1:].copy()
        
        # Verificar si tenemos suficientes datos
        if last_row.isnull().values.any():
            print(f"  [!] Datos insuficientes para {symbol}, omitiendo...")
            continue
        
        # Extraer características
        X = last_row[features]
        
        # Hacer predicción
        if using_per_symbol_models:
            # Usar modelo específico para este símbolo
            if symbol in models_per_symbol and symbol in scalers_per_symbol:
                symbol_model = models_per_symbol[symbol]
                symbol_scaler = scalers_per_symbol[symbol]
                
                # Escalar características
                X_scaled = symbol_scaler.transform(X)
                
                # Obtener probabilidades
                proba = symbol_model.predict_proba(X_scaled)[0, 1]
                
                # Obtener umbral específico para este símbolo
                symbol_threshold = 0.5
                if symbol in per_symbol_thresholds:
                    symbol_thresholds = per_symbol_thresholds[symbol]
                    if isinstance(symbol_thresholds, dict) and threshold_metric in symbol_thresholds:
                        symbol_threshold = symbol_thresholds[threshold_metric]
                    elif not isinstance(symbol_thresholds, dict):
                        symbol_threshold = symbol_thresholds
                
                # Aplicar umbral para decisión
                prediction = "BULLISH" if proba >= symbol_threshold else "BEARISH"
                
                # Calcular confianza
                confidence = proba if prediction == "BULLISH" else 1 - proba
                confidence_pct = confidence * 100
                
                # Verificar señales de volatilidad y sentimiento
                volatility_signal = 0
                market_sentiment = 0
                temporal_consensus = 0
                divergence_signal = 0
                extreme_alert = 0
                
                if 'volatility_signal' in last_row.columns:
                    volatility_signal = last_row['volatility_signal'].values[0]
                    
                    # Ajustar confianza basada en volatilidad
                    if volatility_signal == -1:  # Alta volatilidad
                        confidence *= 0.8  # Reduce confianza en 20%
                        print(f"  ⚠️ Alta volatilidad detectada - reduciendo confianza")
                    elif volatility_signal == 1:  # Baja volatilidad
                        confidence *= 1.1  # Aumenta confianza en 10%
                        print(f"  ✓ Baja volatilidad detectada - aumentando confianza")
                
                if 'market_sentiment' in last_row.columns:
                    market_sentiment = last_row['market_sentiment'].values[0]
                    
                    if market_sentiment == 1 and prediction == "BULLISH":
                        confidence *= 1.1  # Aumenta confianza en 10%
                        print(f"  ✓ Sentimiento positivo confirma predicción BULLISH")
                    elif market_sentiment == -1 and prediction == "BEARISH":
                        confidence *= 1.1  # Aumenta confianza en 10%
                        print(f"  ✓ Sentimiento negativo confirma predicción BEARISH")
                    elif market_sentiment == 1 and prediction == "BEARISH":
                        confidence *= 0.9  # Reduce confianza en 10%
                        print(f"  ⚠️ Sentimiento positivo contradice predicción BEARISH")
                    elif market_sentiment == -1 and prediction == "BULLISH":
                        confidence *= 0.9  # Reduce confianza en 10%
                        print(f"  ⚠️ Sentimiento negativo contradice predicción BULLISH")
                
                # Sistema de consenso temporal
                if 'temporal_consensus' in last_row.columns:
                    temporal_consensus = last_row['temporal_consensus'].values[0]
                    
                    # Ajustar por consenso temporal
                    if temporal_consensus == 3 and prediction == "BULLISH":
                        confidence *= 1.2  # Aumenta confianza en 20%
                        print(f"  ✓ Consenso temporal fuerte confirma predicción BULLISH")
                    elif temporal_consensus == 0 and prediction == "BEARISH":
                        confidence *= 1.2  # Aumenta confianza en 20%
                        print(f"  ✓ Consenso temporal fuerte confirma predicción BEARISH")
                    elif temporal_consensus == 3 and prediction == "BEARISH":
                        confidence *= 0.8  # Reduce confianza en 20%
                        print(f"  ⚠️ Consenso temporal contradice predicción BEARISH")
                    elif temporal_consensus == 0 and prediction == "BULLISH":
                        confidence *= 0.8  # Reduce confianza en 20%
                        print(f"  ⚠️ Consenso temporal contradice predicción BULLISH")
                
                # Análisis de divergencias
                if 'bearish_divergence' in last_row.columns and 'bullish_divergence' in last_row.columns:
                    bearish_divergence = last_row['bearish_divergence'].values[0]
                    bullish_divergence = last_row['bullish_divergence'].values[0]
                    
                    # Ajustar por divergencias
                    if bearish_divergence == 1 and prediction == "BULLISH":
                        confidence *= 0.7  # Reduce confianza en 30%
                        print(f"  ⚠️ Divergencia bajista detectada - contradice predicción BULLISH")
                    elif bullish_divergence == 1 and prediction == "BEARISH":
                        confidence *= 0.7  # Reduce confianza en 30%
                        print(f"  ⚠️ Divergencia alcista detectada - contradice predicción BEARISH")
                
                # Sistema de alertas para condiciones extremas
                if 'market_extreme_alert' in last_row.columns:
                    extreme_alert = last_row['market_extreme_alert'].values[0]
                    
                    # Ajustar por alertas de mercado extremo
                    if extreme_alert == -1 and prediction == "BULLISH":
                        confidence *= 0.85  # Reduce confianza en 15%
                        print(f"  ⚠️ Condición de sobrecompra detectada - reduce confianza en BULLISH")
                    elif extreme_alert == 1 and prediction == "BEARISH":
                        confidence *= 0.85  # Reduce confianza en 15%
                        print(f"  ⚠️ Condición de sobreventa detectada - reduce confianza en BEARISH")
                
                # Sistema de filtrado adaptativo basado en condiciones de mercado
                if 'market_regime' in last_row.columns and 'adaptive_threshold' in last_row.columns:
                    market_regime = last_row['market_regime'].values[0]
                    adaptive_threshold = last_row['adaptive_threshold'].values[0]
                    
                    # Ajustar confianza según régimen de mercado
                    if market_regime == 1 and prediction == "BULLISH":
                        confidence *= 1.15  # Aumenta confianza en 15%
                        print(f"  ✓ Régimen de mercado alcista confirma predicción BULLISH")
                    elif market_regime == -1 and prediction == "BEARISH":
                        confidence *= 1.15  # Aumenta confianza en 15%
                        print(f"  ✓ Régimen de mercado bajista confirma predicción BEARISH")
                    elif market_regime == 2:  # Alta volatilidad
                        confidence *= 0.85  # Reduce confianza en 15%
                        print(f"  ⚠️ Régimen de alta volatilidad - reduce confianza general")
                
                # Análisis de volumen relativo
                if 'volume_confirmation' in last_row.columns:
                    volume_confirmation = last_row['volume_confirmation'].values[0]
                    
                    # Ajustar confianza según confirmación de volumen
                    if volume_confirmation == 1 and prediction == "BULLISH":
                        confidence *= 1.12  # Aumenta confianza en 12%
                        print(f"  ✓ Alto volumen confirma predicción BULLISH")
                    elif volume_confirmation == -1 and prediction == "BEARISH":
                        confidence *= 1.12  # Aumenta confianza en 12%
                        print(f"  ✓ Alto volumen confirma predicción BEARISH")
                    elif volume_confirmation == -2:  # Volumen bajo
                        confidence *= 0.9  # Reduce confianza en 10%
                        print(f"  ⚠️ Bajo volumen detectado - reduce confianza general")
                
                # Detección de patrones de velas japonesas
                if 'candlestick_signal' in last_row.columns:
                    candlestick_signal = last_row['candlestick_signal'].values[0]
                    
                    # Ajustar confianza según patrones de velas
                    if candlestick_signal == 1 and prediction == "BULLISH":
                        confidence *= 1.1  # Aumenta confianza en 10%
                        print(f"  ✓ Patrón de vela alcista confirma predicción BULLISH")
                    elif candlestick_signal == -1 and prediction == "BEARISH":
                        confidence *= 1.1  # Aumenta confianza en 10%
                        print(f"  ✓ Patrón de vela bajista confirma predicción BEARISH")
                    elif candlestick_signal == 1 and prediction == "BEARISH":
                        confidence *= 0.9  # Reduce confianza en 10%
                        print(f"  ⚠️ Patrón de vela alcista contradice predicción BEARISH")
                    elif candlestick_signal == -1 and prediction == "BULLISH":
                        confidence *= 0.9  # Reduce confianza en 10%
                        print(f"  ⚠️ Patrón de vela bajista contradice predicción BULLISH")
                    elif candlestick_signal == 2:  # Doji (indecisión)
                        confidence *= 0.95  # Reduce confianza en 5%
                        print(f"  ⚠️ Patrón de vela Doji indica indecisión - reduce confianza ligeramente")
                
                # Sistema de ponderación dinámica de indicadores
                if 'weighted_confidence' in last_row.columns:
                    weighted_confidence = last_row['weighted_confidence'].values[0]
                    weighted_signal = last_row['weighted_signal_norm'].values[0]
                    
                    # Ajustar confianza basada en el sistema de ponderación dinámica
                    if (weighted_signal > 0 and prediction == "BULLISH") or (weighted_signal < 0 and prediction == "BEARISH"):
                        # La señal ponderada confirma la predicción
                        confidence_boost = min(weighted_confidence * 0.25, 0.25)  # Máximo 25% de aumento
                        confidence *= (1 + confidence_boost)
                        print(f"  ✓✓ Sistema de ponderación dinámica confirma predicción con {weighted_confidence:.2f} de confianza")
                        print(f"  ✓✓ Aumentando confianza en {confidence_boost*100:.1f}%")
                    elif (weighted_signal > 0 and prediction == "BEARISH") or (weighted_signal < 0 and prediction == "BULLISH"):
                        # La señal ponderada contradice la predicción
                        confidence_reduction = min(weighted_confidence * 0.3, 0.3)  # Máximo 30% de reducción
                        confidence *= (1 - confidence_reduction)
                        print(f"  ⚠️⚠️ Sistema de ponderación dinámica contradice predicción con {weighted_confidence:.2f} de confianza")
                        print(f"  ⚠️⚠️ Reduciendo confianza en {confidence_reduction*100:.1f}%")
                
                # Análisis de correlación con otras criptomonedas
                if 'correlation_signal' in last_row.columns and 'correlation_strength' in last_row.columns:
                    correlation_signal = last_row['correlation_signal'].values[0]
                    correlation_strength = last_row['correlation_strength'].values[0]
                    
                    if correlation_strength > 0.7:  # Alta correlación
                        if (correlation_signal > 0 and prediction == "BULLISH") or (correlation_signal < 0 and prediction == "BEARISH"):
                            # La correlación confirma la predicción
                            confidence_boost = min(correlation_strength * 0.2, 0.2)  # Máximo 20% de aumento
                            confidence *= (1 + confidence_boost)
                            print(f"  ✓ Alta correlación con otras criptomonedas confirma predicción {prediction}")
                            print(f"  ✓ Aumentando confianza en {confidence_boost*100:.1f}%")
                        elif (correlation_signal > 0 and prediction == "BEARISH") or (correlation_signal < 0 and prediction == "BULLISH"):
                            # La correlación contradice la predicción
                            confidence_reduction = min(correlation_strength * 0.25, 0.25)  # Máximo 25% de reducción
                            confidence *= (1 - confidence_reduction)
                            print(f"  ⚠️ Alta correlación con otras criptomonedas contradice predicción {prediction}")
                            print(f"  ⚠️ Reduciendo confianza en {confidence_reduction*100:.1f}%")
                
                # Análisis de sentimiento de redes sociales
                if 'social_sentiment_signal' in last_row.columns and 'social_sentiment_strength' in last_row.columns:
                    sentiment_signal = last_row['social_sentiment_signal'].values[0]
                    sentiment_strength = last_row['social_sentiment_strength'].values[0]
                    mention_volume = last_row['social_mention_volume'].values[0]
                    
                    if sentiment_strength > 0.2:  # Sentimiento significativo
                        # Ajustar el impacto según el volumen de menciones
                        impact_factor = min(0.25, sentiment_strength * mention_volume * 0.3)
                        
                        if (sentiment_signal > 0 and prediction == "BULLISH") or (sentiment_signal < 0 and prediction == "BEARISH"):
                            # El sentimiento confirma la predicción
                            confidence *= (1 + impact_factor)
                            print(f"  ✓ Sentimiento social {'positivo' if sentiment_signal > 0 else 'negativo'} confirma predicción {prediction}")
                            print(f"  ✓ Aumentando confianza en {impact_factor*100:.1f}%")
                        elif (sentiment_signal > 0 and prediction == "BEARISH") or (sentiment_signal < 0 and prediction == "BULLISH"):
                            # El sentimiento contradice la predicción
                            confidence *= (1 - impact_factor)
                            print(f"  ⚠️ Sentimiento social {'positivo' if sentiment_signal > 0 else 'negativo'} contradice predicción {prediction}")
                            print(f"  ⚠️ Reduciendo confianza en {impact_factor*100:.1f}%")
                
                # Predicción de volatilidad futura
                if 'future_volatility_prediction' in last_row.columns:
                    vol_prediction = last_row['future_volatility_prediction'].values[0]
                    vol_confidence = last_row['future_volatility_confidence'].values[0]
                    
                    # Ajustar confianza basada en la predicción de volatilidad
                    if vol_prediction == 'baja':
                        if prediction == "BULLISH":
                            # Baja volatilidad favorece tendencias alcistas estables
                            confidence_boost = vol_confidence * 0.15
                            confidence *= (1 + confidence_boost)
                            print(f"  ✓ Predicción de baja volatilidad futura favorece tendencia BULLISH")
                            print(f"  ✓ Aumentando confianza en {confidence_boost*100:.1f}%")
                    elif vol_prediction == 'alta':
                        # Alta volatilidad reduce confianza general
                        confidence_reduction = vol_confidence * 0.2
                        confidence *= (1 - confidence_reduction)
                        print(f"  ⚠️ Predicción de alta volatilidad futura reduce confianza general")
                        print(f"  ⚠️ Reduciendo confianza en {confidence_reduction*100:.1f}%")
                    elif vol_prediction == 'creciente' and prediction == "BEARISH":
                        # Volatilidad creciente puede favorecer tendencias bajistas
                        confidence_boost = vol_confidence * 0.1
                        confidence *= (1 + confidence_boost)
                        print(f"  ✓ Predicción de volatilidad creciente favorece tendencia BEARISH")
                        print(f"  ✓ Aumentando confianza en {confidence_boost*100:.1f}%")
                
                # Asegurar que la confianza esté en el rango [0, 1]
                confidence = min(max(confidence, 0), 1)
                confidence_pct = confidence * 100
                
                # Determinar estado de confianza
                confidence_status = "ALTA" if confidence >= min_confidence_threshold else "BAJA"
                
                # Guardar predicción
                predictions[symbol] = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "confidence_status": confidence_status,
                    "threshold": symbol_threshold,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Mostrar resultado
                print(f"  Predicción: {prediction}")
                print(f"  Confianza: {confidence_pct:.2f}% ({confidence_status})")
                print(f"  Umbral de decisión: {symbol_threshold:.2f}")
                print()
        else:
            # Usar modelo único para todos los símbolos
            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)[0, 1]
            
            # Obtener umbral específico para este símbolo o usar el global
            symbol_threshold = decision_threshold
            if symbol in per_symbol_thresholds:
                symbol_thresholds = per_symbol_thresholds[symbol]
                if isinstance(symbol_thresholds, dict) and threshold_metric in symbol_thresholds:
                    symbol_threshold = symbol_thresholds[threshold_metric]
                elif not isinstance(symbol_thresholds, dict):
                    symbol_threshold = symbol_thresholds
            elif threshold_metric in decision_thresholds:
                symbol_threshold = decision_thresholds[threshold_metric]
            
            prediction = "BULLISH" if proba >= symbol_threshold else "BEARISH"
            confidence = proba if prediction == "BULLISH" else 1 - proba
            confidence_pct = confidence * 100
            
            print(f"  Predicción: {prediction} con {confidence_pct:.2f}% de confianza (umbral: {symbol_threshold:.2f})")
            
            predictions[symbol] = {
                "symbol": symbol,
                "prediction": prediction,
                # Normalizamos confianza a porcentaje en 0-100
                "confidence": confidence_pct,
                "confidence_status": confidence_status,
                "threshold": symbol_threshold,
                # Señales internas del modelo
                "volatility_signal": int(volatility_signal) if 'volatility_signal' in last_row.columns else 0,
                "market_sentiment": int(market_sentiment) if 'market_sentiment' in last_row.columns else 0,
                "temporal_consensus": int(temporal_consensus) if 'temporal_consensus' in last_row.columns else 0,
                "divergence_signal": int(divergence_signal),
                "extreme_alert": int(extreme_alert) if 'market_extreme_alert' in last_row.columns else 0,
                "market_regime": int(market_regime) if 'market_regime' in last_row.columns else 0,
                "volume_confirmation": int(volume_confirmation) if 'volume_confirmation' in last_row.columns else 0,
                "candlestick_signal": int(candlestick_signal) if 'candlestick_signal' in last_row.columns else 0,
                # Nuevas mejoras expuestas al frontend
                "future_volatility_prediction": (last_row['future_volatility_prediction'].values[0]
                                                  if 'future_volatility_prediction' in last_row.columns else None),
                "future_volatility_confidence": (float(last_row['future_volatility_confidence'].values[0])
                                                  if 'future_volatility_confidence' in last_row.columns else None),
                "social_sentiment_signal": (int(last_row['social_sentiment_signal'].values[0])
                                             if 'social_sentiment_signal' in last_row.columns else None),
                "social_sentiment_strength": (float(last_row['social_sentiment_strength'].values[0])
                                               if 'social_sentiment_strength' in last_row.columns else None),
                "social_mention_volume": (int(last_row['social_mention_volume'].values[0])
                                            if 'social_mention_volume' in last_row.columns else None),
                "timestamp": datetime.now().isoformat()
            }
    
    # Guardar predicciones
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n✓ Predicciones guardadas en predictions.json")
    
    # Mostrar resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE PREDICCIONES")
    print(f"{'='*60}")
    
    for symbol, pred in predictions.items():
        confidence_txt = f"{pred['confidence']:.2f}%"
        if using_per_symbol_models and 'confidence_status' in pred:
            confidence_status = pred['confidence_status']
            if "BAJA CONFIANZA" in confidence_status:
                confidence_txt += f" ({confidence_status})"
            elif "ALTA CONFIANZA" in confidence_status:
                confidence_txt += f" (ALTA CONFIANZA)"
        print(f"{symbol}: {pred['prediction']} con {confidence_txt}")
    
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar predicciones con selección de métrica de umbral")
    parser.add_argument("--threshold-metric", choices=["balanced_accuracy", "f1"], default="balanced_accuracy", help="Métrica para seleccionar el umbral de decisión")
    args = parser.parse_args()
    make_predictions(threshold_metric=args.threshold_metric)
