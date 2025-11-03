import json
import requests
from datetime import datetime, timedelta
import time
import os

def fetch_historical_data(symbol, days=365):
    """
    Obtiene datos históricos de criptomonedas desde CoinGecko API
    """
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    
    headers = {
        'User-Agent': 'crypto-predictor/1.0'
    }
    
    try:
        print(f"Obteniendo datos históricos para {symbol}...")
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Procesar datos
        prices = data['prices']
        volumes = data['total_volumes']
        market_caps = data['market_caps']
        
        processed_data = []
        for i in range(len(prices)):
            timestamp = prices[i][0]
            date_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
            
            processed_data.append({
                'timestamp': timestamp,
                'date': date_str,
                'price': prices[i][1],
                'volume': volumes[i][1],
                'market_cap': market_caps[i][1]
            })
        
        print(f"✓ Datos obtenidos para {symbol}: {len(processed_data)} registros")
        return processed_data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"Error de límite de tasa para {symbol}. Esperando 60 segundos...")
            time.sleep(60)
            return fetch_historical_data(symbol, days)  # Reintentar después de esperar
        else:
            print(f"Error HTTP para {symbol}: {str(e)}")
            return None
    except Exception as e:
        print(f"Error obteniendo datos para {symbol}: {str(e)}")
        return None

def main():
    # Criptomonedas a obtener
    cryptos = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH',
        'solana': 'SOL',
        'binancecoin': 'BNB',
        'cardano': 'ADA',
        'ripple': 'XRP',
        'dogecoin': 'DOGE'
    }
    
    print(f"\n{'='*60}")
    print("OBTENIENDO DATOS HISTÓRICOS DE CRIPTOMONEDAS")
    print(f"{'='*60}\n")
    
    all_data = {}
    
    for crypto_id, symbol in cryptos.items():
        # Esperar entre solicitudes para evitar límites de tasa
        time.sleep(1)
        
        # Obtener datos históricos (1 año)
        data = fetch_historical_data(crypto_id, days=365)
        
        if data:
            all_data[symbol] = data
    
    # Guardar datos en archivo JSON
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'crypto_historical_data.json')
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n✓ Datos guardados en {output_path}")
    print(f"✓ Total de criptomonedas: {len(all_data)}")
    
    # Mostrar estadísticas
    for symbol, data in all_data.items():
        if data:
            print(f"  - {symbol}: {len(data)} días de datos históricos")
    
    print("\nEjecuta ahora: python scripts/train_model.py para entrenar el modelo")

if __name__ == "__main__":
    main()
