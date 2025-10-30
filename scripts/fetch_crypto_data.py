import json
import requests
from datetime import datetime, timedelta
import time

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
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Procesar datos
        prices = data['prices']
        volumes = data['total_volumes']
        market_caps = data['market_caps']
        
        processed_data = []
        for i in range(len(prices)):
            processed_data.append({
                'timestamp': prices[i][0],
                'date': datetime.fromtimestamp(prices[i][0] / 1000).strftime('%Y-%m-%d'),
                'price': prices[i][1],
                'volume': volumes[i][1],
                'market_cap': market_caps[i][1]
            })
        
        return processed_data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def main():
    # Criptomonedas a obtener
    cryptos = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH',
        'solana': 'SOL',
        'binancecoin': 'BNB'
    }
    
    all_data = {}
    
    for crypto_id, symbol in cryptos.items():
        print(f"Fetching data for {symbol}...")
        data = fetch_historical_data(crypto_id, days=365)
        
        if data:
            all_data[symbol] = data
            print(f"✓ Successfully fetched {len(data)} days of data for {symbol}")
        else:
            print(f"✗ Failed to fetch data for {symbol}")
        
        # Esperar para no exceder rate limits
        time.sleep(1)
    
    # Guardar datos
    with open('crypto_historical_data.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n✓ Data saved to crypto_historical_data.json")
    print(f"Total cryptocurrencies: {len(all_data)}")

if __name__ == "__main__":
    main()
