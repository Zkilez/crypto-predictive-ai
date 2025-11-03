import { NextResponse } from "next/server"

const COINGECKO = "https://api.coingecko.com/api/v3"

// Mapeo de símbolos a IDs de CoinGecko
const COIN_MAP = {
  BTC: "bitcoin",
  ETH: "ethereum",
  SOL: "solana",
  BNB: "binancecoin",
  ADA: "cardano",
  XRP: "ripple",
  DOGE: "dogecoin",
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url)
  const symbols = searchParams.get("symbols")?.split(",") || ["BTC", "ETH", "SOL", "BNB"]
  const vs = searchParams.get("vs_currency") || "usd"
  
  // Convertir símbolos a IDs de CoinGecko
  const ids = symbols
    .map(symbol => COIN_MAP[symbol as keyof typeof COIN_MAP])
    .filter(Boolean)
    .join(",")
  
  try {
    const url = `${COINGECKO}/simple/price?ids=${ids}&vs_currencies=${vs}&include_24hr_change=true&include_market_cap=true&include_last_updated_at=true`
    
    const res = await fetch(url, {
      headers: { "User-Agent": "crypto-predictor/1.0" },
      next: { revalidate: 30 }, // Cache por 30 segundos
    })

    if (!res.ok) {
      throw new Error(`CoinGecko error: ${res.status} ${await res.text()}`)
    }

    const data = await res.json()
    
    // Transformar respuesta de CoinGecko al formato esperado por el frontend
    const prices: Record<string, any> = {}
    
    // Convertir de vuelta a los símbolos originales
    Object.entries(COIN_MAP).forEach(([symbol, id]) => {
      if (data[id]) {
        prices[symbol] = {
          price: data[id][vs],
          priceChange24h: data[id][`${vs}_24h_change`] || 0,
          marketCap: data[id][`${vs}_market_cap`] || 0,
          lastUpdated: data[id].last_updated_at ? new Date(data[id].last_updated_at * 1000).toISOString() : null,
          exchange: "coingecko",
          symbol: symbol
        }
      }
    })
    
    return NextResponse.json({ 
      success: true, 
      prices, 
      timestamp: new Date().toISOString() 
    })
  } catch (error) {
    console.error("Error fetching prices:", error)
    return NextResponse.json(
      { success: false, message: "Failed to fetch cryptocurrency prices", prices: {} },
      { status: 500 },
    )
  }
}