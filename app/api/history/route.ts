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
  const symbol = searchParams.get("symbol") || "BTC"
  const vs = searchParams.get("vs_currency") || "usd"
  const days = searchParams.get("days") || "1" // 1 día por defecto
  const interval = searchParams.get("interval") || "hourly" // hourly, daily

  // Convertir símbolo a ID de CoinGecko
  const id = COIN_MAP[symbol as keyof typeof COIN_MAP]
  if (!id) {
    return NextResponse.json(
      { success: false, error: `Symbol ${symbol} not supported` },
      { status: 400 }
    )
  }

  try {
    const url = `${COINGECKO}/coins/${id}/market_chart?vs_currency=${vs}&days=${days}&interval=${interval}`
    
    const res = await fetch(url, {
      headers: { "User-Agent": "crypto-predictor/1.0" },
      next: { revalidate: 60 }, // Cache por 60 segundos
    })

    if (!res.ok) {
      throw new Error(`CoinGecko error: ${res.status} ${await res.text()}`)
    }

    const data = await res.json()
    
    // Transformar datos al formato esperado por el frontend
    const prices = data.prices.map(([timestamp, price]: [number, number]) => ({
      timestamp,
      price,
      date: new Date(timestamp).toISOString(),
    }))

    const volumes = data.total_volumes?.map(([timestamp, volume]: [number, number]) => ({
      timestamp,
      volume,
      date: new Date(timestamp).toISOString(),
    }))

    return NextResponse.json({
      success: true,
      symbol,
      vs_currency: vs,
      prices,
      volumes,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Error fetching price history:", error)
    return NextResponse.json(
      { success: false, message: "Failed to fetch price history", prices: [] },
      { status: 500 },
    )
  }
}