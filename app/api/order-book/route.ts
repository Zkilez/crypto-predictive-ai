import { NextResponse } from "next/server"

const BINANCE_API = "https://api.binance.com/api/v3/depth"

// Mapeo de símbolos a pares de trading de Binance
const SYMBOL_PAIRS = {
  BTC: "BTCUSDT",
  ETH: "ETHUSDT",
  SOL: "SOLUSDT",
  BNB: "BNBUSDT",
  ADA: "ADAUSDT",
  XRP: "XRPUSDT",
  DOGE: "DOGEUSDT",
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url)
  const symbol = searchParams.get("symbol") || "BTC"
  const limit = searchParams.get("limit") || "50"

  // Convertir símbolo al par de trading de Binance
  const pair = SYMBOL_PAIRS[symbol as keyof typeof SYMBOL_PAIRS]
  if (!pair) {
    return NextResponse.json(
      { success: false, error: `Symbol ${symbol} not supported` },
      { status: 400 }
    )
  }

  try {
    const url = `${BINANCE_API}?symbol=${pair}&limit=${limit}`
    
    const res = await fetch(url, { 
      next: { revalidate: 5 } // Datos muy volátiles, cache por solo 5 segundos
    })

    if (!res.ok) {
      throw new Error(`Binance API error: ${res.status} ${await res.text()}`)
    }

    const data = await res.json()
    
    // Transformar datos al formato esperado por el frontend
    const bids = data.bids.map((bid: string[]) => ({
      price: parseFloat(bid[0]),
      quantity: parseFloat(bid[1]),
    }))

    const asks = data.asks.map((ask: string[]) => ({
      price: parseFloat(ask[0]),
      quantity: parseFloat(ask[1]),
    }))

    return NextResponse.json({
      success: true,
      symbol,
      pair,
      bids,
      asks,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Error fetching order book:", error)
    return NextResponse.json(
      { success: false, message: "Failed to fetch order book data", bids: [], asks: [] },
      { status: 500 },
    )
  }
}