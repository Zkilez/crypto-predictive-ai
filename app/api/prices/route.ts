import { NextResponse } from "next/server"

const BINANCE_BASE = "https://api.binance.com/api/v3/ticker/24hr"

const SYMBOLS = [
  { symbol: "BTC", pair: "BTCUSDT" },
  { symbol: "ETH", pair: "ETHUSDT" },
  { symbol: "SOL", pair: "SOLUSDT" },
  { symbol: "BNB", pair: "BNBUSDT" },
]

export async function GET() {
  try {
    const results = await Promise.all(
      SYMBOLS.map(async ({ symbol, pair }) => {
        const res = await fetch(`${BINANCE_BASE}?symbol=${pair}`)
        if (!res.ok) throw new Error(`Binance error for ${pair}: ${res.status}`)
        const data = await res.json()
        const price = parseFloat(data.lastPrice)
        const priceChange24h = parseFloat(data.priceChangePercent)
        const volume24h = parseFloat(data.quoteVolume)
        const high24h = parseFloat(data.highPrice)
        const low24h = parseFloat(data.lowPrice)
        const openPrice = parseFloat(data.openPrice)
        return [symbol, { price, priceChange24h, volume24h, high24h, low24h, openPrice, exchange: "binance", pair }]
      }),
    )
    const prices = Object.fromEntries(results)
    return NextResponse.json({ success: true, prices, timestamp: new Date().toISOString() })
  } catch (error) {
    console.error("Error fetching prices:", error)
    return NextResponse.json(
      { success: false, message: "Failed to fetch prices", prices: {} },
      { status: 500 },
    )
  }
}