import { NextResponse } from "next/server"

// Lista de monedas soportadas para conversión desde USD
const SUPPORTED_CURRENCIES = ["USD", "COP", "EUR", "MXN", "BRL"] as const

type Supported = (typeof SUPPORTED_CURRENCIES)[number]

export async function GET() {
  try {
    // Usamos exchangerate.host que no requiere API key
    const res = await fetch("https://api.exchangerate.host/latest?base=USD&symbols=EUR,MXN,COP,BRL,USD", {
      next: { revalidate: 3600 } // Cache por 1 hora, las tasas no cambian tan rápido
    })
    
    if (!res.ok) throw new Error(`FX error: ${res.status} ${await res.text()}`)
    const data = await res.json()

    const rates: Record<Supported, number> = {
      USD: 1,
      COP: data.rates?.COP ?? null,
      EUR: data.rates?.EUR ?? null,
      MXN: data.rates?.MXN ?? null,
      BRL: data.rates?.BRL ?? null,
    }

    return NextResponse.json({ success: true, rates, timestamp: new Date().toISOString() })
  } catch (error) {
    console.error("Error fetching FX rates:", error)
    // Fallback básico con solo USD si falla el proveedor externo
    const rates = { USD: 1, COP: null, EUR: null, MXN: null, BRL: null }
    return NextResponse.json({ success: false, rates, message: "Failed to fetch FX rates" }, { status: 500 })
  }
}