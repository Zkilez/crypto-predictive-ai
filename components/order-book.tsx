"use client"

import { ArrowDown, ArrowUp } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { formatCurrencyByCode } from "@/lib/utils"

interface OrderBookProps {
  symbol: string
  currentPrice: number
  selectedCurrency?: string
  rate?: number
}

export function OrderBook({ symbol, currentPrice, selectedCurrency = "USD", rate = 1 }: OrderBookProps) {
  // Generación sólo en cliente para evitar desajustes de hidratación
  const [bids, setBids] = useState<Array<{ price: number; amount: number; total: number }>>([])
  const [asks, setAsks] = useState<Array<{ price: number; amount: number; total: number }>>([])

  useEffect(() => {
    const gen = (isBid: boolean) =>
      Array.from({ length: 8 }, (_, i) => {
        const priceOffset = isBid ? -(i + 1) * 50 : (i + 1) * 50
        const price = currentPrice + priceOffset
        const amount = Math.random() * 2 + 0.5
        const total = price * amount
        return { price, amount, total }
      })
    setBids(gen(true))
    setAsks(gen(false))
  }, [currentPrice])

  const maxTotal = useMemo(() => {
    return Math.max(...bids.map((b) => b.total), ...asks.map((a) => a.total), 0)
  }, [bids, asks])

  return (
    <div className="space-y-4">
      {/* Asks (Ventas) */}
      <div className="space-y-1">
        <div className="flex items-center gap-2 mb-2">
          <ArrowDown className="h-4 w-4 text-destructive" />
          <span className="text-xs font-bold uppercase tracking-wide text-muted-foreground">Órdenes de Venta</span>
        </div>
        <div className="grid grid-cols-3 gap-2 text-xs font-semibold text-muted-foreground mb-1 px-2">
          <div className="text-right">Precio ({selectedCurrency})</div>
          <div className="text-right">Cantidad</div>
          <div className="text-right">Total ({selectedCurrency})</div>
        </div>
        {asks.reverse().map((ask, i) => (
          <div key={i} className="relative">
            <div
              className="absolute inset-0 bg-destructive/10 rounded"
              style={{ width: `${(ask.total / maxTotal) * 100}%` }}
            />
            <div className="relative grid grid-cols-3 gap-2 text-xs py-1.5 px-2 hover:bg-muted/50 rounded transition-colors">
              <div className="text-right font-mono font-semibold text-destructive">
                {formatCurrencyByCode(ask.price * rate, selectedCurrency)}
              </div>
              <div className="text-right font-mono">{ask.amount.toFixed(4)}</div>
              <div className="text-right font-mono text-muted-foreground">
                {formatCurrencyByCode(ask.total * rate, selectedCurrency, { maximumFractionDigits: 0 })}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Precio Actual */}
      <div className="flex items-center justify-center gap-3 py-3 bg-primary/5 rounded-lg border-2 border-primary/20">
        <span className="text-xs font-bold uppercase tracking-wide text-muted-foreground">Precio Actual</span>
        <span className="text-xl font-bold text-primary font-mono">
          {formatCurrencyByCode(currentPrice * rate, selectedCurrency)}
        </span>
      </div>

      {/* Bids (Compras) */}
      <div className="space-y-1">
        <div className="flex items-center gap-2 mb-2">
          <ArrowUp className="h-4 w-4 text-accent" />
          <span className="text-xs font-bold uppercase tracking-wide text-muted-foreground">Órdenes de Compra</span>
        </div>
        <div className="grid grid-cols-3 gap-2 text-xs font-semibold text-muted-foreground mb-1 px-2">
          <div className="text-right">Precio ({selectedCurrency})</div>
          <div className="text-right">Cantidad</div>
          <div className="text-right">Total ({selectedCurrency})</div>
        </div>
        {bids.map((bid, i) => (
          <div key={i} className="relative">
            <div
              className="absolute inset-0 bg-accent/10 rounded"
              style={{ width: `${(bid.total / maxTotal) * 100}%` }}
            />
            <div className="relative grid grid-cols-3 gap-2 text-xs py-1.5 px-2 hover:bg-muted/50 rounded transition-colors">
              <div className="text-right font-mono font-semibold text-accent">
                {formatCurrencyByCode(bid.price * rate, selectedCurrency)}
              </div>
              <div className="text-right font-mono">{bid.amount.toFixed(4)}</div>
              <div className="text-right font-mono text-muted-foreground">
                {formatCurrencyByCode(bid.total * rate, selectedCurrency, { maximumFractionDigits: 0 })}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
