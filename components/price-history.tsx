"use client"

import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { useEffect, useState } from "react"
import { formatCurrencyByCode } from "@/lib/utils"

interface PriceHistoryProps {
  symbol: string
  currentPrice: number
  prediction: "bullish" | "bearish"
  selectedCurrency?: string
  rate?: number
}

export function PriceHistory({ symbol, currentPrice, prediction, selectedCurrency = "USD", rate = 1 }: PriceHistoryProps) {
  // Generación sólo en cliente para evitar desajustes de hidratación (Math.random/Date.now)
  const [allData, setAllData] = useState<Array<{ time: string; price: number; predicted?: boolean; timestamp: number }>>([])

  useEffect(() => {
    const historicalData = Array.from({ length: 24 }, (_, i) => {
      const variance = (Math.random() - 0.5) * 0.05
      const price = currentPrice * (1 + variance - i * 0.001)
      return {
        time: `${23 - i}h`,
        price: price,
        timestamp: Date.now() - (23 - i) * 3600000,
      }
    }).reverse()

    const futureData = Array.from({ length: 6 }, (_, i) => {
      const trend = prediction === "bullish" ? 0.008 : -0.008
      const price = currentPrice * (1 + trend * (i + 1))
      return {
        time: `+${i + 1}h`,
        price: price,
        predicted: true,
        timestamp: Date.now() + (i + 1) * 3600000,
      }
    })

    setAllData([...historicalData, ...futureData])
  }, [currentPrice, prediction])

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={allData}>
          <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
          <YAxis
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => formatCurrencyByCode(Number(value) * rate, selectedCurrency)}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="rounded-lg border bg-card p-3 shadow-sm">
                    <div className="flex flex-col gap-1">
                      <span className="text-xs text-muted-foreground">
                        {payload[0].payload.predicted ? "Predicción" : "Histórico"}
                      </span>
                      <span className="font-bold">
                        {formatCurrencyByCode(Number(payload[0].value) * rate, selectedCurrency)}
                      </span>
                      <span className="text-xs text-muted-foreground">{payload[0].payload.time}</span>
                    </div>
                  </div>
                )
              }
              return null
            }}
          />
          <Line
            type="monotone"
            dataKey="price"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="price"
            stroke={prediction === "bullish" ? "hsl(var(--accent))" : "hsl(var(--destructive))"}
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            data={allData.filter((d) => d.predicted)}
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-4 flex items-center justify-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-primary" />
          <span className="text-muted-foreground">Histórico</span>
        </div>
        <div className="flex items-center gap-2">
          <div className={`h-3 w-3 rounded-full ${prediction === "bullish" ? "bg-accent" : "bg-destructive"}`} />
          <span className="text-muted-foreground">Predicción</span>
        </div>
      </div>
    </div>
  )
}
