"use client"

import { TrendingUp, TrendingDown } from "lucide-react"

interface PredictionChartProps {
  prediction: "bullish" | "bearish"
  predictedChange: number
}

export function PredictionChart({ prediction, predictedChange }: PredictionChartProps) {
  const isBullish = prediction === "bullish"

  const bars = Array.from({ length: 9 }, (_, i) => {
    const progress = (i + 1) / 9
    return {
      height: isBullish ? progress * 100 : 100 - progress * 100,
      opacity: 0.4 + progress * 0.6,
    }
  })

  return (
    <div className="flex flex-col items-center justify-center py-6">
      <div className="flex items-end justify-center gap-2 h-40 px-4">
        {bars.map((bar, i) => (
          <div
            key={i}
            className={`w-7 rounded-t-lg transition-all duration-500 ${isBullish ? "bg-accent" : "bg-destructive"}`}
            style={{
              height: `${bar.height}%`,
              opacity: bar.opacity,
              transitionDelay: `${i * 50}ms`,
              boxShadow: `0 0 ${bar.opacity * 20}px ${isBullish ? "hsl(var(--accent))" : "hsl(var(--destructive))"}`,
            }}
          />
        ))}
      </div>
      <div className="mt-8 text-center">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div
            className={`flex h-12 w-12 items-center justify-center rounded-xl ${
              isBullish ? "bg-accent/10" : "bg-destructive/10"
            }`}
          >
            {isBullish ? (
              <TrendingUp className="h-6 w-6 text-accent" />
            ) : (
              <TrendingDown className="h-6 w-6 text-destructive" />
            )}
          </div>
          <span className={`text-3xl font-bold tracking-tight ${isBullish ? "text-accent" : "text-destructive"}`}>
            {predictedChange >= 0 ? "+" : ""}
            {predictedChange}%
          </span>
        </div>
        <p className="text-sm font-medium text-muted-foreground leading-relaxed max-w-xs mx-auto">
          {isBullish ? "Tendencia alcista esperada" : "Tendencia bajista esperada"}
        </p>
      </div>
    </div>
  )
}
