"use client"

import { TrendingUp, TrendingDown } from "lucide-react"

interface ProbabilityDistributionProps {
  prediction: string
  confidence: number
  predictedChange: number
}

export function ProbabilityDistribution({ prediction, confidence, predictedChange }: ProbabilityDistributionProps) {
  // Generar distribución de probabilidad basada en la confianza
  const generateDistribution = () => {
    const points = 50
    const data = []
    const mean = predictedChange
    const stdDev = (100 - confidence) / 10 // Menor confianza = mayor dispersión

    for (let i = 0; i < points; i++) {
      const x = (i / points) * 20 - 10 // Rango de -10% a +10%
      const probability = Math.exp(-Math.pow(x - mean, 2) / (2 * Math.pow(stdDev, 2)))
      data.push({ x, probability })
    }

    return data
  }

  const distribution = generateDistribution()
  const maxProbability = Math.max(...distribution.map((d) => d.probability))

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-sm font-semibold">Distribución de Probabilidad</h4>
          <p className="text-xs text-muted-foreground">Rango de cambios de precio posibles</p>
        </div>
        <div className="text-right">
          <div className="flex items-center gap-1.5">
            {prediction === "bullish" ? (
              <TrendingUp className="h-4 w-4 text-accent" />
            ) : (
              <TrendingDown className="h-4 w-4 text-destructive" />
            )}
            <span className="text-lg font-bold">{confidence}%</span>
          </div>
          <p className="text-xs text-muted-foreground">Confianza</p>
        </div>
      </div>

      <div className="relative h-48 bg-muted/20 rounded-lg p-4">
        {/* Línea central (0%) */}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />

        {/* Curva de distribución */}
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          <defs>
            <linearGradient id="distributionGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop
                offset="0%"
                stopColor={prediction === "bullish" ? "rgb(34, 197, 94)" : "rgb(239, 68, 68)"}
                stopOpacity="0.5"
              />
              <stop
                offset="100%"
                stopColor={prediction === "bullish" ? "rgb(34, 197, 94)" : "rgb(239, 68, 68)"}
                stopOpacity="0.1"
              />
            </linearGradient>
          </defs>
          <path
            d={`M 0 100 ${distribution.map((d, i) => `L ${(i / distribution.length) * 100} ${100 - (d.probability / maxProbability) * 90}`).join(" ")} L 100 100 Z`}
            fill="url(#distributionGradient)"
            stroke={prediction === "bullish" ? "rgb(34, 197, 94)" : "rgb(239, 68, 68)"}
            strokeWidth="0.5"
          />
        </svg>

        {/* Etiquetas del eje X */}
        <div className="absolute bottom-0 left-0 right-0 flex justify-between px-4 text-xs text-muted-foreground font-medium">
          <span>-10%</span>
          <span>-5%</span>
          <span className="font-bold text-foreground">0%</span>
          <span>+5%</span>
          <span>+10%</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 pt-2 border-t">
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Cambio Esperado</p>
          <p className={`text-lg font-bold ${predictedChange >= 0 ? "text-accent" : "text-destructive"}`}>
            {predictedChange >= 0 ? "+" : ""}
            {predictedChange}%
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Rango Probable</p>
          <p className="text-lg font-bold">
            {(predictedChange - 2).toFixed(1)}% - {(predictedChange + 2).toFixed(1)}%
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Volatilidad</p>
          <p className="text-lg font-bold">{((100 - confidence) / 10).toFixed(1)}σ</p>
        </div>
      </div>
    </div>
  )
}
