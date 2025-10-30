"use client"

interface ConfidenceGaugeProps {
  confidence: number
}

export function ConfidenceGauge({ confidence }: ConfidenceGaugeProps) {
  const getColor = (value: number) => {
    if (value >= 70) return "text-accent"
    if (value >= 50) return "text-primary"
    return "text-destructive"
  }

  const getLabel = (value: number) => {
    if (value >= 70) return "Alta"
    if (value >= 50) return "Media"
    return "Baja"
  }

  const getDescription = (value: number) => {
    if (value >= 70) return "Predicción altamente confiable"
    if (value >= 50) return "Predicción con confianza moderada"
    return "Predicción con alta incertidumbre"
  }

  return (
    <div className="flex flex-col items-center justify-center py-6">
      <div className="relative h-48 w-48">
        <svg className="h-full w-full -rotate-90" viewBox="0 0 100 100">
          {/* Background circle */}
          <circle cx="50" cy="50" r="42" fill="none" stroke="currentColor" strokeWidth="10" className="text-muted/30" />
          {/* Progress circle */}
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke="currentColor"
            strokeWidth="10"
            strokeDasharray={`${confidence * 2.64} 264`}
            strokeLinecap="round"
            className={`${getColor(confidence)} transition-all duration-1000`}
            style={{
              filter: "drop-shadow(0 0 8px currentColor)",
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-5xl font-bold tracking-tight ${getColor(confidence)}`}>{confidence}%</span>
          <span className="text-sm font-semibold text-muted-foreground mt-1 uppercase tracking-wide">
            {getLabel(confidence)}
          </span>
        </div>
      </div>
      <div className="mt-6 text-center max-w-xs">
        <p className="text-sm font-medium text-muted-foreground leading-relaxed">{getDescription(confidence)}</p>
      </div>
    </div>
  )
}
