"use client"

interface CorrelationHeatmapProps {
  cryptos: string[]
}

export function CorrelationHeatmap({ cryptos }: CorrelationHeatmapProps) {
  // Generar datos de correlación simulados
  const generateCorrelation = () => Math.random() * 2 - 1

  const correlationMatrix = cryptos.map(() => cryptos.map(() => generateCorrelation()))

  const getColor = (value: number) => {
    if (value > 0.7) return "bg-accent/90"
    if (value > 0.4) return "bg-accent/60"
    if (value > 0) return "bg-accent/30"
    if (value > -0.4) return "bg-destructive/30"
    if (value > -0.7) return "bg-destructive/60"
    return "bg-destructive/90"
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-1" style={{ gridTemplateColumns: `60px repeat(${cryptos.length}, 1fr)` }}>
        <div />
        {cryptos.map((crypto) => (
          <div key={crypto} className="text-center text-xs font-bold text-muted-foreground">
            {crypto}
          </div>
        ))}

        {cryptos.map((rowCrypto, i) => (
          <div key={rowCrypto} className="contents">
            <div className="flex items-center justify-end pr-2 text-xs font-bold text-muted-foreground">
              {rowCrypto}
            </div>
            {correlationMatrix[i].map((value, j) => (
              <div
                key={j}
                className={`aspect-square rounded-md ${getColor(value)} flex items-center justify-center text-xs font-semibold transition-all hover:scale-110 hover:z-10 hover:shadow-lg cursor-pointer ${
                  i === j ? "ring-2 ring-primary" : ""
                }`}
                title={`${cryptos[i]} vs ${cryptos[j]}: ${value.toFixed(2)}`}
              >
                {i === j ? "1.0" : value.toFixed(1)}
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="flex items-center justify-center gap-6 pt-2">
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 rounded bg-destructive/90" />
          <span className="text-xs text-muted-foreground">Correlación Negativa</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 rounded bg-muted" />
          <span className="text-xs text-muted-foreground">Sin Correlación</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 rounded bg-accent/90" />
          <span className="text-xs text-muted-foreground">Correlación Positiva</span>
        </div>
      </div>
    </div>
  )
}
