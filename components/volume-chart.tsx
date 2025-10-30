"use client"

import { BarChart3 } from "lucide-react"

interface VolumeChartProps {
  symbol: string
}

export function VolumeChart({ symbol }: VolumeChartProps) {
  // Generar datos de volumen simulados para 24 horas
  const volumeData = Array.from({ length: 24 }, (_, i) => ({
    hour: i,
    volume: Math.random() * 100 + 20,
    trend: Math.random() > 0.5 ? "up" : "down",
  }))

  const maxVolume = Math.max(...volumeData.map((d) => d.volume))

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
            <BarChart3 className="h-4 w-4 text-primary" />
          </div>
          <span className="text-sm font-semibold">Volumen de Trading (24h)</span>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded bg-accent" />
            <span className="text-muted-foreground">Alcista</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded bg-destructive" />
            <span className="text-muted-foreground">Bajista</span>
          </div>
        </div>
      </div>

      <div className="flex items-end justify-between gap-1 h-48">
        {volumeData.map((data, i) => (
          <div key={i} className="flex-1 flex flex-col items-center gap-1 group">
            <div
              className={`w-full rounded-t transition-all hover:opacity-80 ${
                data.trend === "up" ? "bg-accent" : "bg-destructive"
              }`}
              style={{ height: `${(data.volume / maxVolume) * 100}%` }}
              title={`${data.hour}:00 - Vol: ${data.volume.toFixed(1)}M`}
            />
            {i % 4 === 0 && <span className="text-[10px] text-muted-foreground font-medium">{data.hour}h</span>}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-4 pt-2 border-t">
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Volumen Total</p>
          <p className="text-lg font-bold">
            $
            {(volumeData.reduce((acc, d) => acc + d.volume, 0) * 1000000).toLocaleString(undefined, {
              maximumFractionDigits: 0,
            })}
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Volumen Promedio</p>
          <p className="text-lg font-bold">
            $
            {((volumeData.reduce((acc, d) => acc + d.volume, 0) / volumeData.length) * 1000000).toLocaleString(
              undefined,
              { maximumFractionDigits: 0 },
            )}
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Pico Máximo</p>
          <p className="text-lg font-bold">
            ${(maxVolume * 1000000).toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </p>
        </div>
      </div>
    </div>
  )
}
