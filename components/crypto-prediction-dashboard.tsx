"use client"

import { useEffect, useState } from "react"
import { formatTimeUTC } from "@/lib/utils"
import useSWR from "swr"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { TrendingUp, TrendingDown, Activity, Clock, Target, BarChart3, Sparkles, Network } from "lucide-react"
import { PredictionChart } from "@/components/prediction-chart"
import { ConfidenceGauge } from "@/components/confidence-gauge"
import { PriceHistory } from "@/components/price-history"
import { CorrelationHeatmap } from "@/components/correlation-heatmap"
import { OrderBook } from "@/components/order-book"
import { VolumeChart } from "@/components/volume-chart"
import { ProbabilityDistribution } from "@/components/probability-distribution"

const fetcher = (url: string) => fetch(url).then((res) => res.json())

export function CryptoPredictionDashboard() {
  const { data, error, isLoading, mutate } = useSWR("/api/predictions", fetcher, {
    refreshInterval: 60000, // Refrescar cada minuto
    revalidateOnFocus: true,
  })

  const { data: pricesData } = useSWR("/api/prices", fetcher, {
    refreshInterval: 15000, // Precios en tiempo casi real
    revalidateOnFocus: true,
  })

  // Tipos de cambio para convertir desde USD a otras monedas
  const { data: fxData, mutate: mutateFx } = useSWR("/api/fx", fetcher, {
    refreshInterval: 60 * 60 * 1000, // Actualizar cada hora
    revalidateOnFocus: false,
  })

  const [timeframe, setTimeframe] = useState("24h")
  const [selectedCurrency, setSelectedCurrency] = useState<string>("USD")

  // Cargar preferencia y refrescar tasas al cambiar moneda
  useEffect(() => {
    try {
      const saved = typeof window !== "undefined" ? localStorage.getItem("currency") : null
      if (saved) setSelectedCurrency(saved)
    } catch {}
  }, [])
  useEffect(() => {
    try {
      if (typeof window !== "undefined") localStorage.setItem("currency", selectedCurrency)
    } catch {}
    mutateFx && mutateFx()
  }, [selectedCurrency, mutateFx])

  const fxRates: Record<string, number | null> = fxData?.rates || { USD: 1 }
  const rate = fxRates[selectedCurrency] ?? 1
  const convert = (usd: number) => (usd ?? 0) * (rate || 1)
  const localeByCurrency: Record<string, string> = {
    USD: "en-US",
    COP: "es-CO",
    EUR: "es-ES",
    MXN: "es-MX",
    BRL: "pt-BR",
  }
  const formatCurrency = (value: number) =>
    new Intl.NumberFormat(localeByCurrency[selectedCurrency] || "en-US", {
      style: "currency",
      currency: selectedCurrency,
      currencyDisplay: "code", // Muestra el código explícito (p. ej., COP, USD)
      maximumFractionDigits: selectedCurrency === "COP" ? 0 : 2,
    }).format(value)

  const formatCurrencyCompact = (value: number) =>
    new Intl.NumberFormat(localeByCurrency[selectedCurrency] || "en-US", {
      style: "currency",
      currency: selectedCurrency,
      currencyDisplay: "code",
      notation: "compact",
      maximumFractionDigits: selectedCurrency === "COP" ? 0 : 2,
    }).format(value)

  const cryptoData = data?.predictions && typeof data.predictions === "object"
    ? Object.entries<any>(data.predictions).map(([symbol, pred]) => {
        if (!pred) return null
        const live = pricesData?.prices?.[symbol]
        const symbolName =
          symbol === "BTC"
            ? "Bitcoin"
            : symbol === "ETH"
              ? "Ethereum"
              : symbol === "SOL"
                ? "Solana"
                : symbol === "BNB"
                  ? "Binance Coin"
                  : symbol
        const rawConfidence = pred?.confidence ?? 0
        const confidencePct = rawConfidence > 1 ? Math.round(rawConfidence) : Math.round(rawConfidence * 100)
        const predictedChange = (pred?.prediction || "").toUpperCase() === "BULLISH"
          ? Math.abs(confidencePct / 15)
          : -Math.abs(confidencePct / 15)
      return {
        id: symbol,
        name: symbolName,
        symbol,
        // Preferir precio en tiempo real; fallback a predicción
        currentPrice: (live?.price ?? pred?.current_price ?? 0),
        prediction: (pred?.prediction || "").toLowerCase() === "bullish" ? "bullish" : "bearish",
        confidence: confidencePct,
        // Usar cambio 24h del feed en vivo si está disponible
        priceChange24h: (live?.priceChange24h ?? pred?.price_change_24h ?? 0),
        predictedChange,
        // Marcar última actualización con timestamp de precios si existe
        lastUpdateTs:
          (pricesData?.timestamp ? Date.parse(pricesData.timestamp) : undefined) ?? pred?.timestamp ?? Date.now(),
        // Guardar valores numéricos (USD) para conversión
        volumeUsd: (live?.volume24h ?? pred?.volume_24h ?? null),
        marketCapUsd: pred?.market_cap ?? null,
        // Fallback en USD abreviado
        volume24h: pred?.volume_24h ? `${(pred.volume_24h / 1e9).toFixed(1)}B` : "-",
        marketCap: pred?.market_cap ? `${(pred.market_cap / 1e9).toFixed(1)}B` : "-",
        rsi: pred?.rsi ?? null,
          // Rangos de precio en vivo (USD)
          high24h: live?.high24h ?? null,
          low24h: live?.low24h ?? null,
          openPrice: live?.openPrice ?? null,
          // Nuevos campos
          futureVolatility: pred?.future_volatility_prediction || null,
          futureVolatilityConfidence:
            pred?.future_volatility_confidence != null
              ? (pred.future_volatility_confidence > 1
                  ? Math.round(pred.future_volatility_confidence)
                  : Math.round(pred.future_volatility_confidence * 100))
              : null,
          socialSentiment:
            pred?.social_sentiment_signal === 1
              ? "positivo"
              : pred?.social_sentiment_signal === -1
                ? "negativo"
                : pred?.social_sentiment_signal === 0
                  ? "neutral"
                  : null,
          socialSentimentStrength:
            pred?.social_sentiment_strength != null
              ? (pred.social_sentiment_strength > 1
                  ? Math.round(pred.social_sentiment_strength)
                  : Math.round(pred.social_sentiment_strength * 100))
              : null,
        socialMentionVolume: pred?.social_mention_volume ?? null,
      }
    }).filter(Boolean)
    : []

  // Mezclar precios en tiempo real del endpoint /api/prices
  const livePrices = pricesData?.prices || {}
  const displayData = cryptoData.map((c: any) => {
    const lp = livePrices[c.symbol]
    if (lp) {
      const volumeStr = lp.volume24h ? `${(lp.volume24h / 1e9).toFixed(1)}B` : c.volume24h
      return {
        ...c,
        currentPrice: lp.price ?? c.currentPrice,
        priceChange24h: lp.priceChange24h ?? c.priceChange24h,
        volume24h: volumeStr,
        volumeUsd: lp.volume24h ?? c.volumeUsd ?? null,
        high24h: lp.high24h ?? null,
        low24h: lp.low24h ?? null,
        openPrice: lp.openPrice ?? null,
      }
    }
    return c
  })

  const [selectedCrypto, setSelectedCrypto] = useState<any>(displayData[0] || null)

  // Calcular la hora "Actualizado" solo en cliente para evitar desajustes de hidratación
  const [lastUpdateDisplay, setLastUpdateDisplay] = useState<string>("")
  useEffect(() => {
    const ts = selectedCrypto?.lastUpdateTs
    if (ts != null) {
      setLastUpdateDisplay(formatTimeUTC(ts))
    } else {
      setLastUpdateDisplay("")
    }
  }, [selectedCrypto?.lastUpdateTs])

  useEffect(() => {
    if (displayData.length > 0 && !selectedCrypto) {
      setSelectedCrypto(displayData[0])
    } else if (selectedCrypto) {
      const updated = displayData.find((c: any) => c.id === selectedCrypto.id)
      if (updated) setSelectedCrypto(updated)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayData.length, pricesData])

  // Evita leer propiedades de undefined en el primer render antes de que useEffect establezca selectedCrypto
  if (cryptoData.length > 0 && !selectedCrypto) {
    return null
  }

  const bullishCount = cryptoData.filter((c: any) => c.prediction === "bullish").length
  const bearishCount = cryptoData.filter((c: any) => c.prediction === "bearish").length

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 p-6 md:p-10" suppressHydrationWarning>
        <div className="mx-auto max-w-[1600px] flex items-center justify-center min-h-[60vh]">
          <div className="text-center space-y-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-primary/10 ring-1 ring-primary/20 mx-auto animate-pulse">
              <Sparkles className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h2 className="text-2xl font-bold">Cargando predicciones...</h2>
              <p className="text-muted-foreground mt-2">Obteniendo datos del modelo de IA</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (error || !data?.success || cryptoData.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 p-6 md:p-10" suppressHydrationWarning>
        <div className="mx-auto max-w-[1600px] flex items-center justify-center min-h-[60vh]">
          <Card className="max-w-2xl w-full border-2">
            <CardHeader>
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-destructive/10 ring-1 ring-destructive/20 mb-4">
                <Activity className="h-6 w-6 text-destructive" />
              </div>
              <CardTitle className="text-2xl">No hay predicciones disponibles</CardTitle>
              <CardDescription className="text-base">
                El modelo de IA aún no ha generado predicciones. Sigue estos pasos:
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-lg bg-muted/50 p-4 border space-y-3">
                <div className="flex items-start gap-3">
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">
                    1
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold">Obtener datos históricos</p>
                    <code className="text-sm bg-background px-2 py-1 rounded mt-1 block">
                      python scripts/fetch_crypto_data.py
                    </code>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">
                    2
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold">Entrenar el modelo</p>
                    <code className="text-sm bg-background px-2 py-1 rounded mt-1 block">
                      python scripts/train_model.py
                    </code>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">
                    3
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold">Generar predicciones</p>
                    <code className="text-sm bg-background px-2 py-1 rounded mt-1 block">
                      python scripts/predict.py
                    </code>
                  </div>
                </div>
              </div>
              <Button onClick={() => mutate()} className="w-full" size="lg">
                <Activity className="mr-2 h-5 w-5" />
                Reintentar Carga
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 p-6 md:p-10">
      <div className="mx-auto max-w-[1600px] space-y-8">
        <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 ring-1 ring-primary/20">
                <Sparkles className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-balance">Predictor de Criptomonedas</h1>
                <p className="text-muted-foreground mt-1 text-base">
                  Análisis predictivo impulsado por inteligencia artificial
                </p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger className="w-36 h-11 font-medium">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1h">1 Hora</SelectItem>
                <SelectItem value="24h">24 Horas</SelectItem>
                <SelectItem value="7d">7 Días</SelectItem>
                <SelectItem value="30d">30 Días</SelectItem>
              </SelectContent>
            </Select>
            <Button
              variant="outline"
              size="icon"
              className="h-11 w-11 bg-transparent"
              onClick={() => mutate()}
              title="Refrescar predicciones"
            >
              <Activity className="h-5 w-5" />
            </Button>
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <Card className="border-2 hover:border-accent/50 transition-colors">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                Predicciones Alcistas
              </CardTitle>
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10">
                <TrendingUp className="h-5 w-5 text-accent" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-bold tracking-tight">{bullishCount}</div>
              <p className="text-sm text-muted-foreground mt-2 font-medium">
                {((bullishCount / cryptoData.length) * 100).toFixed(0)}% del portafolio
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 hover:border-destructive/50 transition-colors">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                Predicciones Bajistas
              </CardTitle>
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-destructive/10">
                <TrendingDown className="h-5 w-5 text-destructive" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-bold tracking-tight">{bearishCount}</div>
              <p className="text-sm text-muted-foreground mt-2 font-medium">
                {((bearishCount / cryptoData.length) * 100).toFixed(0)}% del portafolio
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 hover:border-primary/50 transition-colors">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                Confianza Promedio
              </CardTitle>
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Target className="h-5 w-5 text-primary" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-bold tracking-tight">
                {(cryptoData.reduce((acc, c) => acc + c.confidence, 0) / cryptoData.length).toFixed(0)}%
              </div>
              <p className="text-sm text-muted-foreground mt-2 font-medium">Precisión del modelo IA</p>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-6 lg:grid-cols-[400px_1fr]">
          <Card className="border-2">
            <CardHeader className="pb-4">
              <CardTitle className="text-xl">Activos Monitoreados</CardTitle>
              <CardDescription>Selecciona un activo para análisis detallado</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {displayData.map((crypto) => (
                <button
                  key={crypto.id}
                  onClick={() => setSelectedCrypto(crypto)}
                  className={`w-full rounded-xl border-2 p-5 text-left transition-all hover:shadow-md ${
                    selectedCrypto && selectedCrypto.id === crypto.id
                      ? "border-primary bg-primary/5 shadow-sm"
                      : "border-border hover:border-muted-foreground/30"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-bold text-lg">{crypto.symbol}</span>
                        <Badge
                          variant={crypto.prediction === "bullish" ? "default" : "destructive"}
                          className="text-xs font-semibold"
                        >
                          {crypto.prediction === "bullish" ? (
                            <TrendingUp className="mr-1 h-3 w-3" />
                          ) : (
                            <TrendingDown className="mr-1 h-3 w-3" />
                          )}
                          {crypto.prediction === "bullish" ? "Alcista" : "Bajista"}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground font-medium mb-3">{crypto.name}</p>
                      <div className="flex items-baseline gap-2">
                        <span className="text-xl font-bold tracking-tight">
                          {formatCurrency(convert(crypto.currentPrice))}
                        </span>
                        <span
                          className={`text-sm font-semibold ${
                            crypto.priceChange24h >= 0 ? "text-accent" : "text-destructive"
                          }`}
                        >
                          {crypto.priceChange24h >= 0 ? "+" : ""}
                          {crypto.priceChange24h}%
                        </span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold text-primary tracking-tight">{crypto.confidence}%</div>
                      <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">Confianza</p>
                    </div>
                  </div>
                </button>
              ))}
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card className="border-2">
              <CardHeader className="pb-6">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <CardTitle className="text-3xl font-bold tracking-tight mb-2">{selectedCrypto.name}</CardTitle>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <span className="font-mono font-semibold text-base text-foreground">{selectedCrypto.symbol}</span>
                      <span className="flex items-center gap-1.5" suppressHydrationWarning>
                        <Clock className="h-3.5 w-3.5" />
                        Actualizado {lastUpdateDisplay || "—"}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-[160px]">
                      <Select value={selectedCurrency} onValueChange={setSelectedCurrency}>
                        <SelectTrigger className="h-10">
                          <SelectValue placeholder="Moneda" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="USD">USD — Dólar</SelectItem>
                          <SelectItem value="COP">COP — Peso Colombiano</SelectItem>
                          <SelectItem value="EUR">EUR — Euro</SelectItem>
                          <SelectItem value="MXN">MXN — Peso Mexicano</SelectItem>
                          <SelectItem value="BRL">BRL — Real Brasileño</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Badge
                      variant={selectedCrypto.prediction === "bullish" ? "default" : "destructive"}
                      className="text-base px-5 py-2.5 font-semibold"
                    >
                      {selectedCrypto.prediction === "bullish" ? (
                        <TrendingUp className="mr-2 h-5 w-5" />
                      ) : (
                        <TrendingDown className="mr-2 h-5 w-5" />
                      )}
                      {selectedCrypto.prediction === "bullish" ? "Alcista" : "Bajista"}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-8 md:grid-cols-2">
                  <div className="space-y-2">
                    <p className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Precio Actual</p>
                    <p className="text-4xl font-bold tracking-tight">{formatCurrency(convert(selectedCrypto.currentPrice))}</p>
                    <p className="text-xs text-muted-foreground">Moneda seleccionada: {selectedCurrency} · Tasa: 1 USD = {rate ? rate.toLocaleString(localeByCurrency[selectedCurrency] || "en-US") : "--"} {selectedCurrency}</p>
                    <p
                      className={`text-base font-semibold ${
                        selectedCrypto.priceChange24h >= 0 ? "text-accent" : "text-destructive"
                      }`}
                    >
                      {selectedCrypto.priceChange24h >= 0 ? "+" : ""}
                      {selectedCrypto.priceChange24h}% en 24h
                    </p>
                    <div className="pt-2 space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Volumen 24h ({selectedCurrency}):</span>
                        <span className="font-semibold" title={selectedCrypto?.volumeUsd ? formatCurrency(convert(selectedCrypto.volumeUsd)) : undefined}>
                          {selectedCrypto?.volumeUsd
                            ? formatCurrencyCompact(convert(selectedCrypto.volumeUsd))
                            : `$${selectedCrypto.volume24h}`}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Cap. de Mercado ({selectedCurrency}):</span>
                        <span className="font-semibold" title={selectedCrypto?.marketCapUsd ? formatCurrency(convert(selectedCrypto.marketCapUsd)) : undefined}>
                          {selectedCrypto?.marketCapUsd
                            ? formatCurrencyCompact(convert(selectedCrypto.marketCapUsd))
                            : `$${selectedCrypto.marketCap}`}
                        </span>
                      </div>
                      {selectedCrypto.high24h && selectedCrypto.low24h && (
                        <>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Máximo 24h:</span>
                            <span className="font-semibold text-accent">{formatCurrency(convert(selectedCrypto.high24h))}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Mínimo 24h:</span>
                            <span className="font-semibold text-destructive">{formatCurrency(convert(selectedCrypto.low24h))}</span>
                          </div>
                        </>
                      )}
                      {selectedCrypto.openPrice && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Precio apertura:</span>
                          <span className="font-semibold">{formatCurrency(convert(selectedCrypto.openPrice))}</span>
                        </div>
                      )}
                    </div>

                    {/* Volatilidad futura */}
                    {selectedCrypto.futureVolatility && (
                      <div className="mt-4 grid gap-2">
                        <p className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                          Volatilidad Futura
                        </p>
                        <div className="flex items-center justify-between">
                          <span className="font-medium capitalize">{selectedCrypto.futureVolatility}</span>
                          {selectedCrypto.futureVolatilityConfidence != null && (
                            <span className="text-sm font-semibold text-muted-foreground">
                              Conf.: {selectedCrypto.futureVolatilityConfidence}%
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                      Cambio Predicho
                    </p>
                    <p
                      className={`text-4xl font-bold tracking-tight ${
                        selectedCrypto.predictedChange >= 0 ? "text-accent" : "text-destructive"
                      }`}
                    >
                      {selectedCrypto.predictedChange >= 0 ? "+" : ""}
                      {selectedCrypto.predictedChange}%
                    </p>
                    <p className="text-base font-medium text-muted-foreground">Proyección para {timeframe}</p>
                    <div className="pt-2">
                      <div className="rounded-lg bg-muted/50 p-3 border">
                        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                          Precio Objetivo
                        </p>
                        <p className="text-xl font-bold">
                          {formatCurrency(
                            convert(selectedCrypto.currentPrice * (1 + selectedCrypto.predictedChange / 100)),
                          )}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid gap-6 md:grid-cols-2">
              <Card className="border-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
                      <Target className="h-5 w-5 text-primary" />
                    </div>
                    Confianza del Modelo
                  </CardTitle>
                  <CardDescription>Nivel de certeza de la predicción IA</CardDescription>
                </CardHeader>
                <CardContent>
                  <ConfidenceGauge confidence={selectedCrypto.confidence} />
                </CardContent>
              </Card>

              <Card className="border-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
                      <BarChart3 className="h-5 w-5 text-primary" />
                    </div>
                    Tendencia Predicha
                  </CardTitle>
                  <CardDescription>Proyección del modelo para {timeframe}</CardDescription>
                </CardHeader>
                <CardContent>
                  <PredictionChart
                    prediction={selectedCrypto.prediction}
                    predictedChange={selectedCrypto.predictedChange}
                  />
                </CardContent>
              </Card>

              {/* Sentimiento social */}
              {selectedCrypto.socialSentiment && (
                <Card className="border-2">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
                        <Activity className="h-5 w-5 text-primary" />
                      </div>
                      Sentimiento Social
                    </CardTitle>
                    <CardDescription>
                      Señales agregadas de redes sociales y volumen de menciones
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-3 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Dirección:</span>
                        <span className="font-semibold capitalize">{selectedCrypto.socialSentiment}</span>
                      </div>
                      {selectedCrypto.socialSentimentStrength != null && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Fuerza:</span>
                          <span className="font-semibold">{selectedCrypto.socialSentimentStrength}%</span>
                        </div>
                      )}
                      {selectedCrypto.socialMentionVolume != null && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Menciones:</span>
                          <span className="font-semibold">{selectedCrypto.socialMentionVolume.toLocaleString()}</span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-xl">Análisis de Probabilidad</CardTitle>
                <CardDescription>Distribución estadística del cambio de precio predicho</CardDescription>
              </CardHeader>
              <CardContent>
                <ProbabilityDistribution
                  prediction={selectedCrypto.prediction}
                  confidence={selectedCrypto.confidence}
                  predictedChange={selectedCrypto.predictedChange}
                />
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-xl">Análisis de Volumen</CardTitle>
                <CardDescription>Actividad de trading en las últimas 24 horas</CardDescription>
              </CardHeader>
              <CardContent>
                <VolumeChart symbol={selectedCrypto.symbol} />
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-xl">Historial y Proyección</CardTitle>
                <CardDescription>Datos históricos de 24h con predicción del modelo</CardDescription>
              </CardHeader>
              <CardContent>
                <PriceHistory
                  symbol={selectedCrypto.symbol}
                  currentPrice={selectedCrypto.currentPrice}
                  prediction={selectedCrypto.prediction}
                  selectedCurrency={selectedCurrency}
                  rate={rate || 1}
                />
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-xl">Profundidad de Mercado</CardTitle>
                <CardDescription>Órdenes de compra y venta en tiempo real</CardDescription>
              </CardHeader>
              <CardContent>
                <OrderBook
                  symbol={selectedCrypto.symbol}
                  currentPrice={selectedCrypto.currentPrice}
                  selectedCurrency={selectedCurrency}
                  rate={rate || 1}
                />
              </CardContent>
            </Card>
          </div>
        </div>

        <Card className="border-2">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Network className="h-5 w-5 text-primary" />
              </div>
              <div>
                <CardTitle className="text-xl">Matriz de Correlación</CardTitle>
                <CardDescription>Relación entre movimientos de diferentes criptomonedas</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <CorrelationHeatmap cryptos={cryptoData.map((c) => c.symbol)} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
