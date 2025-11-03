import { NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { join } from "path"

export async function GET() {
  try {
    // Leer predicciones del archivo generado por el script de Python
    const predictionsPath = join(process.cwd(), "predictions.json")
    const predictionsData = await readFile(predictionsPath, "utf-8")
    const predictions = JSON.parse(predictionsData)

    return NextResponse.json({
      success: true,
      predictions,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Error loading predictions:", error)

    // Si no hay predicciones, devolver datos de ejemplo
    const examplePredictions = {
      "BTC": {
        "symbol": "BTC",
        "prediction": "bullish",
        "confidence": 0.65,
        "current_price": 68500,
        "price_change_24h": 2.3,
        "volume_24h": 28500000000,
        "market_cap": 1350000000000,
        "rsi": 58,
        "timestamp": Date.now(),
        "future_volatility_prediction": "medium",
        "future_volatility_confidence": 0.72,
        "decision_threshold": 0.55
      },
      "ETH": {
        "symbol": "ETH",
        "prediction": "bullish",
        "confidence": 0.59,
        "current_price": 3450,
        "price_change_24h": 1.8,
        "volume_24h": 15700000000,
        "market_cap": 415000000000,
        "rsi": 55,
        "timestamp": Date.now(),
        "future_volatility_prediction": "low",
        "future_volatility_confidence": 0.68,
        "decision_threshold": 0.52
      },
      "SOL": {
        "symbol": "SOL",
        "prediction": "bearish",
        "confidence": 0.61,
        "current_price": 142,
        "price_change_24h": -1.2,
        "volume_24h": 3800000000,
        "market_cap": 62000000000,
        "rsi": 42,
        "timestamp": Date.now(),
        "future_volatility_prediction": "high",
        "future_volatility_confidence": 0.75,
        "decision_threshold": 0.58
      },
      "BNB": {
        "symbol": "BNB",
        "prediction": "bullish",
        "confidence": 0.53,
        "current_price": 580,
        "price_change_24h": 0.5,
        "volume_24h": 1900000000,
        "market_cap": 89000000000,
        "rsi": 51,
        "timestamp": Date.now(),
        "future_volatility_prediction": "medium",
        "future_volatility_confidence": 0.63,
        "decision_threshold": 0.51
      }
    };

    return NextResponse.json({
      success: true,
      message: "Usando predicciones de ejemplo. Para datos reales, ejecute los scripts de Python.",
      predictions: examplePredictions,
      timestamp: new Date().toISOString(),
    })
  }
}

export async function POST() {
  try {
    // Endpoint para triggear una nueva predicción
    // En producción, esto ejecutaría el script de Python
    return NextResponse.json({
      success: true,
      message: "Prediction job queued. Run: python scripts/predict.py",
    })
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        message: "Error queuing prediction job",
      },
      { status: 500 },
    )
  }
}
