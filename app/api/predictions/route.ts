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
    return NextResponse.json(
      {
        success: false,
        message: "No predictions available. Run the Python scripts first.",
        predictions: {},
      },
      { status: 404 },
    )
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
