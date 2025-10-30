import { NextResponse } from "next/server"

export async function POST() {
  try {
    // Endpoint para triggear el entrenamiento del modelo
    // En producción, esto ejecutaría los scripts de Python en secuencia
    return NextResponse.json({
      success: true,
      message: "Training job queued",
      steps: [
        "1. Run: python scripts/fetch_crypto_data.py",
        "2. Run: python scripts/train_model.py",
        "3. Run: python scripts/predict.py",
      ],
    })
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        message: "Error queuing training job",
      },
      { status: 500 },
    )
  }
}
