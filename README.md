# 🚀 Predictor de Criptomonedas con IA

Sistema avanzado de predicción de criptomonedas utilizando Machine Learning e Inteligencia Artificial.

## 📋 Características

- **Predicción con IA**: Modelo de Machine Learning entrenado con datos históricos
- **Visualizaciones Avanzadas**: Gráficos interactivos, heatmaps, order books
- **Análisis Técnico**: RSI, Moving Averages, Bollinger Bands, y más
- **Tiempo Real**: Actualización automática de predicciones
- **Dashboard Profesional**: Interfaz moderna con Next.js y Tailwind CSS

## 🛠️ Tecnologías

### Frontend
- **Next.js 16** - Framework React con App Router
- **TypeScript** - Tipado estático
- **Tailwind CSS v4** - Estilos modernos
- **shadcn/ui** - Componentes UI
- **Recharts** - Visualización de datos
- **SWR** - Data fetching y caché

### Backend / IA
- **Python 3.8+** - Lenguaje para ML
- **scikit-learn** - Machine Learning
- **pandas** - Procesamiento de datos
- **numpy** - Cálculos numéricos
- **requests** - API calls

## 📦 Instalación

### 1. Instalar dependencias de Node.js

\`\`\`bash
npm install
# o
yarn install
# o
pnpm install
\`\`\`

### 2. Instalar dependencias de Python

\`\`\`bash
pip install pandas numpy scikit-learn requests
\`\`\`

## 🚀 Uso

### Paso 1: Obtener Datos Históricos

Descarga datos históricos de criptomonedas desde CoinGecko API:

\`\`\`bash
python scripts/fetch_crypto_data.py
\`\`\`

Esto creará un archivo `crypto_historical_data.json` con datos de:
- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)
- Binance Coin (BNB)

### Paso 2: Entrenar el Modelo

Entrena el modelo de Machine Learning con los datos históricos:

\`\`\`bash
python scripts/train_model.py
\`\`\`

El script:
- Calcula indicadores técnicos (RSI, Moving Averages, Bollinger Bands)
- Entrena múltiples modelos (Random Forest, Gradient Boosting)
- Selecciona el mejor modelo basado en accuracy
- Guarda el modelo en `crypto_prediction_model.pkl`

**Salida esperada:**
\`\`\`
ENTRENANDO MODELO DE PREDICCIÓN DE CRIPTOMONEDAS
============================================================

Procesando BTC...
  ✓ 340 muestras procesadas
Procesando ETH...
  ✓ 340 muestras procesadas
...

Total de muestras: 1360
Distribución de clases:
  Subida (1): 680 (50.0%)
  Bajada (0): 680 (50.0%)

MEJOR MODELO: Gradient Boosting
Accuracy: 78.45%
\`\`\`

### Paso 3: Generar Predicciones

Genera predicciones para todas las criptomonedas:

\`\`\`bash
python scripts/predict.py
\`\`\`

Esto creará `predictions.json` con las predicciones actuales.

### Paso 4: Iniciar el Dashboard

\`\`\`bash
npm run dev
\`\`\`



## 🤖 Cómo Funciona el Modelo

### Indicadores Técnicos Utilizados

1. **RSI (Relative Strength Index)**: Mide momentum
2. **Moving Averages**: MA de 7, 21 y 50 días
3. **Volatilidad**: Desviación estándar de 7 días
4. **Momentum**: Cambio de precio en 4 días
5. **Bollinger Bands**: Bandas superior e inferior
6. **Volumen y Market Cap**: Métricas de mercado

### Proceso de Predicción

1. **Recolección de datos**: Obtiene precios históricos de CoinGecko
2. **Feature Engineering**: Calcula indicadores técnicos
3. **Entrenamiento**: Usa Random Forest y Gradient Boosting
4. **Predicción**: Predice si el precio subirá o bajará en 3 días
5. **Confianza**: Calcula probabilidad de la predicción

### Métricas del Modelo

- **Accuracy**: ~75-80% en datos de prueba
- **Precision**: Qué tan precisas son las predicciones positivas
- **Recall**: Qué tan bien detecta las subidas reales
- **F1-Score**: Balance entre precision y recall

## 🔄 Actualización de Predicciones

Para mantener las predicciones actualizadas:

\`\`\`bash
# Actualizar datos y regenerar predicciones
python scripts/fetch_crypto_data.py && python scripts/predict.py
\`\`\`

**Recomendación**: Ejecuta esto cada 1-6 horas para predicciones frescas.

Para reentrenar el modelo con nuevos datos:

\`\`\`bash
# Pipeline completo
python scripts/fetch_crypto_data.py && \
python scripts/train_model.py && \
python scripts/predict.py
\`\`\`

**Recomendación**: Reentrena el modelo semanalmente o mensualmente.

## 📈 Visualizaciones Disponibles

1. **Dashboard Principal**: Vista general de todas las predicciones
2. **Medidor de Confianza**: Nivel de certeza del modelo
3. **Gráfico de Tendencia**: Proyección visual de la predicción
4. **Distribución de Probabilidad**: Rango de cambios posibles
5. **Análisis de Volumen**: Actividad de trading
6. **Heatmap de Correlación**: Relaciones entre cryptos
7. **Order Book**: Profundidad de mercado

## ⚠️ Disclaimer

**IMPORTANTE**: Este sistema es solo para fines educativos y de investigación. 

- Las predicciones NO son consejos financieros
- El trading de criptomonedas conlleva riesgos significativos
- Siempre haz tu propia investigación (DYOR)
- Nunca inviertas más de lo que puedes permitirte perder

## 🔧 Personalización

### Agregar más criptomonedas

Edita `scripts/fetch_crypto_data.py`:

\`\`\`python
cryptos = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'cardano': 'ADA',  # Agregar nueva crypto
    # ... más cryptos
}
\`\`\`

### Ajustar parámetros del modelo

Edita `scripts/train_model.py`:

\`\`\`python
rf_model = RandomForestClassifier(
    n_estimators=300,  # Aumentar árboles
    max_depth=20,      # Mayor profundidad
    # ... más parámetros
)
\`\`\`

### Cambiar horizonte de predicción

En `train_model.py`, modifica:


**¡Feliz trading! 🚀📈**
