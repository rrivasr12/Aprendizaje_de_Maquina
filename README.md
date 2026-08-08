# AirPrice ML — Plataforma de Predicción de Tarifas Aéreas, Explicabilidad SHAP y Telemetría

> **Curso:** Aprendizaje de Máquinas (ACIF104) — Fase 3  
> **Proyecto:** Sistema End-to-End de Inferencia de Precios de Vuelos en India, Interpretabilidad con SHAP Real y Monitoreo Operacional  
> **Arquitectura:** Lead DevOps & Software Architect Standard  

---

## 📋 Resumen del Proyecto

Este repositorio contiene la arquitectura completa y el código refactorizado para el sistema de predicción de tarifas aéreas. El sistema predice en tiempo real el precio continuo de pasajes aéreos en Rupias Indias (INR), Dólares Estadounidenses (USD) y Pesos Chilenos (CLP), entregando explicabilidad basada en **SHAP (SHapley Additive exPlanations) reales calculadas vía `TreeExplainer`** y exponiendo telemetría en tiempo real sobre la latencia y estado de salud de la API.

---

## 🛠️ Matriz de Requisitos Cumplidos (100% Rúbrica Fase 3)

| ID | Categoría | Descripción del Requisito | Implementación / Evidencia | Estado |
|---|---|---|---|:---:|
| **RF-01** | Funcional | Predicción continua del precio de pasajes aéreos. | Modelo Regresor Optimizado (`best_rf_model.joblib`). | **CUMPLIDO** |
| **RF-02** | Preprocesamiento | Pipeline con `ColumnTransformer`: `StandardScaler` (Z-score) para variables numéricas (`duration`, `days_left`); `OneHotEncoder(drop='first')` para variables categóricas. | `scripts/train.py` y `backend/main.py`. | **CUMPLIDO** |
| **RNF-01** | Usabilidad | Calculadora automática de duración estimada del vuelo basada en la combinación origen-destino-escalas. | Frontend UI (`frontend/app.js` & `index.html`). | **CUMPLIDO** |
| **RNF-02** | Explicabilidad | Descomposición de variables SHAP en tiempo real servida en JSON por la API y renderizada gráficamente en Frontend. | `TreeExplainer` en `/api/explain`. | **CUMPLIDO** |
| **RNF-03** | Blindaje Data Leakage | Ajuste exclusivo de transformadores sobre el conjunto de entrenamiento (Train 70%). | `ColumnTransformer.fit_transform(X_train)`. | **CUMPLIDO** |
| **RNF-04** | Confiabilidad | Evaluación en conjunto de prueba intocado (15% ~45.023 muestras) garantizando $R^2 \ge 0,90$. | **Random Forest $R^2 = 0,9769$**, RMSE = ₹ 3.436. | **CUMPLIDO** |
| **RNF-05** | Seguridad & Tipo | Validación estricta de esquema Pydantic en endpoints (rangos, tipos, enums de aerolíneas, ciudades y cabina). | `backend/main.py` (`FlightPredictionInput`). | **CUMPLIDO** |
| **RNF-06** | Escalabilidad & Latencia | Middleware en FastAPI inyectando cabecera `X-Process-Time-MS` y endpoint `/api/metrics` con latencia promedio < 200ms. | Latencia media real $\approx 12,4\text{ ms}$. | **CUMPLIDO** |

---

## 🚀 Guía de Instalación y Despliegue Paso a Paso

### 1. Requisitos Previos del Sistema

- **Python:** versión 3.9, 3.10 o 3.11.
- **Git:** para clonar o gestionar el repositorio.

### 2. Creación del Entorno Virtual

Crear y activar un entorno virtual limpio en la raíz del proyecto:

```bash
# En Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# En Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalación de Dependencias (`requirements.txt` Riguroso)

Instalar las dependencias estrictamente necesarias sin paquetes no relacionados:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecución del Entrenamiento Reproducible

Para ejecutar todo el pipeline de análisis exploratorio (EDA), experimentación de balanceo de clases, división estricta de datos (70/15/15), evaluación de curvas de convergencia reales de Deep Learning (MLP), benchmark de modelos y exportación de artefactos optimizados:

```bash
python scripts/train.py
```

*Salidas generadas:*
- Artefactos guardados en `models/`: `preprocessor.joblib`, `y_scaler.joblib`, `best_rf_model.joblib`, `best_mlp_model.joblib`, `feature_names.json`.
- Gráficos exportados en `figures/`: `fig_eda_categoricas.png`, `fig_balanceo_clases.png`, `fig_curva_convergencia_dl.png`, `fig_comparacion_modelos.png`, `fig_shap_summary.png`.
- Reportes JSON: `eda_numerical.json`, `eda_categorical.json`, `resultados_balanceo.json`, `tuning_pytorch.json`, `comparacion_modelos.json`, `shap_summary.json`.

### 5. Despliegue del Sistema Completo (Backend API + Frontend Web)

Para levantar de forma orquestada la API REST en Uvicorn y el servidor de archivos estáticos en el puerto 8000:

```bash
python scripts/run_system.py
```

El script abrirá automáticamente la interfaz web en su navegador predeterminado: `http://127.0.0.1:8000`.

---

## 🌐 Especificación de Endpoints de la API REST

### 1. `GET /api/health`
Verifica la disponibilidad y el estado de carga de los artefactos del modelo.

- **Respuesta:**
```json
{
  "status": "healthy",
  "timestamp": "2026-08-07 22:45:00",
  "models_loaded": true
}
```

### 2. `POST /api/predict`
Calcula la predicción continua de tarifa aérea para el itinerario ingresado.

- **Cuerpo de la Petición (Payload JSON):**
```json
{
  "airline": "Vistara",
  "source_city": "Delhi",
  "departure_time": "Morning",
  "stops": "one",
  "arrival_time": "Night",
  "destination_city": "Mumbai",
  "class": "Economy",
  "duration": 6.5,
  "days_left": 15
}
```

- **Respuesta JSON:**
```json
{
  "predicted_price_inr": 5953.0,
  "formatted_price": "₹ 5.953,00",
  "predicted_price_usd": 71.44,
  "predicted_price_clp": 68460.0,
  "model_used": "Random Forest Regressor (Tuned)",
  "latency_ms": 12.4,
  "status": "success"
}
```

### 3. `POST /api/explain` (SHAP Real vía `TreeExplainer`)
Devuelve la descomposición dinámica de valores SHAP calculados directamente sobre la muestra preprocesada.

- **Respuesta JSON:**
```json
{
  "base_price_inr": 20889.0,
  "predicted_price_inr": 5953.0,
  "contributions": [
    {
      "feature": "class_Economy",
      "contribution": -14250.5,
      "direction": "decreases_price"
    },
    {
      "feature": "days_left",
      "contribution": 1850.2,
      "direction": "increases_price"
    },
    {
      "feature": "duration",
      "contribution": 420.3,
      "direction": "increases_price"
    }
  ]
}
```

### 4. `GET /api/metrics` (Telemetría & Monitoreo)
Expone las métricas operacionales de rendimiento del servidor y el modelo.

- **Respuesta JSON:**
```json
{
  "total_predictions": 12,
  "total_explanations": 8,
  "avg_latency_ms": 12.4,
  "min_latency_ms": 8.1,
  "max_latency_ms": 24.5,
  "target_latency_rnf02_met": true,
  "model_metadata": {
    "primary_model": "Random Forest Regressor (Tuned)",
    "r2_score": 0.9769,
    "rmse_inr": 3436.21,
    "mae_inr": 1784.55
  },
  "system_health": "Optimal",
  "uptime_seconds": 345.2
}
```

---

## 📁 Estructura del Repositorio

```text
Aprendizaje_de_Maquina-master/
├── backend/
│   ├── main.py              # API REST FastAPI con /api/predict, /api/explain (SHAP) y /api/metrics
│   └── test_api.py          # Pruebas unitarias automatizadas para la API
├── frontend/
│   ├── index.html           # Interfaz web responsiva con pestañas de Cotizador, SHAP y Telemetría
│   ├── style.css            # Sistema de diseño con glassmorphism, gradientes y micro-animaciones
│   └── app.js               # Lógica del cliente, calculador de duración y renderización SHAP
├── models/
│   ├── best_rf_model.joblib # Modelo Random Forest Tuned (R² = 0.9769)
│   ├── best_mlp_model.joblib# Modelo Red Neuronal MLP Tuned
│   ├── preprocessor.joblib  # ColumnTransformer ajustado en conjunto Train
│   ├── y_scaler.joblib      # StandardScaler para el target en modelos Deep Learning
│   └── feature_names.json   # Lista de características transformadas OneHotEncoder/StandardScaler
├── notebooks/
│   ├── analisis_s2f2_fase1.ipynb
│   ├── analisis_s3f3_fase2.ipynb
│   ├── analisis_s4f4_fase3.ipynb
│   └── analisis_s4f4_fase4.ipynb
├── scripts/
│   ├── train.py             # Script ejecutable de entrenamiento reproducible end-to-end
│   └── run_system.py        # Script orquestador del sistema (FastAPI + Frontend)
├── Clean_Dataset.csv        # Dataset original de 300.153 registros de vuelos
├── requirements.txt         # Dependencias exactas y optimizadas del proyecto
├── run_system.py            # Launcher de nivel raíz
└── README.md                # Documentación oficial del proyecto
```

---

## 🏆 Resultados del Benchmark de Modelos en Test Set (15% Intocado — 45.023 Muestras)

| Modelo / Arquitectura | Tipo | RMSE (₹) | MAE (₹) | R² Score | Estado RNF-04 |
|---|---|---|---|:---:|:---:|
| **Random Forest Regressor (Tuned)** ⭐ | Machine Learning | **₹ 3.436,21** | **₹ 1.784,55** | **0,9769** | **Excelente** |
| Extra Trees Regressor (Tuned) | Machine Learning | ₹ 3.544,58 | ₹ 1.863,95 | 0,9754 | Excelente |
| Red Neuronal Profunda (MLP Standard) | Deep Learning | ₹ 3.545,16 | ₹ 1.956,02 | 0,9754 | Excelente |
| Gradient Boosting Regressor (Tuned) | Machine Learning | ₹ 4.062,29 | ₹ 2.361,06 | 0,9677 | Excelente |
| Ridge Regression (Baseline) | ML Lineal | ₹ 6.666,65 | ₹ 4.528,72 | 0,9129 | Cumplido |

---

## 👨‍💻 Créditos y Autores

Desarrollado para la asignatura **Aprendizaje de Máquinas (ACIF104)** — Universidad Andrés Bello (UNAB).
