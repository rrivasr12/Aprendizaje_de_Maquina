# ✈️ AirPrice ML — Plataforma de Predicción de Tarifas Aéreas, Explicabilidad SHAP y Telemetría

> **Curso:** Aprendizaje de Máquinas (ACIF104) — Fase 4 Consolidada Final  
> **Proyecto:** Sistema End-to-End de Inferencia de Precios de Vuelos en India, Interpretabilidad con SHAP Real y Monitoreo Operacional  
> **Integrantes:** Manuel Miranda & Rodrigo Rivas  
> **Institución:** Universidad Andrés Bello (UNAB)  
> **Arquitectura:** Lead ML & Software Engineer Standard  

---

## 📋 Resumen Ejecutivo del Proyecto

Este repositorio contiene la arquitectura completa, modular y reproducible para la estimación precisa de tarifas aéreas sobre el conjunto de datos `Clean_Dataset.csv` (300.153 registros).
El sistema predice en tiempo real el precio continuo de pasajes aéreos en **Rupias Indias (INR)**, **Dólares Estadounidenses (USD)** y **Pesos Chilenos (CLP)**, ofreciendo:
- **Alternancia de Inferencia en Producción:** Soporte nativo para **Random Forest Regressor** y **Red Neuronal Profunda (MLPRegressor)** refinada.
- **Prevención Estricta de Data Leakage:** Partición estratificada determinista (70% Train, 15% Val, 15% Test) con ajuste de `ColumnTransformer` (StandardScaler + OneHotEncoder `drop='first'`) exclusivamente sobre el conjunto de entrenamiento.
- **Interpretabilidad Nativa:** Explicabilidad marginal mediante **SHAP (`TreeExplainer`)** sobre el modelo Random Forest entregando contribuciones monetarias exactas en cada predicción.
- **Telemetría y Resiliencia Operacional:** Middleware de auditoría con cabecera `X-Process-Time-MS` y endpoint de telemetría `/api/metrics` cumpliendo el requerimiento de latencia RNF02 (< 200 ms).
- **Suite de Pruebas Automatizadas:** Cobertura con `pytest backend/test_api.py` (100% aprobado).
- **Cuaderno Técnico Consolidado:** `notebooks/analisis_s4f4_fase4_consolidado.ipynb` con todas sus celdas ejecutadas y salidas persistentes.

---

## 📊 Resultados de Benchmarking y Evaluación en Test Set

### 1. Benchmarking de Modelos Tradicionales (Conjunto de Validación — 45.023 muestras)
| Modelo | R² Score | RMSE (INR) | MAE (INR) | Tiempo Entren. (s) |
|---|:---:|:---:|:---:|:---:|
| **Random Forest (Seleccionado)** 🏆 | **0.9786** | **₹ 3.318,92** | **₹ 1.670,05** | 26.2 s |
| Extra Trees | 0.9766 | ₹ 3.473,03 | ₹ 1.783,95 | 37.0 s |
| Gradient Boosting | 0.9606 | ₹ 4.500,99 | ₹ 2.647,20 | 45.2 s |
| Ridge Regression (Baseline) | 0.9095 | ₹ 6.825,05 | ₹ 4.613,32 | 0.1 s |

### 2. Benchmarking de Arquitecturas Deep Learning (MLPRegressor en Validación)
| Arquitectura | Capas Ocultas | Época de Parada | Val R² Score | Val RMSE (INR) |
|---|:---:|:---:|:---:|:---:|
| **Arch 2: MLP Ancha (Seleccionada)** 🏆 | **(256, 128, 64)** | **100** | **0.9810** | **₹ 3.124,59** |
| Arch 3: MLP Profunda | (128, 128, 64, 32) | 79 (Early Stop) | 0.9794 | ₹ 3.258,78 |
| Arch 1: MLP Standard | (128, 64, 32) | 100 | 0.9776 | ₹ 3.394,01 |

> **Refinamiento de Hiperparámetros (MLP Ancha):**  
> Parámetros óptimos tras búsqueda de retícula: `alpha=0.0001`, `learning_rate_init=0.001` con activación ReLU y optimizador Adam.

### 3. Evaluación Final en el Conjunto de Prueba Intocado (Test Set — 45.023 muestras nunca vistas)
| Métrica Evaluada | Random Forest Regressor | Red Neuronal Profunda (MLP Tuned) |
|---|:---:|:---:|
| **$R^2$ Score** | **0.9785** | **0.9809** |
| **RMSE (Error Cuadrático Medio)** | **₹ 3.327,96** | **₹ 3.140,59** |
| **MAE (Error Absoluto Medio)** | **₹ 1.667,92** | **₹ 1.653,03** |
| **MSE** | 11.075.331,37 | 9.863.307,71 |
| **Latencia Media de Inferencia** | 34.11 ms / muestra | **0.52 ms / muestra** |
| **Tamaño del Artefacto en Disco** | 151.26 MB | **1.13 MB** |

*El informe completo estructurado se almacena en [`models/training_metrics.json`](file:///models/training_metrics.json).*

---

## 🛠️ Matriz de Requisitos Cumplidos

| ID | Categoría | Descripción del Requisito | Evidencia / Archivo | Estado |
|---|---|---|---|:---:|
| **RF-01** | Funcional | Predicción continua del precio de pasajes en INR, USD y CLP. | `backend/main.py` (`/api/predict`). | **CUMPLIDO** |
| **RF-02** | Alternancia | Soporte de inferencia configurable entre Random Forest y MLP. | Parámetro `model_type` en `/api/predict`. | **CUMPLIDO** |
| **RF-03** | Preprocesamiento | Pipeline con `ColumnTransformer`: `StandardScaler` en continuas y `OneHotEncoder(drop='first')` en categóricas sin multicolinealidad. | `models/preprocessor.joblib`. | **CUMPLIDO** |
| **RNF-01** | Explicabilidad | SHAP real calculado con `TreeExplainer` con valores marginales en Rupias. | `backend/main.py` (`/api/explain`). | **CUMPLIDO** |
| **RNF-02** | Latencia & SLA | Tiempo de respuesta del servicio < 200 ms con cabecera `X-Process-Time-MS`. | Inferencia MLP ~0.52 ms, RF ~34 ms. | **CUMPLIDO** |
| **RNF-03** | Anti-Data Leakage | Partición 70/15/15 con ajuste exclusivo sobre Train Set. | `scripts/train_and_audit.py`. | **CUMPLIDO** |
| **RNF-04** | Robustez de Entrada | Validación Pydantic estricta con códigos HTTP 422 y 400 ante datos fuera de dominio. | `FlightPredictionInput` en `backend/main.py`. | **CUMPLIDO** |
| **RNF-05** | Testing Integral | Suite de pruebas unitarias e integración con 100% de aprobación. | `backend/test_api.py` (9 tests OK). | **CUMPLIDO** |

---

## 🚀 Guía de Reproducción y Despliegue Local Paso a Paso

### 1. Requisitos Previos
- **Python:** versión 3.11 recomendada.
- **Git:** para clonar y versionar.

### 2. Creación del Entorno e Instalación de Dependencias
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# En Linux / macOS:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Ejecución del Pipeline de Entrenamiento y Auditoría Rigurosa
Para ejecutar todo el flujo (limpieza, división 70/15/15, preprocesamiento sin leakage, benchmarking tradicional, benchmarking de 3 arquitecturas MLP, selección programática, tuning de hiperparámetros y evaluación única en Test Set):
```bash
python scripts/train_and_audit.py
```
*Artefactos generados en `models/`:*
- `preprocessor.joblib`: Preprocesador de entrada (30 atributos).
- `y_scaler.joblib`: Escalador de la variable continua objetivo para la red neuronal.
- `best_rf_model.joblib`: Modelo ganador tradicional (Random Forest).
- `best_mlp_model.joblib`: Modelo ganador de Deep Learning refinado (MLP 256-128-64).
- `training_metrics.json`: Reporte exhaustivo de auditoría con todas las métricas de validación y test.
- `feature_names.json`: Lista canónica de nombres de características.

### 4. Ejecución de la Suite de Pruebas Automatizadas
Para verificar el funcionamiento correcto de todos los endpoints, modelos y validaciones:
```bash
python -m pytest backend/test_api.py -v
```
*Resultado esperado:* 9 pruebas pasadas con 100% de éxito.

### 5. Lanzamiento del Sistema Completo (Backend API + Frontend Web)
Para iniciar de manera unificada la API FastAPI y la interfaz Web en `http://127.0.0.1:8000`:
```bash
python run_system.py
```
La aplicación abrirá automáticamente el navegador web para realizar predicciones interactivas y explorar las explicaciones SHAP.

---

## 📖 Cuadernos Jupyter del Proyecto (`notebooks/`)

| Cuaderno | Descripción | Estado |
|---|---|:---:|
| [`analisis_s4f4_fase4_consolidado.ipynb`](notebooks/analisis_s4f4_fase4_consolidado.ipynb) | **Cuaderno Consolidado Final:** Abarca todo el flujo end-to-end (EDA IQR, prevención de leakage, balanceo de clases, benchmark ML, benchmark 3 DL MLPs, selección programática, tuning, test set intocado y explicabilidad SHAP). | **Ejecutado & Persistido** |
| [`analisis_s4f4_fase4.ipynb`](notebooks/analisis_s4f4_fase4.ipynb) | Cuaderno de la entrega Fase 4. | Completado |
| [`analisis_s9s2_fase3.ipynb`](notebooks/analisis_s9s2_fase3.ipynb) | Cuaderno de análisis experimental Fase 3. | Completado |
| [`analisis_s3f3_fase2.ipynb`](notebooks/analisis_s3f3_fase2.ipynb) | Cuaderno de EDA y modelos preliminares Fase 2. | Completado |
| [`analisis_s2f2_fase1.ipynb`](notebooks/analisis_s2f2_fase1.ipynb) | Cuaderno exploratorio inicial Fase 1. | Completado |

---

## 🌐 Endpoints Principales de la API REST

1. **`GET /api/health`**: Estado del servicio y confirmación de carga de modelos.
2. **`POST /api/predict`**: Inferencia de tarifa aérea (`model_type="rf"` o `"mlp"`).
   - Recibe: `airline`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `class`, `duration`, `days_left`, `model_type`.
   - Entrega: precio predicho en INR, USD y CLP, nombre del modelo utilizado y latencia en ms.
3. **`POST /api/explain`**: Interpretabilidad con SHAP `TreeExplainer` sobre Random Forest.
   - Entrega: `base_price_inr`, `predicted_price_inr` y top 10 atributos de mayor impacto marginal.
4. **`GET /api/metrics`**: Métricas operacionales en vivo (total predicciones, latencia promedio, percentiles, cumplimiento RNF02).
