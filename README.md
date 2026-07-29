# ✈️ AirPrice ML Intelligence System (ACIF104 — Fase 4 Final)

Sistema integral de **Machine Learning** y **Deep Learning** para la predicción dinámica de tarifas aéreas, explicabilidad transparente mediante valores **SHAP**, cálculo automático de duración de ruta y telemetría de monitoreo de desempeño en tiempo real.

---

## 👥 Integrantes del Grupo 1
- **Manuel Miranda**
- **Rodrigo Rivas**
- **Curso**: Aprendizaje de Máquinas (ACIF104) — Universidad Andrés Bello (UNAB)
- **Repositorio Oficial**: [https://github.com/rrivasr12/Aprendizaje_de_Maquina](https://github.com/rrivasr12/Aprendizaje_de_Maquina)

---

## 📁 Estructura del Repositorio

```
Aprendizaje_de_Maquina-master/
│
├── notebooks/                     <-- 📓 Análisis de Datos y Modelos (Jupyter Notebooks)
│   ├── analisis_s2f2_fase1.ipynb   (EDA Categórico e Histograma Target)
│   ├── analisis_s3f3_fase2.ipynb   (Modelos Baseline y Balanceo de Clases)
│   ├── analisis_s4f4_fase3.ipynb   (Deep Learning, Tuning de Modelos y SHAP)
│   └── analisis_s4f4_fase4.ipynb   (Análisis Fase 4 Final: EDA numérico, IQR, 4 ML vs 3 DL y SHAP)
│
├── src/                           <-- 🐍 Código Fuente de Experimentos ML/DL
│   ├── run_experiments_s6.py       (Script principal: 4 ML, 3 DL, 4 balanceos, IQR y SHAP)
│   └── create_notebook.py          (Utilidad generadora de notebooks)
│
├── backend/                       <-- ⚡ Proyecto Backend (API REST en FastAPI)
│   ├── main.py                     (Endpoints /api/predict, /api/explain, /api/metrics)
│   └── test_api.py                 (Pruebas unitarias automatizadas del backend)
│
├── frontend/                      <-- 💻 Proyecto Frontend (Aplicación Web Glassmorphic)
│   ├── index.html                  (Interfaz Web interactiva con calculadora automática)
│   ├── style.css                   (Diseño del sistema y tokens CSS)
│   └── app.js                      (Lógica JS, calculadora de ruta y gráficos SHAP)
│
├── models/                        <-- 💾 Modelos Guardados para Inferencia en Vivo
│   ├── best_rf_model.joblib        (Modelo principal Random Forest Regressor - Tuned)
│   ├── best_mlp_model.joblib       (Modelo secundario Deep Learning MLP Standard)
│   ├── preprocessor.joblib         (Transformer One-Hot drop='first' & StandardScaler)
│   ├── y_scaler.joblib             (Escalador Z del Target Price)
│   └── feature_names.json          (Nombres de las 30 características procesadas)
│
├── Clean_Dataset.csv              (Dataset procesado de 300.153 vuelos comerciales)
├── run_system.py                  (Lanzador de 1 solo comando: Backend + Frontend)
├── requirements.txt               (Dependencias del proyecto)
├── README.md                      (Documentación oficial del repositorio en GitHub)
└── .gitignore                     (Filtro de archivos temporales)
```

---

## 📈 Análisis Exploratorio de Datos Numérico y Outliers (IQR)

### 📊 Tabla de Estadísticas Descriptivas
| Variable | Media | Desv. Std | Mínimo | Q1 (25%) | Mediana | Q3 (75%) | Máximo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **duration** (horas) | 12,22 h | 7,19 h | 0,83 h | 6,83 h | 11,25 h | 16,17 h | 49,83 h |
| **days_left** (días) | 26,00 días | 13,56 días | 1,00 día | 15,00 días | 26,00 días | 38,00 días | 49,00 días |
| **price** (Rupias ₹) | ₹ 20.889,66 | ₹ 22.697,77 | ₹ 1.105,00 | ₹ 4.783,00 | ₹ 7.425,00 | ₹ 42.521,00 | ₹ 123.071,00 |

### 🔍 Análisis Formal de Outliers (Método IQR)
- **Rango Intercuartílico**: $\text{IQR} = Q3 - Q1 = 42.521 - 4.783 = 37.738\text{ Rupias (₹)}$.
- **Límite Superior Estadístico**: $Q3 + 1,5 \times \text{IQR} = 42.521 + 1,5 \times 37.738 = 99.128\text{ Rupias (₹)}$.
- **Muestras Atípicas Superiores**: 123 muestras ($0,04\%$ del dataset).
- **Justificación Técnica**: Al desagregar por cabina, la media de **Economy** es ₹ 6.572 y la media de **Business** asciende a ₹ 52.540 (máximo ₹ 123.071). Los valores superiores a ₹ 99.128 representan tarifas reales ejecutivas de alta demanda reservadas cerca del despegue; conservarlos es indispensable para permitir al modelo estimar tarifas de primera categoría.

---

## 🛠️ Matriz de Requisitos del Sistema (RF y RNF Secuenciales)

| ID Requisito | Dimensión Rúbrica | Descripción Técnica del Requisito | Criterio de Verificación |
| :---: | :---: | :--- | :--- |
| **RF-01** | Funcional | Predicción continua y dinámica de tarifas aéreas basada en atributos de itinerario. | Modelo RF/MLP genera precio predicho en Rupias (₹), USD y CLP. |
| **RF-02** | Funcional | Preprocesamiento automatizado y codificación de características categóricas sin data leakage. | `ColumnTransformer` ejecuta estandarización Z-score y One-Hot Encoding. |
| **RNF-01** | Usabilidad | Calculadora automática de duración estimada de vuelo según origen, destino y escalas. | Interfaz Web dinámica (JS) elimina entrada manual de duración. |
| **RNF-02** | Explicabilidad | Atribución e interpretación transparente de variables en tiempo real mediante valores SHAP. | Endpoint `/api/explain` retorna descomposiciones SHAP ($+$ / $-$ en Rupias). |
| **RNF-03** | Escalabilidad | Arquitectura desacoplada Backend REST en FastAPI / Frontend Web UI con inferencia asíncrona. | Frontend consume la API FastAPI mediante peticiones HTTP JSON. |
| **RNF-04** | Confiabilidad | Precisión predictiva $R^2 \ge 0,90$ y ciego absoluto contra data leakage entre particiones. | $R^2 = 0,9769$ en Test Set intocado (70/15/15). |
| **RNF-05** | Seguridad | Validación y sanitización estricta de parámetros de entrada mediante esquemas Pydantic. | Middleware FastAPI valida tipos y rangos, bloqueando inyecciones. |
| **RNF-06** | Monitoreabilidad | Telemetría de desempeño y latencia en tiempo real expuesta en el endpoint `/api/metrics` ($< 200\text{ ms}$). | Middleware `X-Process-Time` reporta latencia media de 12,4 ms. |

---

## 🧠 Vector de Entrada (30 Neuronas) y Prevención de la Dummy Variable Trap

- **Dummy Variable Trap**: Para evitar multicolinealidad perfecta en matrices de One-Hot Encoding, se configuró `OneHotEncoder(drop='first')`, eliminando la primera columna binaria de cada categoría como nivel de referencia.
- **Vector de Entrada Procesado (30 Neuronas)**:
  - 2 variables continuas escaladas con Z-score (`duration`, `days_left`).
  - 28 variables binarias derivadas del One-Hot Encoder $[(6-1)+(6-1)+(6-1)+(3-1)+(6-1)+(6-1)+(2-1) = 28]$.
- **Capa de Salida**: 1 neurona lineal sin activación acotada para la proyección del precio escalar continuo.

---

## 📊 Benchmarks de Desempeño (Evaluación en Test Set Intocado — 45.023 Muestras)

| Modelo / Arquitectura | Tipo de Técnica | MSE ($₹^2$) | RMSE ($₹$) | MAE ($₹$) | $R^2$ Score | Tiempo Entren. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor (Tuned)** ⭐ | Machine Learning | 11.807.573,49 | **₹ 3.436,21** | **₹ 1.784,55** | **0,9769** | 34,6 s |
| **Extra Trees Regressor (Tuned)** | Machine Learning | 12.564.052,14 | ₹ 3.544,58 | ₹ 1.863,95 | 0,9754 | 47,8 s |
| **Red Neuronal Profunda (MLP Standard)** | Deep Learning | 12.568.174,89 | ₹ 3.545,16 | ₹ 1.956,02 | **0,9754** | 111,3 s |
| **Gradient Boosting Regressor (Tuned)** | Machine Learning | 16.502.214,86 | ₹ 4.062,29 | ₹ 2.361,06 | 0,9677 | 93,7 s |
| **Ridge Regression (Baseline)** | ML Lineal | 44.444.162,25 | ₹ 6.666,65 | ₹ 4.528,72 | 0,9129 | 0,1 s |

---

## 🧠 Arquitecturas de Deep Learning Evaluadas

1. **DL Arch 1 (MLP Standard - Seleccionada)**: Capas ocultas `(128, 64, 32)` con ReLU, Dropout (`p=0,1`) y Batch Normalization ($R^2 = 0,9754$).
2. **DL Arch 2 (MLP Ancha)**: Capas ocultas `(256, 128, 64)` para alta capacidad de representación.
3. **DL Arch 3 (MLP Profunda)**: Capas ocultas `(128, 128, 64, 32)` para abstracciones jerárquicas multinivel.

---

## 🔍 Explicabilidad SHAP y Escenario Práctico

- **Escenario**: Cotización Vistara, Delhi ➔ Mumbai, Clase Business, 1 Escala, Horario Mañana, 15 días previos.
- **Precio Base Promedio**: ₹ 20.889,00.
- **Atribuciones SHAP**: `class_Business` ($+ \text{₹ } 25.000$), `days_left=15` ($+ \text{₹ } 2.700$), `airline_Vistara` ($+ \text{₹ } 1.200$), `stops_one` ($+ \text{₹ } 850$).
- **Precio Predicho Final**: $\approx \text{₹ } 50.639,00$.

---

## 📡 Endpoints de la API REST (Backend FastAPI)

- `POST /api/predict`: Inferencia continua en tiempo real (retorna precio en ₹, USD, CLP y latencia).
- `POST /api/explain`: Descomposición explicativa SHAP de las características.
- `GET /api/metrics`: Telemetría operacional (latencia media 12,4 ms, peticiones totales, RNF-06 status y salud).
- `GET /api/health`: Healthcheck básico del estado del servidor.

---

## 🚀 Instalación y Ejecución

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 2. Iniciar el Sistema Completo (Backend + Frontend Web)
```bash
python run_system.py
```
Este comando levanta el servidor FastAPI en `http://127.0.0.1:8000` y abre automáticamente la aplicación Web en tu navegador.

### 3. Ejecutar Pruebas Automatizadas del Backend
```bash
powershell -Command "$env:PYTHONPATH='.'; python -m pytest backend/test_api.py"
```

---

## 🔗 Enlace al Repositorio de GitHub
👉 **[https://github.com/rrivasr12/Aprendizaje_de_Maquina](https://github.com/rrivasr12/Aprendizaje_de_Maquina)**
