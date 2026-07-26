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
│   └── analisis_s4f4_fase3.ipynb   (Deep Learning, Tuning de Modelos y SHAP)
│
├── src/                           <-- 🐍 Código Fuente de Experimentos ML/DL
│   ├── run_experiments_s6.py       (Script principal: 4 ML, 3 DL, 4 balanceos, SHAP)
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
│   ├── preprocessor.joblib         (Transformer One-Hot & StandardScaler)
│   ├── y_scaler.joblib             (Escalador Z del Target Price)
│   └── feature_names.json          (Nombres de características procesadas)
│
├── Clean_Dataset.csv              (Dataset procesado de 300.153 vuelos comerciales)
├── run_system.py                  (Lanzador de 1 solo comando: Backend + Frontend)
├── requirements.txt               (Dependencias del proyecto)
├── README.md                      (Documentación oficial del repositorio en GitHub)
└── .gitignore                     (Filtro de archivos temporales)
```

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

Se diseñaron y compararon 3 arquitecturas neuronales de Perceptrón Multicapa (MLP) utilizando `scikit-learn` / `PyTorch`:
1. **DL Arch 1 (MLP Standard)**: Capas ocultas `(128, 64, 32)` con activación ReLU y regularización ($R^2 = 0,9754$).
2. **DL Arch 2 (MLP Ancha)**: Capas ocultas `(256, 128, 64)` con alta capacidad de representación.
3. **DL Arch 3 (MLP Profunda)**: Capas ocultas `(128, 128, 64, 32)` orientadas a extracción jerárquica.

---

## ⚖️ Experimento de Balanceo de Clases

En la tarea auxiliar de clasificación de cabina (Business vs Economy), se evaluó el impacto de 4 técnicas de balanceo (Sin Balanceo / Baseline, RandomOverSampler, RandomUnderSampler y SMOTE). El análisis empírico demostró que la clase Business no requiere balanceo forzado, ya que este desplaza el hiperplano lineal inflando falsos positivos.

---

## ⏱️ Calculadora Automática de Duración por Ruta

Atendiendo al principio de usabilidad, se eliminó la selección manual de duración de vuelo en la interfaz web. El sistema ahora **calcula automáticamente** la duración estimada en horas según:
- Ciudad de Origen (Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai).
- Ciudad de Destino (excluyendo automáticamente la ciudad de origen).
- Número de escalas (Directo, 1 Escala o 2+ Escalas).

---

## 🔍 Explicabilidad SHAP (SHapley Additive exPlanations)

El sistema integra valores SHAP mediante `TreeExplainer` para atribuir la contribución exacta ($+$ / $-$ en Rupias) de cada variable en las predicciones.
- **Factor #1**: Clase de Cabina (Business añade $+\text{₹ } 25.000$ en promedio).
- **Factor #2**: Días Restantes (compra a última hora incrementa significativamente el valor).
- **Factor #3**: Duración del Vuelo y Aerolínea operadora.

---

## 📡 Endpoints de la API REST (Backend FastAPI)

- `POST /api/predict`: Recibe el itinerario del vuelo y retorna el precio estimado en Rupias (₹), USD ($) y CLP ($), junto a la latencia de respuesta ($\approx 12.4\text{ ms}$).
- `POST /api/explain`: Retorna la descomposición explicativa SHAP de las características para la cotización actual.
- `GET /api/metrics`: Endpoint de telemetría y monitoreo de desempeño (latencia media, peticiones totales, estado de salud y cumplimiento del RNF-02 $< 200\text{ ms}$).
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

### 3. Re-entrenar Modelos y Generar Benchmarks (Opcional)
```bash
python src/run_experiments_s6.py
```

### 4. Ejecutar Pruebas Automatizadas del Backend
```bash
pytest backend/test_api.py
```

---

## 🔗 Enlace al Repositorio de GitHub
👉 **[https://github.com/rrivasr12/Aprendizaje_de_Maquina](https://github.com/rrivasr12/Aprendizaje_de_Maquina)**
