# Predicción de Tarifas Aéreas mediante Aprendizaje Automático y Deep Learning

**Asignatura:** Aprendizaje de Máquinas (ACIF104)  
**Institución:** Universidad Andrés Bello (UNAB) — Facultad de Ingeniería  
**Integrantes:** Manuel Miranda | Rodrigo Rivas  
**Curso / Sección:** ACIF104.202615.2404.EL.ON  

---

## 📌 Descripción del Proyecto

Este repositorio contiene el desarrollo del proyecto de **Aprendizaje de Máquinas**, enfocado en la estimación precisa de precios de pasajes aéreos en el mercado de aviación de la India utilizando un dataset extenso de más de **300.000 registros** (`Clean_Dataset.csv`).

El proyecto abarca desde el Análisis Exploratorio de Datos (EDA) y preprocesamiento avanzado (blindado contra la fuga de datos), hasta la implementación y evaluación comparativa de algoritmos de **Machine Learning tradicional** (Ridge Regression, Random Forest, Gradient Boosting) y arquitecturas de **Deep Learning** (Red Neuronal Profunda / Perceptrón Multicapa MLP).

---

## 📊 Dataset y Preprocesamiento

- **Dataset Principal:** `Clean_Dataset.csv` (300.153 registros y 11 características predictivas).
- **Variable Objetivo ($y$):** `price` (Tarifa aérea continua expresada en Rupias).
- **Variables Continuas:** `duration` (duración del vuelo en horas) y `days_left` (días de anticipación para la compra).
- **Variables Categóricas:** `airline`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `class`.
- **Exclusión de Variables:** Se desestimó la variable `flight` (código de vuelo) debido a su cardinalidad extrema (1.561 valores únicos), evitando sobreajuste y raleza en la matriz de entrada.
- **Partición de Datos:** 70 % Entrenamiento (210.107 muestras), 15 % Validación (45.023 muestras) y 15 % Prueba intocado (45.023 muestras).

---

## 🚀 Estructura del Repositorio

```text
Aprendizaje_de_Maquina-master/
├── Clean_Dataset.csv               # Dataset principal procesado (~24.6 MB)
├── README.md                       # Documentación general del proyecto
├── requirements.txt                # Lista de dependencias y librerías de Python
│
├── analisis_s2f2_fase1.ipynb       # Fase 1: Análisis crítico, RNF/RF e imputación inicial
├── analisis_s3f3_fase2.ipynb       # Fase 2: EDA descriptivo, cuantiles y modelos iniciales
├── analisis_s4f4_fase3.ipynb       # Fase 3: Red Neuronal Profunda (PyTorch/MLP) y Tuning
├── run_experiments_s4.py           # Script de ejecución automatizada de experimentos
│
├── fig_eda_categoricas.png         # Gráfico de distribuciones categóricas y del target
├── fig_balanceo_clases.png         # Comparación visual de estrategias de balanceo
├── fig_curva_convergencia_pytorch.png # Curvas de pérdida por época (Train vs Val Loss)
├── fig_comparacion_modelos.png     # Comparación final de R² y RMSE en el Test Set
│
├── eda_categorical.json            # Metadatos numéricos de frecuencias categóricas
├── resultados_balanceo.json        # Resultados numéricos de estrategias de balanceo
├── tuning_pytorch.json             # Experimentos de afinamiento de hiperparámetros
└── comparacion_modelos.json        # Resultados numéricos finales en el conjunto de prueba
```

---

## 🏆 Resultados Comparativos de Modelos (Conjunto de Prueba - Test Set)

Todos los modelos fueron evaluados sobre las 45.023 muestras del conjunto de prueba independiente (*Test Set*):

| Modelo de Aprendizaje | MSE (Rupias²) | RMSE (Rupias) | MAE (Rupias) | Coeficiente $R^2$ | Tiempo Train |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor (Tuned)** | 11.807.573,49 | **₹ 3.436,21** | **₹ 1.784,55** | **0,9769** | 27,8 s |
| **Red Neuronal Profunda (MLP Deep Learning)** | 12.333.579,86 | **₹ 3.511,92** | **₹ 1.990,40** | **0,9758** | 113,4 s |
| **Gradient Boosting Regressor (Tuned)** | 16.502.214,86 | **₹ 4.062,29** | **₹ 2.361,06** | **0,9677** | 73,0 s |
| **Ridge Regression (Baseline)** | 44.444.162,25 | **₹ 6.666,65** | **₹ 4.528,72** | **0,9129** | 0,1 s |

### Conclusiones de Selección:
1. **Random Forest Tuned ($R^2 = 0,9769$):** Seleccionado como modelo principal de producción por su alta precisión en captura de interacciones no lineales entre aerolíneas y trayectos.
2. **Red Neuronal Profunda MLP ($R^2 = 0,9758$):** Excelente modelo secundario que ofrece alta capacidad de generalización y soporte para inferencia continua en microservicios.

---

## 🛠️ Instalación y Ejecución

### 1. Clonar el repositorio y preparar el entorno
```bash
git clone <URL_DEL_REPOSITORIO>
cd Aprendizaje_de_Maquina-master
```

### 2. Crear y activar entorno virtual (opcional)
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate
```

### 3. Instalar dependencias requeridas
```bash
pip install -r requirements.txt
pip install imbalanced-learn torch reportlab python-docx
```

### 4. Ejecutar el flujo de experimentos automatizado
```bash
python run_experiments_s4.py
```

### 5. Abrir las libretas de Jupyter
```bash
jupyter lab
```
