import json
import os
import sys
from pathlib import Path

# Configurar salida utf-8 para Windows consola
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add short path hook for Windows MAX_PATH if needed
if sys.platform == "win32":
    import ctypes
    def get_short_path(path):
        try:
            buf = ctypes.create_unicode_buffer(1024)
            res = ctypes.windll.kernel32.GetShortPathNameW(os.path.abspath(path), buf, 1024)
            if res > 0:
                return buf.value
        except Exception:
            pass
        return path
    sys.path[:] = [get_short_path(p) if p and os.path.exists(p) else p for p in sys.path]

import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Header
    cells.append(nbf.v4.new_markdown_cell("""# ✈️ Aprendizaje de Máquinas (ACIF104) — Informe Técnico Consolidado Final
**Predicción de Tarifas Aéreas, Prevención de Data Leakage, Benchmarking ML & Deep Learning (MLP), Selección Programática, Explicabilidad SHAP y Auditoría de Despliegue**

---
### 👥 Integrantes del Proyecto:
- **Manuel Miranda**
- **Rodrigo Rivas**
- **Curso:** ACIF104 — Aprendizaje de Máquinas | Universidad Andrés Bello (UNAB)
- **Repositorio:** [https://github.com/rrivasr12/Aprendizaje_de_Maquina](https://github.com/rrivasr12/Aprendizaje_de_Maquina)

---
### 🎯 Objetivos de este Cuaderno Consolidado:
1. **Limpieza & EDA Numérico:** Exclusión rigurosa del identificador `flight` y columnas residuales para prevenir explosión dimensional; cálculo de tabla descriptiva completa.
2. **Análisis Formal de Outliers (IQR):** Rango Intercuartílico para `price` y justificación de conservación de tarifas ejecutivas (Business).
3. **Preprocesamiento Estricto:** Partición 70% Train / 15% Val / 15% Test determinista (semilla 42), ajuste de `ColumnTransformer` (StandardScaler + OneHotEncoder drop='first') **exclusivamente sobre Train** (30 neuronas de entrada, prevención de Dummy Variable Trap).
4. **Experimento de Balanceo de Clases:** Evaluación de Baseline, ROS, RUS y SMOTE en clasificación de cabina.
5. **Benchmarking de Modelos Tradicionales:** Ridge, Random Forest, Extra Trees y Gradient Boosting en conjunto de **Validación**.
6. **Benchmarking de 3 Arquitecturas MLP Deep Learning:** Arch 1 (128-64-32), Arch 2 (256-128-64) y Arch 3 (128-128-64-32) con registro de época exacta de convergencia.
7. **Selección Programática & Refinamiento:** Selección automática del mejor modelo ML y mejor MLP en Validación, con ajuste de regularización `alpha` y tasa de aprendizaje.
8. **Evaluación Final en Test Set:** Evaluación única en 45.023 muestras no vistas (R², RMSE, MAE, MSE, latencia ms y tamaño en disco).
9. **Explicabilidad con SHAP:** Análisis global y local con `TreeExplainer`."""))

    # Cell 2: Imports
    cells.append(nbf.v4.new_code_cell("""import json
import os
import sys
import time
from pathlib import Path

# Configuración para compatibilidad de rutas en Windows
if sys.platform == "win32":
    import ctypes
    def get_short_path(path):
        try:
            buf = ctypes.create_unicode_buffer(1024)
            res = ctypes.windll.kernel32.GetShortPathNameW(os.path.abspath(path), buf, 1024)
            if res > 0:
                return buf.value
        except Exception:
            pass
        return path
    sys.path[:] = [get_short_path(p) if p and os.path.exists(p) else p for p in sys.path]

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

np.random.seed(42)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})
print("✅ Entorno de ML y librerías inicializados con éxito.")"""))

    # Cell 3: Markdown Section 1
    cells.append(nbf.v4.new_markdown_cell("""## 1. Carga de Datos, Limpieza y Tabla de Estadísticas Descriptivas
Cargamos el archivo `Clean_Dataset.csv` (300.153 registros) y descartamos la columna de alta cardinalidad `flight` junto a índices residuales (`Unnamed: 0`) para evitar la explosión del espacio de características."""))

    # Cell 4: Code Section 1
    cells.append(nbf.v4.new_code_cell("""data_path = Path("../Clean_Dataset.csv") if Path("../Clean_Dataset.csv").exists() else Path("Clean_Dataset.csv")
df = pd.read_csv(data_path)
print(f"Dimensiones iniciales: {df.shape[0]:,} filas, {df.shape[1]} columnas")

# Exclusión de identificadores
drop_cols = [c for c in ["Unnamed: 0", "flight"] if c in df.columns]
df_clean = df.drop(columns=drop_cols)
print(f"Dimensiones limpias: {df_clean.shape[0]:,} filas, {df_clean.shape[1]} columnas")

# Estadísticas descriptivas
num_stats = []
for col in ["duration", "days_left", "price"]:
    s = df_clean[col]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    num_stats.append({
        "Variable": col,
        "Media": round(s.mean(), 2),
        "Desv_Std": round(s.std(), 2),
        "Mínimo": round(s.min(), 2),
        "Q1 (25%)": round(q1, 2),
        "Mediana": round(s.median(), 2),
        "Q3 (75%)": round(q3, 2),
        "Máximo": round(s.max(), 2),
    })

df_num_stats = pd.DataFrame(num_stats)
display(df_num_stats)"""))

    # Cell 5: Markdown Section 2
    cells.append(nbf.v4.new_markdown_cell("""## 2. Análisis Formal de Valores Atípicos (Outliers - Método IQR)
Calculamos formalmente el Rango Intercuartílico ($IQR = Q3 - Q1$) para la variable objetivo `price` y analizamos el límite superior estadístico ($Q3 + 1,5 \\times IQR$)."""))

    # Cell 6: Code Section 2
    cells.append(nbf.v4.new_code_cell("""q1_p = df_clean["price"].quantile(0.25)
q3_p = df_clean["price"].quantile(0.75)
iqr_p = q3_p - q1_p
upper_bound = q3_p + 1.5 * iqr_p
outliers_p = df_clean[df_clean["price"] > upper_bound]

print(f"Q1: INR {q1_p:,.2f} | Q3: INR {q3_p:,.2f} | IQR: INR {iqr_p:,.2f}")
print(f"Límite Superior IQR: INR {upper_bound:,.2f}")
print(f"Outliers detectados: {len(outliers_p):,} ({len(outliers_p)/len(df_clean)*100:.2f}%)")

# Desglose por cabina
display(df_clean.groupby("class")["price"].describe().round(2))

# Visualización
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df_clean, x="class", y="price", palette="Set2", ax=ax[0])
ax[0].axhline(upper_bound, color="red", linestyle="--", label=f"Límite IQR ({upper_bound:,.0f})")
ax[0].set_title("Distribución de Tarifas por Clase y Límite IQR", fontweight="bold")
ax[0].legend()

sns.histplot(data=df_clean, x="price", hue="class", kde=True, bins=40, palette="Set2", ax=ax[1])
ax[1].set_title("Histograma Bimodal del Precio (Economy vs Business)", fontweight="bold")
plt.tight_layout()
plt.show()"""))

    # Cell 7: Markdown Section 3
    cells.append(nbf.v4.new_markdown_cell("""### 💡 Justificación Técnica de Conservación de Outliers:
Como se evidencia en la tabla y los gráficos, la distribución es **bimodal**. Los precios superiores al umbral estadístico de ₹ 99.128 corresponden a pasajes genuinos de clase ejecutiva (`Business`) reservados en fechas críticas. No constituyen errores de captura ni valores espurios; eliminarlos sesgaría negativamente al modelo al predecir tarifas ejecutivas.

---
## 3. Partición Determinista 70/15/15 y Preprocesamiento Estricto
Para garantizar la validez metodológica y evitar **Data Leakage**, el `ColumnTransformer` se ajusta (`fit`) **únicamente sobre el 70% de Train**.
- **Variables Continuas:** `StandardScaler` sobre `duration` y `days_left`.
- **Variables Categóricas:** `OneHotEncoder(drop='first')` sobre 7 atributos para prevenir la trampa de variables ficticias (*Dummy Variable Trap*).
- **Dimensión resultante:** Exactamente **30 neuronas de entrada**."""))

    # Cell 8: Code Section 3
    cells.append(nbf.v4.new_code_cell("""cat_cols = ["airline", "source_city", "departure_time", "stops", "arrival_time", "destination_city", "class"]
num_cols = ["duration", "days_left"]

X = df_clean[cat_cols + num_cols]
y = df_clean["price"].values

# Split determinista 70% Train, 15% Val, 15% Test
X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=X["class"]
)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp_raw, y_temp_raw, test_size=0.50, random_state=42, stratify=X_temp_raw["class"]
)

# Ajuste ÚNICAMENTE sobre Train
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
    ]
)

X_train = preprocessor.fit_transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)
X_test = preprocessor.transform(X_test_raw)

# Escalado de target para red neuronal
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
y_val_scaled = y_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

feature_names = list(preprocessor.get_feature_names_out())
print(f"Dimensiones de entrada: {X_train.shape[1]} neuronas/características.")
print(f"Muestras: Train={X_train.shape[0]:,} (70%) | Val={X_val.shape[0]:,} (15%) | Test={X_test.shape[0]:,} (15%)")"""))

    # Cell 9: Markdown Section 4
    cells.append(nbf.v4.new_markdown_cell("""## 4. Experimento de Balanceo de Clases (Clasificación de Cabina)
Evaluamos el impacto de 4 técnicas de balanceo (Baseline, RandomOverSampler, RandomUnderSampler y SMOTE) en la tarea de clasificación de cabina (`Business` vs. `Economy`)."""))

    # Cell 10: Code Section 4
    cells.append(nbf.v4.new_code_cell("""X_cls = df_clean.drop(columns=["class", "price"])
y_cls = (df_clean["class"] == "Business").astype(int).values
cat_cls = [c for c in cat_cols if c != "class"]

prep_cls = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cls),
    ]
)

X_tr_c, X_val_c, y_tr_c, y_val_c = train_test_split(X_cls, y_cls, test_size=0.20, random_state=42, stratify=y_cls)
X_tr_proc = prep_cls.fit_transform(X_tr_c)
X_val_proc = prep_cls.transform(X_val_c)

samplers = {
    "Sin Balanceo (Baseline)": None,
    "Sobremuestreo (ROS)": RandomOverSampler(random_state=42),
    "Submuestreo (RUS)": RandomUnderSampler(random_state=42),
    "SMOTE (Sintético)": SMOTE(random_state=42),
}

cls_res = []
for name, sampler in samplers.items():
    if sampler:
        X_res, y_res = sampler.fit_resample(X_tr_proc, y_tr_c)
    else:
        X_res, y_res = X_tr_proc, y_tr_c
    
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_res, y_res)
    pred_c = clf.predict(X_val_proc)
    cls_res.append({
        "Estrategia": name,
        "Accuracy": round(accuracy_score(y_val_c, pred_c), 4),
        "Precision": round(precision_score(y_val_c, pred_c), 4),
        "Recall": round(recall_score(y_val_c, pred_c), 4),
        "F1-Score": round(f1_score(y_val_c, pred_c), 4),
        "Muestras Train": len(y_res)
    })

display(pd.DataFrame(cls_res))"""))

    # Cell 11: Markdown Section 5
    cells.append(nbf.v4.new_markdown_cell("""## 5. Benchmarking de Modelos Tradicionales de Regresión (en Validación)
Evaluamos 4 familias de modelos en el conjunto de **VALIDACIÓN**:
1. **Ridge Regression:** Modelo lineal regularizado L2.
2. **Random Forest:** Ensamble bagging de árboles de decisión.
3. **Extra Trees:** Árboles extremadamente aleatorizados.
4. **Gradient Boosting:** Ensamble secuencial boosting."""))

    # Cell 12: Code Section 5
    cells.append(nbf.v4.new_code_cell("""metrics_file = Path("../models/training_metrics.json") if Path("../models/training_metrics.json").exists() else Path("models/training_metrics.json")

with open(metrics_file, "r", encoding="utf-8") as f:
    audit_data = json.load(f)

trad_bench = audit_data["traditional_models_validation_benchmark"]
df_trad = pd.DataFrame.from_dict(trad_bench, orient="index").reset_index()
df_trad.rename(columns={"index": "Modelo Tradicional"}, inplace=True)
display(df_trad)

# Visualización comparativa
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
sns.barplot(data=df_trad, x="Modelo Tradicional", y="r2", palette="crest", ax=ax[0])
ax[0].set_title("Comparativa de R² Score en Validación", fontweight="bold")
ax[0].set_ylim(0.85, 1.0)
for p in ax[0].patches:
    ax[0].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

sns.barplot(data=df_trad, x="Modelo Tradicional", y="rmse", palette="rocket", ax=ax[1])
ax[1].set_title("Comparativa de RMSE (INR) en Validación", fontweight="bold")
for p in ax[1].patches:
    ax[1].annotate(f"₹{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

plt.tight_layout()
plt.show()"""))

    # Cell 13: Markdown Section 6
    cells.append(nbf.v4.new_markdown_cell("""## 6. Benchmarking de 3 Arquitecturas Deep Learning (MLP)
Evaluamos 3 arquitecturas de Perceptrón Multicapa con `early_stopping=True`, `n_iter_no_change=15`, `tol=1e-4` y `max_iter=100`:
- **Arch 1 (Standard):** `(128, 64, 32)`
- **Arch 2 (Ancha):** `(256, 128, 64)`
- **Arch 3 (Profunda):** `(128, 128, 64, 32)`
Registramos la **época real exacta de parada** (`n_iter_`)."""))

    # Cell 14: Code Section 6
    cells.append(nbf.v4.new_code_cell("""mlp_bench = audit_data["mlp_architectures_validation_benchmark"]
df_mlp = pd.DataFrame.from_dict(mlp_bench, orient="index").reset_index()
df_mlp.rename(columns={"index": "Arquitectura MLP"}, inplace=True)
display(df_mlp)

# Visualización
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
sns.barplot(data=df_mlp, x="Arquitectura MLP", y="r2", palette="mako", ax=ax[0])
ax[0].set_title("R² Score en Validación por Arquitectura MLP", fontweight="bold")
ax[0].set_ylim(0.95, 1.0)
for p in ax[0].patches:
    ax[0].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

sns.barplot(data=df_mlp, x="Arquitectura MLP", y="stopped_epoch", palette="viridis", ax=ax[1])
ax[1].set_title("Época Real de Parada (Convergencia)", fontweight="bold")
for p in ax[1].patches:
    ax[1].annotate(f"{int(p.get_height())} épocas", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

plt.tight_layout()
plt.show()"""))

    # Cell 15: Markdown Section 7
    cells.append(nbf.v4.new_markdown_cell("""## 7. Selección Programática del Ganador y Refinamiento de Hiperparámetros
La selección del modelo ganador se realiza de forma **100% programática** comparando $R^2$ y $RMSE$ en el conjunto de Validación.
Posteriormente, se ejecuta un refinamiento real de hiperparámetros sobre la arquitectura ganadora variando `alpha` (L2) y `learning_rate_init`."""))

    # Cell 16: Code Section 7
    cells.append(nbf.v4.new_code_cell("""best_trad = audit_data["selected_best_models"]["best_traditional_model"]
best_mlp = audit_data["selected_best_models"]["best_mlp_architecture"]

print(f"🏆 Modelo Tradicional Ganador (Selección Automática): {best_trad}")
print(f"🏆 Arquitectura MLP Ganadora (Selección Automática): {best_mlp}")

tuning_data = audit_data["hyperparameter_refinement"]
print(f"\\nParámetros iniciales: {tuning_data['initial_parameters']}")
print(f"Parámetros óptimos tras tuning: {tuning_data['tuned_parameters']}")

df_tuning = pd.DataFrame(tuning_data["tuning_experiments"])
display(df_tuning[["params", "r2", "rmse", "mae", "stopped_epoch", "train_time_sec"]])"""))

    # Cell 17: Markdown Section 8
    cells.append(nbf.v4.new_markdown_cell("""## 8. Evaluación Final ÚNICA en el Test Set (Datos No Vistos)
Evaluamos los dos modelos seleccionados (**Random Forest** y **MLP Ancha Tuned**) una única vez sobre las **45.023 muestras** intocadas del Test Set."""))

    # Cell 18: Code Section 8
    cells.append(nbf.v4.new_code_cell("""test_eval = audit_data["final_test_evaluation"]
df_test = pd.DataFrame.from_dict(test_eval, orient="index").reset_index()
df_test.rename(columns={"index": "Modelo Ganador"}, inplace=True)
display(df_test)

# Gráfico de predicciones vs valores reales sobre muestra de test
models_dir = Path("../models") if Path("../models").exists() else Path("models")
rf_model = joblib.load(models_dir / "best_rf_model.joblib")
mlp_model = joblib.load(models_dir / "best_mlp_model.joblib")

sample_idx = np.random.choice(len(X_test), 500, replace=False)
X_test_sample = X_test[sample_idx]
y_test_sample = y_test_raw[sample_idx]

pred_rf_s = rf_model.predict(X_test_sample)
pred_mlp_s = y_scaler.inverse_transform(mlp_model.predict(X_test_sample).reshape(-1, 1)).ravel()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(y_test_sample, pred_rf_s, alpha=0.5, color="royalblue")
ax[0].plot([0, 120000], [0, 120000], "r--", linewidth=2)
ax[0].set_title("Random Forest: Real vs Predicho (Test)", fontweight="bold")
ax[0].set_xlabel("Tarifa Real (INR)")
ax[0].set_ylabel("Tarifa Predicha (INR)")

ax[1].scatter(y_test_sample, pred_mlp_s, alpha=0.5, color="teal")
ax[1].plot([0, 120000], [0, 120000], "r--", linewidth=2)
ax[1].set_title("MLP Ancha Tuned: Real vs Predicho (Test)", fontweight="bold")
ax[1].set_xlabel("Tarifa Real (INR)")
ax[1].set_ylabel("Tarifa Predicha (INR)")

plt.tight_layout()
plt.show()"""))

    # Cell 19: Markdown Section 9
    cells.append(nbf.v4.new_markdown_cell("""## 9. Explicabilidad Global y Local mediante SHAP
Implementamos `shap.TreeExplainer` sobre el modelo Random Forest para interpretar la contribución marginal de cada variable en el precio del billete de avión."""))

    # Cell 20: Code Section 9
    cells.append(nbf.v4.new_code_cell("""# SHAP TreeExplainer sobre submuestra de test
explainer = shap.TreeExplainer(rf_model)
shap_sample = X_test[:100]
shap_values = explainer.shap_values(shap_sample)

clean_features = [f.replace("cat__", "").replace("num__", "") for f in feature_names]

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, shap_sample, feature_names=clean_features, show=False, max_display=10)
plt.title("SHAP Global Feature Importance (Top 10 Atributos)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# Escenario representativo local
sample_pred = rf_model.predict(shap_sample[:1])[0]
base_val = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
print(f"Escenario de Vuelo Seleccionado:")
print(f"  - Tarifa Base Promedio (Expected Value): INR {base_val:,.2f}")
print(f"  - Tarifa Predicha por Random Forest   : INR {sample_pred:,.2f}")

sample_contribs = sorted(zip(clean_features, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)[:5]
print("\\nTop 5 Contribuciones Marginales:")
for feat, contrib in sample_contribs:
    direction = "INCREMENTA el precio" if contrib > 0 else "REDUCE el precio"
    print(f"  * {feat:25s}: {direction:22s} (INR {contrib:+,.2f})")"""))

    # Cell 21: Markdown Conclusion
    cells.append(nbf.v4.new_markdown_cell("""## 10. Conclusiones y Recomendaciones de Despliegue
1. **Rendimiento Predictivo:**
   - La red neuronal profunda refinada (**MLP Ancha 256-128-64**) obtuvo el mejor desempeño general con un $R^2 = 0.9809$ y un $RMSE = ₹ 3.140,59$, con una latencia de inferencia ultrarrápida de **0.52 ms/muestra** y apenas **1.13 MB** de peso en disco.
   - El modelo **Random Forest Regressor** alcanzó un $R^2 = 0.9785$ y $RMSE = ₹ 3.327,96$, con la ventaja adicional de permitir explicabilidad analítica nativa mediante `shap.TreeExplainer`.
2. **Prevención de Data Leakage:**
   - El escalado y la codificación One-Hot con `drop='first'` ajustados exclusivamente sobre el conjunto de entrenamiento de 210.107 muestras eliminaron cualquier fuga de información hacia los conjuntos de validación y test.
3. **Servicio en Producción:**
   - El backend FastAPI (`backend/main.py`) permite alternar sin interrupciones entre ambos modelos (`model_type="rf"` y `model_type="mlp"`), satisfaciendo el Requisito No Funcional RNF02 (< 200 ms por petición) y entregando explicabilidad dinámica en tiempo real."""))

    nb.cells = cells

    nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "analisis_s4f4_fase4_consolidado.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Notebook consolidado generado exitosamente en: {nb_path}")

if __name__ == "__main__":
    create_notebook()
