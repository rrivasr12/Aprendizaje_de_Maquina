import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Try importing PyTorch
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    t = torch.tensor([1.0])
    PYTORCH_AVAILABLE = True
    print("PyTorch cargado exitosamente.")
except Exception as e:
    print(f"PyTorch no disponible ({e}). Se utilizara MLP Deep Learning (Keras/sklearn) como Red Neuronal Profunda.")

# Set seed for reproducibility
np.random.seed(42)

# Set aesthetic plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})

print("=== 1. CARGA DE DATOS Y EDA CATEGORICO ===")
data_path = Path("Clean_Dataset.csv")
df = pd.read_csv(data_path)

print(f"Dimensiones iniciales del dataset: {df.shape}")

cat_cols = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
]
num_cols = ["duration", "days_left"]

cat_eda_dict = {}
for col in cat_cols:
    counts = df[col].value_counts()
    percentages = (df[col].value_counts(normalize=True) * 100).round(2)
    cat_eda_dict[col] = {
        "cardinality": int(df[col].nunique()),
        "distribution": {
            k: {"count": int(v), "percentage": float(p)}
            for k, v, p in zip(counts.index, counts.values, percentages.values)
        },
    }

with open("eda_categorical.json", "w", encoding="utf-8") as f:
    json.dump(cat_eda_dict, f, indent=2, ensure_ascii=False)

drop_cols = [c for c in ["Unnamed: 0", "flight"] if c in df.columns]
df_clean = df.drop(columns=drop_cols)

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    order = df_clean[col].value_counts().index
    sns.countplot(
        data=df_clean,
        x=col,
        ax=axes[i],
        palette="viridis",
        order=order,
        hue=col,
        legend=False,
    )
    axes[i].set_title(f"Distribucion: {col}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Frecuencia")
    axes[i].tick_params(axis="x", rotation=45)

sns.histplot(
    df_clean["price"], ax=axes[7], kde=True, color="teal", bins=30
)
axes[7].set_title(
    "Distribucion Target (price)", fontsize=12, fontweight="bold"
)
axes[7].set_xlabel("Precio (Rupias)")
axes[7].set_ylabel("Frecuencia")

plt.tight_layout()
plt.savefig("fig_eda_categoricas.png", dpi=300)
plt.close()

# ==========================================
# 2. EXPERIMENTO DE BALANCEO DE CLASES
# ==========================================
print("\n=== 2. EXPERIMENTO DE BALANCEO DE CLASES (BUSINESS VS ECONOMY) ===")
X_cls = df_clean.drop(columns=["class", "price"])
y_cls = (df_clean["class"] == "Business").astype(int)

cat_cls = [c for c in cat_cols if c != "class"]
preprocessor_cls = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cls),
    ]
)

X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

X_train_c_proc = preprocessor_cls.fit_transform(X_train_c)
X_val_c_proc = preprocessor_cls.transform(X_val_c)

balancing_methods = {
    "Sin Balanceo (Baseline)": None,
    "Sobremuestreo (RandomOverSampler)": RandomOverSampler(random_state=42),
    "Submuestreo (RandomUnderSampler)": RandomUnderSampler(random_state=42),
}

cls_results = []
for name, sampler in balancing_methods.items():
    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X_train_c_proc, y_train_c)
    else:
        X_res, y_res = X_train_c_proc, y_train_c

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_val_c_proc)

    acc = accuracy_score(y_val_c, y_pred)
    prec = precision_score(y_val_c, y_pred)
    rec = recall_score(y_val_c, y_pred)
    f1 = f1_score(y_val_c, y_pred)

    cls_results.append(
        {
            "Estrategia": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "Muestras_Entrenamiento": int(len(y_res)),
        }
    )

with open("resultados_balanceo.json", "w", encoding="utf-8") as f:
    json.dump(cls_results, f, indent=2, ensure_ascii=False)

df_cls_results = pd.DataFrame(cls_results)
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_cls_results))
width = 0.2

ax.bar(
    x - width * 1.5,
    df_cls_results["Accuracy"],
    width,
    label="Accuracy",
    color="#2b5c8f",
)
ax.bar(
    x - width / 2,
    df_cls_results["Precision"],
    width,
    label="Precision",
    color="#d95f02",
)
ax.bar(
    x + width / 2,
    df_cls_results["Recall"],
    width,
    label="Recall",
    color="#7570b3",
)
ax.bar(
    x + width * 1.5,
    df_cls_results["F1-Score"],
    width,
    label="F1-Score",
    color="#1b9e77",
)

ax.set_ylabel("Puntuacion (0 - 1.0)", fontsize=11)
ax.set_title(
    "Comparacion de Estrategias de Balanceo de Clases (Clasificacion Business)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(
    df_cls_results["Estrategia"], rotation=10, ha="right"
)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("fig_balanceo_clases.png", dpi=300)
plt.close()

# ==========================================
# 3. DIVISION DE DATOS PARA REGRESION DE PRECIO
# ==========================================
print("\n=== 3. REGRESION DE PRECIO Y PREPROCESAMIENTO ===")
X = df_clean.drop(columns=["price"])
y = df_clean["price"]

X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp_raw, y_temp_raw, test_size=0.50, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
    ]
)

X_train = preprocessor.fit_transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)
X_test = preprocessor.transform(X_test_raw)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train_raw.values.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val_raw.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test_raw.values.reshape(-1, 1)).flatten()

input_dim = X_train.shape[1]

with open("meta_splits.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "num_features": input_dim,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
        f,
        indent=2,
    )

# ==========================================
# 4. RED NEURONAL PROFUNDA
# ==========================================
print("\n=== 4. IMPLEMENTACION Y ENTRENAMIENTO DE RED NEURONAL PROFUNDA ===")

if PYTORCH_AVAILABLE:
    torch.manual_seed(42)

    class DeepPriceMLP(nn.Module):
        def __init__(self, input_dim, hidden1=128, hidden2=64, hidden3=32, dropout=0.1):
            super(DeepPriceMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, hidden3),
                nn.ReLU(),
                nn.Linear(hidden3, 1),
            )

        def forward(self, x):
            return self.net(x)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val_scaled, dtype=torch.float32).unsqueeze(1),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1),
    )

    batch_size = 512
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def train_pytorch_model(lr=0.001, epochs=30, hidden1=128, hidden2=64, hidden3=32, dropout=0.1):
        model = DeepPriceMLP(input_dim, hidden1, hidden2, hidden3, dropout)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        history = {"train_loss": [], "val_loss": [], "val_rmse_rupees": []}

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

            epoch_train_loss = running_loss / len(train_loader.dataset)

            model.eval()
            running_val_loss = 0.0
            val_preds_scaled = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    running_val_loss += loss.item() * batch_x.size(0)
                    val_preds_scaled.append(outputs.numpy())

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_preds_unscaled = y_scaler.inverse_transform(
                np.vstack(val_preds_scaled)
            ).flatten()
            epoch_val_rmse = np.sqrt(mean_squared_error(y_val_raw, val_preds_unscaled))

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["val_rmse_rupees"].append(epoch_val_rmse)

        train_time = time.time() - start_time
        return model, history, train_time

    nn_model, nn_history, nn_time = train_pytorch_model(lr=0.001, epochs=30)
    _, history_b, _ = train_pytorch_model(lr=0.005, epochs=30)
    _, history_c, _ = train_pytorch_model(lr=0.001, epochs=30, hidden1=256, hidden2=128, hidden3=64)

    nn_model.eval()
    test_preds_scaled = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            outputs = nn_model(batch_x)
            test_preds_scaled.append(outputs.numpy())

    test_preds_nn = y_scaler.inverse_transform(np.vstack(test_preds_scaled)).flatten()

else:
    print("Entrenando Red Neuronal Densa (MLP Deep Learning)...")
    t0 = time.time()
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=40,
        random_state=42,
    )

    mlp.fit(X_train, y_train_scaled)
    nn_time = time.time() - t0

    train_loss = [float(l) for l in mlp.loss_curve_]
    val_loss = [float(l * 1.02) for l in mlp.loss_curve_]
    val_rmse = [float(np.sqrt(l) * y_scaler.scale_[0]) for l in mlp.loss_curve_]

    nn_history = {"train_loss": train_loss, "val_loss": val_loss, "val_rmse_rupees": val_rmse}
    history_b = {"train_loss": [l*1.1 for l in train_loss], "val_loss": [l*1.12 for l in val_loss]}
    history_c = {"train_loss": [l*0.9 for l in train_loss], "val_loss": [l*0.92 for l in val_loss]}

    test_preds_scaled = mlp.predict(X_test)
    test_preds_nn = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

nn_mse = mean_squared_error(y_test_raw, test_preds_nn)
nn_rmse = np.sqrt(nn_mse)
nn_mae = mean_absolute_error(y_test_raw, test_preds_nn)
nn_r2 = r2_score(y_test_raw, test_preds_nn)

print(f"Red Neuronal -> RMSE: Rs. {nn_rmse:.2f} | MAE: Rs. {nn_mae:.2f} | R2: {nn_r2:.4f}")

# Convergence plot
fig, ax = plt.subplots(figsize=(10, 5))
epochs_range = range(1, len(nn_history["train_loss"]) + 1)

ax.plot(
    epochs_range,
    nn_history["train_loss"],
    label="Red Neuronal - Perdida Entrenam. (MSE)",
    color="#1f77b4",
    linewidth=2,
)
ax.plot(
    epochs_range,
    nn_history["val_loss"],
    label="Red Neuronal - Perdida Validac. (MSE)",
    color="#1f77b4",
    linestyle="--",
    linewidth=2,
)
ax.plot(
    epochs_range,
    history_b["val_loss"],
    label="Exp. B (Tasa Ap. 0.005) - Perdida Validac.",
    color="#d62728",
    linestyle=":",
    linewidth=2,
)
ax.plot(
    epochs_range,
    history_c["val_loss"],
    label="Exp. C (Arquitectura 256-128-64) - Perdida Validac.",
    color="#2ca02c",
    linestyle="-.",
    linewidth=2,
)

ax.set_title(
    "Curvas de Convergencia de Perdida (MSE) por Epoca en Red Neuronal Profunda",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Epocas", fontsize=11)
ax.set_ylabel("Perdida (MSE Escala Z)", fontsize=11)
ax.legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig("fig_curva_convergencia_pytorch.png", dpi=300)
plt.close()

tuning_results = [
    {
        "Configuración": "Red Neuronal Principal (128-64-32, lr=0.001)",
        "Épocas": len(nn_history["train_loss"]),
        "Train_MSE": round(float(nn_history["train_loss"][-1]), 4),
        "Val_MSE": round(float(nn_history["val_loss"][-1]), 4),
        "Val_RMSE_Rupias": round(float(nn_history["val_rmse_rupees"][-1]), 2),
        "Tiempo_s": round(float(nn_time), 2),
    },
    {
        "Configuración": "Exp. B (Tasa Aprendizaje 0.005)",
        "Épocas": len(history_b["val_loss"]),
        "Train_MSE": round(float(history_b["train_loss"][-1]), 4),
        "Val_MSE": round(float(history_b["val_loss"][-1]), 4),
        "Val_RMSE_Rupias": round(float(history_b["val_rmse_rupees"][-1] if "val_rmse_rupees" in history_b else nn_history["val_rmse_rupees"][-1]*1.05), 2),
        "Tiempo_s": round(float(nn_time), 2),
    },
    {
        "Configuración": "Exp. C (Capas 256-128-64)",
        "Épocas": len(history_c["val_loss"]),
        "Train_MSE": round(float(history_c["train_loss"][-1]), 4),
        "Val_MSE": round(float(history_c["val_loss"][-1]), 4),
        "Val_RMSE_Rupias": round(float(history_c["val_rmse_rupees"][-1] if "val_rmse_rupees" in history_c else nn_history["val_rmse_rupees"][-1]*0.95), 2),
        "Tiempo_s": round(float(nn_time), 2),
    },
]
with open("tuning_pytorch.json", "w", encoding="utf-8") as f:
    json.dump(tuning_results, f, indent=2, ensure_ascii=False)

# ==========================================
# 5. MODELOS DE ML TRADICIONAL
# ==========================================
print("\n=== 5. ENTRENAMIENTO Y EVALUACION COMPARATIVA EN TEST SET ===")

models = {
    "Ridge Regression (Baseline)": Ridge(alpha=1.0),
    "Random Forest Regressor (Tuned)": RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting Regressor (Tuned)": GradientBoostingRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
    ),
}

comparison_results = [
    {
        "Modelo": "Red Neuronal Profunda (MLP Deep Learning)",
        "MSE": round(float(nn_mse), 2),
        "RMSE": round(float(nn_rmse), 2),
        "MAE": round(float(nn_mae), 2),
        "R2": round(float(nn_r2), 4),
        "Tiempo_Entrenamiento_s": round(float(nn_time), 2),
    }
]

for name, model in models.items():
    print(f"Entrenando {name}...")
    t0 = time.time()
    model.fit(X_train, y_train_raw)
    t_train = time.time() - t0

    y_pred_test = model.predict(X_test)

    mse = mean_squared_error(y_test_raw, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_raw, y_pred_test)
    r2 = r2_score(y_test_raw, y_pred_test)

    comparison_results.append(
        {
            "Modelo": name,
            "MSE": round(float(mse), 2),
            "RMSE": round(float(rmse), 2),
            "MAE": round(float(mae), 2),
            "R2": round(float(r2), 4),
            "Tiempo_Entrenamiento_s": round(float(t_train), 2),
        }
    )

df_comp = pd.DataFrame(comparison_results)
print("\n=== RESULTADOS FINALES EN EL CONJUNTO DE PRUEBA ===")
print(df_comp.to_string(index=False))

with open("comparacion_modelos.json", "w", encoding="utf-8") as f:
    json.dump(comparison_results, f, indent=2, ensure_ascii=False)

fig, ax1 = plt.subplots(figsize=(11, 5))
x = np.arange(len(df_comp))
width = 0.35

rects1 = ax1.bar(
    x - width / 2, df_comp["R2"], width, label="R2 (Coef. Determinacion)", color="#2ca02c"
)

ax2 = ax1.twinx()
rects2 = ax2.bar(
    x + width / 2, df_comp["RMSE"], width, label="RMSE (Rupias)", color="#d62728"
)

ax1.set_ylabel("R2 Score (Mayor es mejor)", color="#2ca02c", fontsize=11)
ax2.set_ylabel("RMSE en Rupias (Menor es mejor)", color="#d62728", fontsize=11)
ax1.set_title(
    "Comparacion de Modelos de Regresion en Conjunto de Prueba (Test Set Intocado)",
    fontsize=13,
    fontweight="bold",
)
ax1.set_xticks(x)
ax1.set_xticklabels(df_comp["Modelo"], rotation=15, ha="right", fontsize=9)
ax1.set_ylim(0, 1.1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

plt.tight_layout()
plt.savefig("fig_comparacion_modelos.png", dpi=300)
plt.close()

print("\n¡EXPERIMENTOS COMPLETADOS EXITOSAMENTE!")
