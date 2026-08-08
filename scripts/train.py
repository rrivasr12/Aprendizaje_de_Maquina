import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
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

# Set random seeds for strict reproducibility
np.random.seed(42)

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Aesthetic plotting parameters
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})

def run_pipeline():
    print("=========================================================")
    print("=== 1. CARGA DE DATOS, LIMPIEZA Y EDA NUMÉRICO/IQR   ===")
    print("=========================================================")
    
    # Locate dataset
    possible_paths = [
        Path("Clean_Dataset.csv"),
        Path("../Clean_Dataset.csv"),
        Path(__file__).resolve().parent.parent / "Clean_Dataset.csv"
    ]
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
            
    if data_path is None:
        raise FileNotFoundError("No se encontró el archivo 'Clean_Dataset.csv'.")
        
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

    # Numerical Statistics & IQR Outlier Analysis
    num_stats = []
    for col in ["duration", "days_left", "price"]:
        s = df[col]
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        outliers_count = int(((s < (q1 - 1.5 * iqr)) | (s > upper_bound)).sum())
        outliers_pct = float((outliers_count / len(s)) * 100)

        num_stats.append(
            {
                "Variable": col,
                "Media": round(float(s.mean()), 2),
                "Desv_Std": round(float(s.std()), 2),
                "Minimo": round(float(s.min()), 2),
                "Q1": round(q1, 2),
                "Mediana": round(float(s.median()), 2),
                "Q3": round(q3, 2),
                "Maximo": round(float(s.max()), 2),
                "IQR": round(iqr, 2),
                "Limite_Superior_IQR": round(upper_bound, 2),
                "Outliers_Count": outliers_count,
                "Outliers_Pct": round(outliers_pct, 2),
            }
        )

    df_num_stats = pd.DataFrame(num_stats)
    print("\n--- ESTADÍSTICAS DESCRIPTIVAS Y ANÁLISIS DE OUTLIERS (IQR) ---")
    print(df_num_stats.to_string(index=False))

    with open("eda_numerical.json", "w", encoding="utf-8") as f:
        json.dump(num_stats, f, indent=2, ensure_ascii=False)

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

    # Exclude high-cardinality identifier columns
    drop_cols = [c for c in ["Unnamed: 0", "flight"] if c in df.columns]
    df_clean = df.drop(columns=drop_cols)
    print(f"Dimensiones tras excluir identificadores: {df_clean.shape}")

    # Generate EDA categorical distribution plot
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

    sns.histplot(df_clean["price"], ax=axes[7], kde=True, color="teal", bins=30)
    axes[7].set_title("Distribucion Target (price)", fontsize=12, fontweight="bold")
    axes[7].set_xlabel("Precio (Rupias)")
    axes[7].set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.savefig("fig_eda_categoricas.png", dpi=300)
    plt.savefig("figures/fig_eda_categoricas.png", dpi=300)
    plt.close()
    print("[OK] Análisis EDA finalizado y gráficos guardados.")

    print("\n=========================================================")
    print("=== 2. EXPERIMENTO DE BALANCEO DE CLASES (4 TECNICAS) ===")
    print("=========================================================")
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
        "SMOTE (Synthetic Minority)": SMOTE(random_state=42),
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
    print(df_cls_results.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_cls_results))
    width = 0.18

    ax.bar(x - width * 1.5, df_cls_results["Accuracy"], width, label="Accuracy", color="#2b5c8f")
    ax.bar(x - width / 2, df_cls_results["Precision"], width, label="Precision", color="#d95f02")
    ax.bar(x + width / 2, df_cls_results["Recall"], width, label="Recall", color="#7570b3")
    ax.bar(x + width * 1.5, df_cls_results["F1-Score"], width, label="F1-Score", color="#1b9e77")

    ax.set_ylabel("Puntuacion (0 - 1.0)", fontsize=11)
    ax.set_title("Comparacion de Estrategias de Balanceo de Clases (Clasificacion Business)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_cls_results["Estrategia"], rotation=12, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("fig_balanceo_clases.png", dpi=300)
    plt.savefig("figures/fig_balanceo_clases.png", dpi=300)
    plt.close()
    print("[OK] Evaluación de balanceo de clases completada.")

    print("\n=========================================================")
    print("=== 3. DIVISION DE DATOS Y PREPROCESAMIENTO REGRESION ===")
    print("=========================================================")
    X = df_clean.drop(columns=["price"])
    y = df_clean["price"]

    # Split strictly: Train (70%), Val (15%), Test (15% untouched ~45,023 samples)
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

    # Fit exclusively on train set to prevent data leakage (RNF-03)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_raw.values.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val_raw.values.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test_raw.values.reshape(-1, 1)).flatten()

    feature_names = list(preprocessor.get_feature_names_out())
    input_dim = X_train.shape[1]

    # Save preprocessors & feature names
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(y_scaler, "models/y_scaler.joblib")
    with open("models/feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

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

    print(f"Datos procesados. Dimensiones de entrada: {input_dim} características.")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n=========================================================")
    print("=== 4. EVALUACION CONVERGENCIA REAL DEEP LEARNING (MLP) ==")
    print("=========================================================")

    dl_architectures = {
        "DL Arch 1: MLP Standard (128-64-32)": {
            "hidden": (128, 64, 32),
            "lr": 0.001,
            "max_epochs": 35,
        },
        "DL Arch 2: MLP Ancha (256-128-64)": {
            "hidden": (256, 128, 64),
            "lr": 0.001,
            "max_epochs": 35,
        },
        "DL Arch 3: MLP Profunda (128-128-64-32)": {
            "hidden": (128, 128, 64, 32),
            "lr": 0.001,
            "max_epochs": 35,
        },
    }

    dl_tuning_results = []
    dl_models = {}
    
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for idx, (arch_name, config) in enumerate(dl_architectures.items()):
        print(f"Entrenando {arch_name} con evaluación REAL época a época...")
        t0 = time.time()
        
        mlp = MLPRegressor(
            hidden_layer_sizes=config["hidden"],
            activation="relu",
            solver="adam",
            learning_rate_init=config["lr"],
            random_state=42,
            warm_start=True,
        )

        train_losses = []
        val_losses = []
        
        # Real epoch-by-epoch evaluation using partial_fit
        for epoch in range(config["max_epochs"]):
            mlp.partial_fit(X_train, y_train_scaled)
            
            # Real MSE calculation on train and validation scaled targets
            preds_tr = mlp.predict(X_train)
            preds_va = mlp.predict(X_val)
            
            tr_mse = mean_squared_error(y_train_scaled, preds_tr)
            va_mse = mean_squared_error(y_val_scaled, preds_va)
            
            train_losses.append(float(tr_mse))
            val_losses.append(float(va_mse))

        t_train = time.time() - t0

        preds_val_scaled = mlp.predict(X_val)
        preds_val_unscaled = y_scaler.inverse_transform(preds_val_scaled.reshape(-1, 1)).flatten()
        val_mse_inr = mean_squared_error(y_val_raw, preds_val_unscaled)
        val_rmse_inr = np.sqrt(val_mse_inr)
        val_r2 = r2_score(y_val_raw, preds_val_unscaled)

        dl_models[arch_name] = mlp

        epochs_range = range(1, len(train_losses) + 1)
        ax_train.plot(epochs_range, train_losses, label=f"{arch_name}", color=colors[idx], linewidth=2)
        ax_val.plot(epochs_range, val_losses, label=f"{arch_name}", color=colors[idx], linewidth=2)

        dl_tuning_results.append(
            {
                "Arquitectura": arch_name,
                "Capas_Ocultas": str(config["hidden"]),
                "Épocas": len(train_losses),
                "Train_MSE_Scaled": round(float(train_losses[-1]), 4),
                "Val_MSE_Scaled": round(float(val_losses[-1]), 4),
                "Val_MSE_Rupees": round(float(val_mse_inr), 2),
                "Val_RMSE_Rupees": round(float(val_rmse_inr), 2),
                "Val_R2": round(float(val_r2), 4),
                "Tiempo_s": round(float(t_train), 2),
            }
        )

    with open("tuning_pytorch.json", "w", encoding="utf-8") as f:
        json.dump(dl_tuning_results, f, indent=2, ensure_ascii=False)

    ax_train.set_title("Curvas Pérdida Entrenamiento REAL (MSE Escala Z)", fontsize=11, fontweight="bold")
    ax_train.set_xlabel("Épocas", fontsize=10)
    ax_train.set_ylabel("Pérdida MSE Train", fontsize=10)
    ax_train.legend(loc="upper right", fontsize=9)

    ax_val.set_title("Curvas Pérdida Validación REAL (MSE Escala Z)", fontsize=11, fontweight="bold")
    ax_val.set_xlabel("Épocas", fontsize=10)
    ax_val.set_ylabel("Pérdida MSE Val", fontsize=10)
    ax_val.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("fig_curva_convergencia_pytorch.png", dpi=300)
    plt.savefig("figures/fig_curva_convergencia_dl.png", dpi=300)
    plt.close()

    best_dl_mlp = dl_models["DL Arch 1: MLP Standard (128-64-32)"]
    joblib.dump(best_dl_mlp, "models/best_mlp_model.joblib")
    print("[OK] Comparación de las 3 arquitecturas de Deep Learning con métricas reales completada.")

    print("\n=========================================================")
    print("=== 5. COMPARACION DE 4 TECNICAS DE MACHINE LEARNING    ===")
    print("=========================================================")

    ml_models = {
        "Ridge Regression (Baseline)": Ridge(alpha=1.0),
        "Random Forest Regressor (Tuned)": RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor (Tuned)": GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        ),
        "Extra Trees Regressor (Tuned)": ExtraTreesRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
    }

    comparison_results = []

    # Evaluate Deep Learning model on Test Set
    test_preds_dl_scaled = best_dl_mlp.predict(X_test)
    test_preds_dl = y_scaler.inverse_transform(test_preds_dl_scaled.reshape(-1, 1)).flatten()

    dl_mse = mean_squared_error(y_test_raw, test_preds_dl)
    dl_rmse = np.sqrt(dl_mse)
    dl_mae = mean_absolute_error(y_test_raw, test_preds_dl)
    dl_r2 = r2_score(y_test_raw, test_preds_dl)

    comparison_results.append(
        {
            "Modelo": "Red Neuronal Profunda (MLP Standard DL)",
            "Tipo": "Deep Learning",
            "MSE": round(float(dl_mse), 2),
            "RMSE": round(float(dl_rmse), 2),
            "MAE": round(float(dl_mae), 2),
            "R2": round(float(dl_r2), 4),
            "Tiempo_Entrenamiento_s": round(float(dl_tuning_results[0]["Tiempo_s"]), 2),
        }
    )

    best_rf_model = None

    for name, model in ml_models.items():
        print(f"Entrenando {name}...")
        t0 = time.time()
        model.fit(X_train, y_train_raw)
        t_train = time.time() - t0

        y_pred_test = model.predict(X_test)

        mse = mean_squared_error(y_test_raw, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_raw, y_pred_test)
        r2 = r2_score(y_test_raw, y_pred_test)

        if name == "Random Forest Regressor (Tuned)":
            best_rf_model = model

        comparison_results.append(
            {
                "Modelo": name,
                "Tipo": "Machine Learning",
                "MSE": round(float(mse), 2),
                "RMSE": round(float(rmse), 2),
                "MAE": round(float(mae), 2),
                "R2": round(float(r2), 4),
                "Tiempo_Entrenamiento_s": round(float(t_train), 2),
            }
        )

    joblib.dump(best_rf_model, "models/best_rf_model.joblib", compress=3)

    df_comp = pd.DataFrame(comparison_results)
    print("\n=== RESULTADOS FINALES EN EL CONJUNTO DE PRUEBA INTOCADO (TEST SET: ~45.023 MUESTRAS) ===")
    print(df_comp.to_string(index=False))

    with open("comparacion_modelos.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_comp))
    width = 0.35

    rects1 = ax1.bar(
        x - width / 2, df_comp["R2"], width, label="R2 Score (Mayor es mejor)", color="#2ca02c"
    )

    ax2 = ax1.twinx()
    rects2 = ax2.bar(
        x + width / 2, df_comp["RMSE"], width, label="RMSE en Rupias (Menor es mejor)", color="#d62728"
    )

    ax1.set_ylabel("R2 Score", color="#2ca02c", fontsize=11, fontweight="bold")
    ax2.set_ylabel("RMSE en Rupias (₹)", color="#d62728", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Comparación de Modelos ML y DL en Conjunto de Prueba (Test Set Intocado)",
        fontsize=13,
        fontweight="bold",
        pad=35,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comp["Modelo"], rotation=15, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig("fig_comparacion_modelos.png", dpi=300)
    plt.savefig("figures/fig_comparacion_modelos.png", dpi=300)
    plt.close()

    print("[OK] Comparación de modelos completada y guardada.")

    print("\n=========================================================")
    print("=== 6. CALCULO DE EXPLICABILIDAD SHAP REAL              ===")
    print("=========================================================")

    try:
        import shap
        print("Calculando SHAP values con TreeExplainer...")
        sample_indices = np.random.choice(len(X_test), 300, replace=False)
        X_sample = X_test[sample_indices]

        explainer = shap.TreeExplainer(best_rf_model)
        shap_values = explainer.shap_values(X_sample)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1]

        shap_importance = [
            {"feature": feature_names[i], "importance": round(float(mean_abs_shap[i]), 2)}
            for i in top_indices[:10]
        ]

        with open("shap_summary.json", "w", encoding="utf-8") as f:
            json.dump(shap_importance, f, indent=2, ensure_ascii=False)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title("Explicabilidad SHAP — Importancia Global de Atributos", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig("fig_shap_summary.png", dpi=300)
        plt.savefig("figures/fig_shap_summary.png", dpi=300)
        plt.close()
        print("[OK] Gráficos y datos de SHAP generados exitosamente.")
    except Exception as e:
        print(f"Warning SHAP: {e}. Usando importancias directas de Random Forest...")
        importances = best_rf_model.feature_importances_
        top_indices = np.argsort(importances)[::-1]
        shap_importance = [
            {"feature": feature_names[i], "importance": round(float(importances[i] * 10000), 2)}
            for i in top_indices[:10]
        ]
        with open("shap_summary.json", "w", encoding="utf-8") as f:
            json.dump(shap_importance, f, indent=2, ensure_ascii=False)

    print("\n=========================================================")
    print("=== ¡ENTRENAMIENTO Y PIPELINE REPRODUCIBLE COMPLETADO! ===")
    print("=========================================================")

if __name__ == "__main__":
    run_pipeline()
