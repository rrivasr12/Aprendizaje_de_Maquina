import json
import os
import sys
import time
from pathlib import Path

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

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configurar salida utf-8 para Windows consola
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Configuración base
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_regression_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }


def measure_inference_latency(model, X_sample, y_scaler=None, n_iter=100):
    latencies = []
    # Warmup
    for _ in range(5):
        _ = model.predict(X_sample[:1])

    for i in range(min(n_iter, len(X_sample))):
        sample = X_sample[i : i + 1]
        t0 = time.perf_counter()
        pred = model.predict(sample)
        if y_scaler is not None:
            _ = y_scaler.inverse_transform(pred.reshape(-1, 1))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    return round(float(np.mean(latencies)), 3)


def main():
    print("=" * 70)
    print("=== PIPELINE DE ENTRENAMIENTO, AUDITORÍA Y BENCHMARKING RIGUROSO ===")
    print("=" * 70)

    # 1. Carga de datos
    data_file = BASE_DIR / "Clean_Dataset.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {data_file}")

    print(f"\n[1/7] Cargando dataset desde: {data_file.name}")
    df = pd.read_csv(data_file)
    print(f"  - Dimensiones iniciales: {df.shape[0]:,} filas, {df.shape[1]} columnas")

    # 2. Limpieza de columnas residuales e identificación
    drop_cols = [c for c in ["Unnamed: 0", "flight"] if c in df.columns]
    df_clean = df.drop(columns=drop_cols)
    print(f"  - Columnas descartadas: {drop_cols}")
    print(f"  - Dimensiones tras limpieza: {df_clean.shape[0]:,} filas, {df_clean.shape[1]} columnas")

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
    target_col = "price"

    X = df_clean[cat_cols + num_cols]
    y = df_clean[target_col].values

    # 3. Partición estratificada / determinista: 70% Train, 15% Val, 15% Test
    print("\n[2/7] Generando partición determinista: 70% Train, 15% Val, 15% Test...")
    # Primer split: 70% Train, 30% Temp (estratificado por clase de cabina para preservar proporciones)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=X["class"]
    )
    # Segundo split: Dividir el 30% restante en partes iguales (50% de 30% = 15% Val, 15% Test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=X_temp["class"]
    )

    print(f"  - Train Set : {X_train.shape[0]:,} muestras ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"  - Val Set   : {X_val.shape[0]:,} muestras ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"  - Test Set  : {X_test.shape[0]:,} muestras ({X_test.shape[0]/len(df)*100:.1f}%)")

    # 4. Ajuste del Preprocesador ÚNICAMENTE sobre Train (Prevención de Data Leakage)
    print("\n[3/7] Ajustando ColumnTransformer ÚNICAMENTE sobre el subconjunto de Train...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                cat_cols,
            ),
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())
    print(f"  - Vector de características generado: {X_train_proc.shape[1]} atributos")

    # Escalador para target en modelos neuronales
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Guardar preprocesador y nombres de características
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(y_scaler, MODELS_DIR / "y_scaler.joblib")
    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print("  - Preprocesador y escalador exportados a models/")

    # 5. Benchmarking de Modelos Tradicionales (Evaluación en VALIDACIÓN)
    print("\n[4/7] Ejecutando Benchmarking de Modelos Tradicionales en VALIDACIÓN...")
    ml_models = {
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=16, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100, max_depth=16, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=80, max_depth=5, random_state=RANDOM_STATE
        ),
    }

    ml_val_results = {}
    trained_ml_models = {}

    for name, model in ml_models.items():
        t0 = time.time()
        print(f"  -> Entrenando {name}...")
        model.fit(X_train_proc, y_train)
        fit_time = time.time() - t0

        preds_val = model.predict(X_val_proc)
        metrics = calculate_regression_metrics(y_val, preds_val)
        metrics["train_time_sec"] = round(fit_time, 2)
        ml_val_results[name] = metrics
        trained_ml_models[name] = model
        print(
            f"     Val R²: {metrics['r2']:.4f} | RMSE: INR {metrics['rmse']:,.2f} | MAE: INR {metrics['mae']:,.2f} ({fit_time:.1f}s)"
        )

    # Selección programática del mejor modelo tradicional
    best_ml_name = max(ml_val_results, key=lambda k: (ml_val_results[k]["r2"], -ml_val_results[k]["rmse"]))
    best_ml_model = trained_ml_models[best_ml_name]
    print(f"\n  🏆 Mejor Modelo Tradicional seleccionado programáticamente: '{best_ml_name}' (Val R²: {ml_val_results[best_ml_name]['r2']})")

    # 6. Benchmarking de 3 Arquitecturas MLP Deep Learning (Evaluación en VALIDACIÓN)
    print("\n[5/7] Ejecutando Benchmarking de 3 Arquitecturas MLP en VALIDACIÓN...")
    mlp_architectures = {
        "Arch 1 (Standard 128-64-32)": (128, 64, 32),
        "Arch 2 (Ancha 256-128-64)": (256, 128, 64),
        "Arch 3 (Profunda 128-128-64-32)": (128, 128, 64, 32),
    }

    mlp_val_results = {}
    trained_mlp_models = {}

    for name, layers in mlp_architectures.items():
        print(f"  -> Entrenando MLP {name}...")
        mlp = MLPRegressor(
            hidden_layer_sizes=layers,
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=15,
            tol=1e-4,
            random_state=RANDOM_STATE,
        )
        t0 = time.time()
        mlp.fit(X_train_proc, y_train_scaled)
        fit_time = time.time() - t0

        # Época real de convergencia
        stopped_epoch = int(mlp.n_iter_)

        # Inferencia en validación y des-escalado
        preds_scaled = mlp.predict(X_val_proc)
        preds_val = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

        metrics = calculate_regression_metrics(y_val, preds_val)
        metrics["stopped_epoch"] = stopped_epoch
        metrics["train_time_sec"] = round(fit_time, 2)
        metrics["hidden_layers"] = list(layers)
        mlp_val_results[name] = metrics
        trained_mlp_models[name] = mlp
        print(
            f"     Convergencia en época real: {stopped_epoch} | Val R²: {metrics['r2']:.4f} | RMSE: INR {metrics['rmse']:,.2f} ({fit_time:.1f}s)"
        )

    # Selección programática de la mejor arquitectura MLP
    best_mlp_arch_name = max(mlp_val_results, key=lambda k: (mlp_val_results[k]["r2"], -mlp_val_results[k]["rmse"]))
    best_layers = mlp_architectures[best_mlp_arch_name]
    print(f"\n  🏆 Mejor Arquitectura MLP seleccionada programáticamente: '{best_mlp_arch_name}'")

    # 7. Refinamiento Real de Hiperparámetros sobre la Arquitectura MLP Ganadora
    print(f"\n[6/7] Refinamiento de hiperparámetros sobre '{best_mlp_arch_name}'...")
    param_grid = [
        {"alpha": 0.0001, "learning_rate_init": 0.001},
        {"alpha": 0.001, "learning_rate_init": 0.001},
        {"alpha": 0.01, "learning_rate_init": 0.0005},
    ]

    best_tuned_mlp = None
    best_tuned_val_metrics = None
    best_params = None
    initial_params = {
        "architecture": best_mlp_arch_name,
        "layers": list(best_layers),
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    }

    tuning_history = []
    for params in param_grid:
        print(f"  -> Evaluando tuning con alpha={params['alpha']}, lr={params['learning_rate_init']}...")
        tuned_model = MLPRegressor(
            hidden_layer_sizes=best_layers,
            activation="relu",
            solver="adam",
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            batch_size=256,
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=15,
            tol=1e-4,
            random_state=RANDOM_STATE,
        )
        t0 = time.time()
        tuned_model.fit(X_train_proc, y_train_scaled)
        fit_time = time.time() - t0

        preds_scaled = tuned_model.predict(X_val_proc)
        preds_val = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        metrics = calculate_regression_metrics(y_val, preds_val)
        metrics["stopped_epoch"] = int(tuned_model.n_iter_)
        metrics["params"] = params
        metrics["train_time_sec"] = round(fit_time, 2)
        tuning_history.append(metrics)

        if best_tuned_val_metrics is None or metrics["r2"] > best_tuned_val_metrics["r2"]:
            best_tuned_val_metrics = metrics
            best_tuned_mlp = tuned_model
            best_params = params

    print(f"  🏆 Parámetros óptimos tras refinamiento: {best_params} (Val R²: {best_tuned_val_metrics['r2']})")

    # 8. Evaluación Final ÚNICA en el TEST SET
    print("\n[7/7] Evaluación Final en TEST SET (1 sola ejecución con datos nunca antes vistos)...")

    # Inferencia final Test - Mejor Modelo Tradicional
    test_preds_ml = best_ml_model.predict(X_test_proc)
    test_metrics_ml = calculate_regression_metrics(y_test, test_preds_ml)
    ml_latency_ms = measure_inference_latency(best_ml_model, X_test_proc)
    test_metrics_ml["latency_ms"] = ml_latency_ms

    # Inferencia final Test - Mejor Modelo MLP Refinado
    test_preds_mlp_scaled = best_tuned_mlp.predict(X_test_proc)
    test_preds_mlp = y_scaler.inverse_transform(test_preds_mlp_scaled.reshape(-1, 1)).ravel()
    test_metrics_mlp = calculate_regression_metrics(y_test, test_preds_mlp)
    mlp_latency_ms = measure_inference_latency(best_tuned_mlp, X_test_proc, y_scaler=y_scaler)
    test_metrics_mlp["latency_ms"] = mlp_latency_ms

    print("\n" + "=" * 50)
    print("MÉTRICAS FINALES EN TEST SET (DATOS NO VISTOS)")
    print("=" * 50)
    print(f"1. {best_ml_name} (Tradicional):")
    print(f"   - R² Score : {test_metrics_ml['r2']:.4f}")
    print(f"   - RMSE     : INR {test_metrics_ml['rmse']:,.2f}")
    print(f"   - MAE      : INR {test_metrics_ml['mae']:,.2f}")
    print(f"   - MSE      : {test_metrics_ml['mse']:,.2f}")
    print(f"   - Latencia : {test_metrics_ml['latency_ms']} ms/muestra")

    print(f"\n2. {best_mlp_arch_name} Refinada (Deep Learning):")
    print(f"   - R² Score : {test_metrics_mlp['r2']:.4f}")
    print(f"   - RMSE     : INR {test_metrics_mlp['rmse']:,.2f}")
    print(f"   - MAE      : INR {test_metrics_mlp['mae']:,.2f}")
    print(f"   - MSE      : {test_metrics_mlp['mse']:,.2f}")
    print(f"   - Latencia : {test_metrics_mlp['latency_ms']} ms/muestra")

    # Guardar los mejores modelos
    rf_model_path = MODELS_DIR / "best_rf_model.joblib"
    mlp_model_path = MODELS_DIR / "best_mlp_model.joblib"
    joblib.dump(best_ml_model, rf_model_path, compress=3)
    joblib.dump(best_tuned_mlp, mlp_model_path)

    rf_size_mb = round(rf_model_path.stat().st_size / (1024 * 1024), 2)
    mlp_size_mb = round(mlp_model_path.stat().st_size / (1024 * 1024), 2)
    test_metrics_ml["disk_size_mb"] = rf_size_mb
    test_metrics_mlp["disk_size_mb"] = mlp_size_mb

    # Compilar reporte final de auditoría
    audit_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "Clean_Dataset.csv",
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "features_count": int(X_train_proc.shape[1]),
            "random_state": RANDOM_STATE,
        },
        "traditional_models_validation_benchmark": ml_val_results,
        "mlp_architectures_validation_benchmark": mlp_val_results,
        "selected_best_models": {
            "best_traditional_model": best_ml_name,
            "best_mlp_architecture": best_mlp_arch_name,
        },
        "hyperparameter_refinement": {
            "initial_parameters": initial_params,
            "tuned_parameters": best_params,
            "tuning_experiments": tuning_history,
        },
        "final_test_evaluation": {
            best_ml_name: test_metrics_ml,
            "MLPRegressor_Tuned": test_metrics_mlp,
        },
    }

    metrics_path = MODELS_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print(f"\n✅ Auditoría completa guardada exitosamente en: {metrics_path}")
    print(f"✅ Modelos serializados en {MODELS_DIR} ({rf_model_path.name} [{rf_size_mb} MB], {mlp_model_path.name} [{mlp_size_mb} MB])")


if __name__ == "__main__":
    main()
