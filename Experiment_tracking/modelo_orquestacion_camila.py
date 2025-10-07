import os
import argparse
import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from prefect import task, flow, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact


#from pathlib import Path



logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("orquestacion")

DEFAULT_MLRUNS = "file:./mlruns"

# MLflow configuration with fallback
def setup_mlflow():
    """Setup MLflow with proper error handling and fallback options."""
#    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    

    try:
        #mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_tracking_uri("http://186.121.46.71:5000")
        # Test connection
        mlflow.search_experiments()
        #logger.info(f"Connected to MLflow at: {mlflow_uri}")
    except Exception as e:
        #logger.warning(f"Failed to connect to {mlflow_uri}: {e}")
        logger.info("Falling back to local SQLite database")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    try:
        mlflow.set_experiment("Rent Apartment Camila")
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise

# Initialize MLflow
setup_mlflow()


@task(name="load_data", description="Load precios inmuebles", retries=3, retry_delay_seconds=10)
def load_data(path: str) -> pd.DataFrame:
    #print("cwd ->", Path.cwd())
    logger.info(f"Cargando datos desde: {path}")
    df = pd.read_csv("./data/dataset.csv", sep=";")
    logger.info(f"Datos cargados con shape: {df.shape}")
    return df

@task(name="numeric_data", description="Selecciona columnas numéricas", retries=3, retry_delay_seconds=10)
def preprocess(df: pd.DataFrame, numeric_cols=None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Selecciona columnas numéricas relevantes y estandariza.
    Devuelve X_scaled (DataFrame) y scaler (ajustado).
    """
    if numeric_cols is None:
        X = df.select_dtypes(include=[np.number]).copy()
    else:
        X = df[numeric_cols].copy()
    logger.info(f"Columnas usadas para clustering: {list(X.columns)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled_df, scaler

@task(name="apply_PCA", description="Aplica PCA a los datos", retries=3, retry_delay_seconds=10)
def apply_pca(X: pd.DataFrame, n_components: int=2) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=42)

    ########
    X.dropna(inplace=True)

    X_pca = pca.fit_transform(X)
    logger.info(f"PCA aplicado: var explicada por componente: {pca.explained_variance_ratio_}")
    return X_pca, pca

@task(name="train_model", description="Train Kmeans model with MLflow tracking", retries=3, retry_delay_seconds=10)
def train_kmeans(X: np.ndarray, n_clusters: int=3, random_state: int=42) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    logger.info(f"KMeans entrenado con {n_clusters} clusters. Inertia: {kmeans.inertia_}")
    return kmeans

@task(name="train_model", description="Train Kmeans model with MLflow tracking", retries=3, retry_delay_seconds=10)
def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    results = {}
    results["silhouette_score"] = silhouette_score(X, labels)
    results["davies_bouldin"] = davies_bouldin_score(X, labels)
    results["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    return results

def save_artifacts(out_dir: str, model, scaler=None, pca=None, df_pca=None):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "kmeans_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Modelo guardado en: {model_path}")
    if scaler is not None:
        scaler_path = os.path.join(out_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler guardado en: {scaler_path}")
    if pca is not None:
        pca_path = os.path.join(out_dir, "pca.joblib")
        joblib.dump(pca, pca_path)
        logger.info(f"PCA guardado en: {pca_path}")
    # Guardar gráfico de clusters si df_pca proporcionado
    if df_pca is not None:
        fig_path = os.path.join(out_dir, "clusters_pca.png")
        plt.figure(figsize=(7,5))
        sc = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['cluster'], cmap='tab10', s=20, alpha=0.7)
        plt.colorbar(sc, label='Cluster')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Clusters (visualización PCA)")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Gráfico guardado en: {fig_path}")

def log_mlflow(experiment_name: str, run_name: str, params: dict, metrics: dict, model, input_example, artifacts_dir: str):
    mlflow.set_tracking_uri(DEFAULT_MLRUNS)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        for k,v in params.items():
            mlflow.log_param(k, v)
        for k,v in metrics.items():
            mlflow.log_metric(k, v)
        # inferir signature (si es posible)
        try:
            signature = infer_signature(input_example, model.predict(input_example))
        except Exception:
            signature = None
        # Guardar modelo en MLflow con nombre (recomendado)
        mlflow.sklearn.log_model(sk_model=model, name="kmeans_model", signature=signature, input_example=pd.DataFrame(input_example))
        # Logear artefactos (imágenes)
        # subir todos los archivos del artifacts_dir
        for root, _, files in os.walk(artifacts_dir):
            for fname in files:
                mlflow.log_artifact(os.path.join(root, fname))
        logger.info(f"Run registrado en MLflow. Run ID: {run.info.run_id}")

# -----------------------------
# Main / CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Orquestación MLOps: clustering con PCA y MLflow")
    parser.add_argument("--input", type=str, default="data/processed/inmuebles_limpios.csv", help="Ruta al CSV de entrada")
    parser.add_argument("--out_dir", type=str, default="models", help="Directorio para guardar modelos y artefactos")
    parser.add_argument("--n_clusters", type=int, default=3, help="Número de clusters para KMeans")
    parser.add_argument("--pca_components", type=int, default=2, help="Componentes PCA para reducción (visualización)")
    parser.add_argument("--numeric_cols", type=str, nargs="*", default=None, help="Lista de columnas numéricas a usar (opcional)")
    parser.add_argument("--experiment", type=str, default="Clustering_Inmuebles_PCA", help="Nombre del experimento MLflow")
    parser.add_argument("--run_name", type=str, default="kmeans_run", help="Nombre del run en MLflow")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Argumentos: {args}")

    # 1. Cargar datos
    df = load_data(args.input)

    # 2. Preprocesar
    X_scaled_df, scaler = preprocess(df, numeric_cols=args.numeric_cols)

    # 3. PCA
    X_pca, pca = apply_pca(X_scaled_df, n_components=args.pca_components)

    # 4. Entrenar KMeans
    kmeans = train_kmeans(X_pca, n_clusters=args.n_clusters)

    # 5. Evaluar
    metrics = evaluate_clustering(X_pca, kmeans.labels_)
    logger.info(f"Métricas de clustering: {metrics}")

    # 6. Preparar dataframe PCA para visualizar y guardar
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=X_scaled_df.index)
    df_pca['cluster'] = kmeans.labels_
    # agregar algunas columnas descriptivas si existen en df
    if 'price' in df.columns:
        df_pca['price'] = df['price']

    # 7. Guardar artefactos localmente
    artifacts_dir = os.path.join(args.out_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    save_artifacts(artifacts_dir, kmeans, scaler=scaler, pca=pca, df_pca=df_pca)

    # 8. Registrar en MLflow
    # usar primeras filas escaladas como ejemplo de input para signature
    input_example = X_scaled_df.iloc[:5]
    params = {"n_clusters": args.n_clusters, "pca_components": args.pca_components, "n_features": X_scaled_df.shape[1]}
    # transformar input_example a la forma que espera el modelo (PCA space)
    input_example_pca = pca.transform(input_example)
    log_mlflow(experiment_name=args.experiment, run_name=args.run_name, params=params,
               metrics=metrics, model=kmeans, input_example=input_example_pca, artifacts_dir=artifacts_dir)

    logger.info("Proceso de orquestación finalizado.")

if __name__ == "__main__":
    main()
