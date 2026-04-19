from __future__ import annotations

import argparse

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from salt_ml.data_utils import load_config, read_csv, save_csv


SEQUENCE_COLS = ["VolShrinkRatio", "MaxDisp", "MaxAbsUx", "MaxAbsUz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case-level temporal pattern clustering")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def build_case_sequence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    wide_parts = []
    for col in SEQUENCE_COLS:
        pivoted = df.pivot(index="case_id", columns="year", values=col)
        pivoted.columns = [f"{col}_y{int(c)}" for c in pivoted.columns]
        wide_parts.append(pivoted)
    return pd.concat(wide_parts, axis=1).sort_index()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    df = read_csv(cfg["paths"]["processed_csv"])
    matrix = build_case_sequence_matrix(df)

    scaler = StandardScaler()
    mat_z = scaler.fit_transform(matrix)

    pca_components = int(cfg["clustering"].get("pca_components", 8))
    pca = PCA(n_components=min(pca_components, mat_z.shape[1]))
    emb = pca.fit_transform(mat_z)

    n_clusters = int(cfg["clustering"].get("n_clusters", 4))
    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg.get("seed", 42), n_init="auto")
    labels = kmeans.fit_predict(emb)

    emb_df = pd.DataFrame(emb, index=matrix.index)
    emb_df.insert(0, "case_id", emb_df.index)
    emb_df = emb_df.reset_index(drop=True)

    cluster_df = pd.DataFrame({"case_id": matrix.index, "cluster": labels})

    save_csv(emb_df, cfg["paths"]["embedding_csv"])
    save_csv(cluster_df, cfg["paths"]["cluster_csv"])
    print(f"Saved embeddings: {cfg['paths']['embedding_csv']}")
    print(f"Saved clusters: {cfg['paths']['cluster_csv']}")


if __name__ == "__main__":
    main()
