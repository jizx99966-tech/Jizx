from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
import shap

from salt_ml.data_utils import ensure_parent_dir, load_config, read_csv, save_csv
from salt_ml.feature_engineering import create_case_level_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature importance & coupling analysis")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    df = read_csv(cfg["paths"]["processed_csv"])
    target = cfg["target"]["name"]
    forecast_year = int(cfg["target"]["forecast_year"])
    case_table = create_case_level_targets(df, forecast_year=forecast_year, target=target)

    with open(cfg["paths"]["model_path"], "rb") as f:
        bundle = pickle.load(f)

    feature_cols = bundle["feature_cols"]
    preproc = bundle["preprocessor"]
    model = bundle["model"]

    X = case_table[feature_cols]
    X_arr = preproc.transform(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_arr)

    importance = np.abs(shap_values).mean(axis=0)
    out_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False
    )

    ensure_parent_dir(cfg["paths"]["shap_csv"])
    save_csv(out_df, cfg["paths"]["shap_csv"])
    print(f"Saved SHAP importance: {cfg['paths']['shap_csv']}")


if __name__ == "__main__":
    main()
