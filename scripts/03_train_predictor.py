from __future__ import annotations

import argparse
import pickle

from salt_ml.data_utils import ensure_parent_dir, load_config, read_csv, save_json
from salt_ml.feature_engineering import create_case_level_targets
from salt_ml.modeling import fit_xgb_regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prediction model")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    df = read_csv(cfg["paths"]["processed_csv"])
    target_name = cfg["target"]["name"]
    forecast_year = int(cfg["target"]["forecast_year"])

    case_table = create_case_level_targets(df, forecast_year=forecast_year, target=target_name)

    metrics, split_info, artifacts = fit_xgb_regressor(
        df=case_table,
        target_col=target_name,
        params=cfg["model"]["params"],
        test_size=float(cfg["model"]["test_size"]),
        random_state=int(cfg["model"]["random_state"]),
    )

    model_path = cfg["paths"]["model_path"]
    ensure_parent_dir(model_path)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": artifacts["model"],
                "preprocessor": artifacts["preprocessor"],
                "feature_cols": artifacts["feature_cols"],
            },
            f,
        )

    save_json({"metrics": metrics, "split": split_info}, cfg["paths"]["metrics_json"])
    print(f"Metrics: {metrics}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
