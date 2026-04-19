from __future__ import annotations

import argparse

from salt_ml.data_utils import load_config, read_csv, save_csv, validate_yearly_columns
from salt_ml.feature_engineering import add_interactions, build_base_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare model-ready table from raw case/static data")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    static_df = read_csv(cfg["paths"]["static_csv"])
    yearly_df = read_csv(cfg["paths"]["yearly_csv"])
    validate_yearly_columns(yearly_df)

    table = build_base_features(static_df=static_df, yearly_df=yearly_df)
    table = add_interactions(table, cfg.get("features", {}).get("include_interactions"))

    save_csv(table, cfg["paths"]["processed_csv"])
    print(f"Saved processed table: {cfg['paths']['processed_csv']} | rows={len(table)}")


if __name__ == "__main__":
    main()
