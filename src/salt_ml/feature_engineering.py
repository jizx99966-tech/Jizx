from __future__ import annotations

import numpy as np
import pandas as pd


def build_base_features(static_df: pd.DataFrame, yearly_df: pd.DataFrame) -> pd.DataFrame:
    merged = yearly_df.merge(static_df, on="case_id", how="left", validate="many_to_one")
    merged = merged.sort_values(["case_id", "year"]).reset_index(drop=True)

    grouped = merged.groupby("case_id", group_keys=False)
    merged["vsr_d1"] = grouped["VolShrinkRatio"].diff().fillna(0.0)
    merged["maxdisp_d1"] = grouped["MaxDisp"].diff().fillna(0.0)

    baseline = merged[grouped.cumcount() == 0][["case_id", "VolShrinkRatio", "TotalVolumn"]].rename(
        columns={"VolShrinkRatio": "VolShrinkRatio_0", "TotalVolumn": "TotalVolumn_0"}
    )
    merged = merged.merge(baseline, on="case_id", how="left")
    merged["VolShrinkRatio_rel0"] = merged["VolShrinkRatio"] - merged["VolShrinkRatio_0"]
    merged["TotalVolumn_rel0"] = merged["TotalVolumn"] - merged["TotalVolumn_0"]

    if {"SH", "JH"}.issubset(merged.columns):
        merged["SH_JH_ratio"] = merged["SH"] / merged["JH"].replace(0, np.nan)

    if {"HP", "LP"}.issubset(merged.columns):
        merged["PressureSpan"] = merged["HP"] - merged["LP"]

    return merged


def add_interactions(df: pd.DataFrame, interaction_pairs: list[list[str]] | None) -> pd.DataFrame:
    if not interaction_pairs:
        return df

    out = df.copy()
    for pair in interaction_pairs:
        if len(pair) != 2:
            continue
        a, b = pair
        if a in out.columns and b in out.columns:
            out[f"{a}__x__{b}"] = out[a] * out[b]
    return out


def create_case_level_targets(df: pd.DataFrame, forecast_year: int, target: str) -> pd.DataFrame:
    final_df = df[df["year"] == forecast_year].copy()
    keep_cols = [c for c in final_df.columns if c not in {"source_file"}]
    return final_df[keep_cols + [target]] if target not in keep_cols else final_df[keep_cols]
