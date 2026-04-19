from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .typing_defs import SplitResult


def select_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    ignore = {"case_id", "year", target_col, "source_file"}
    return [c for c in df.columns if c not in ignore]


def fit_xgb_regressor(
    df: pd.DataFrame,
    target_col: str,
    params: dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
):
    feature_cols = select_feature_columns(df, target_col)
    X = df[feature_cols]
    y = df[target_col]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    preproc = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    model = XGBRegressor(**params)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_arr = preproc.fit_transform(X_train)
    X_test_arr = preproc.transform(X_test)

    model.fit(X_train_arr, y_train)
    preds = model.predict(X_test_arr)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }

    split_info = SplitResult(
        train_rows=int(X_train.shape[0]),
        test_rows=int(X_test.shape[0]),
        feature_count=len(feature_cols),
    )

    artifacts = {
        "model": model,
        "preprocessor": preproc,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
    }

    return metrics, asdict(split_info), artifacts
