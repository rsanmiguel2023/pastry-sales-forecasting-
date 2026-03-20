"""Model training utilities for pastry sales forecasting."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


TARGET_COL = "sales"

DEFAULT_FEATURES = [
    "temperature_mean",
    "sunshine_sum",
    "precipitation_sum_boxcox",
    "unsold_boxcox",
    "ordered_boxcox",
    "store_num",
    "is_state_holiday_num",
    "is_school_holiday_num",
    "is_special_day_num",
    "day_of_week",
    "month",
    "is_weekend",
    "sales_lag_1",
    "sales_lag_7",
    "sales_rolling_mean_7",
    "sales_rolling_mean_14",
]


def get_model_dict(random_state: int = 42) -> dict[str, Any]:
    """Initialize the candidate regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=random_state,
            max_depth=6,
        ),
        "Random Forest": RandomForestRegressor(
            random_state=random_state,
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            random_state=random_state,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
        ),
    }


def select_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> list[str]:
    """Keep only valid features present in the processed dataset."""
    return [col for col in DEFAULT_FEATURES if col in df.columns and col != target_col]


def split_data(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split the processed dataset into random train and test subsets."""
    model_df = df.dropna(subset=features + [target_col]).copy()
    X = model_df[features]
    y = model_df[target_col]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def time_based_split(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = TARGET_COL,
    date_col: str = "date",
    valid_fraction: float = 0.2,
):
    """Split the processed dataset by time instead of random shuffling."""
    model_df = df.dropna(subset=features + [target_col, date_col]).copy()
    model_df = model_df.sort_values(date_col).reset_index(drop=True)

    cutoff_index = int(len(model_df) * (1 - valid_fraction))
    train_df = model_df.iloc[:cutoff_index].copy()
    valid_df = model_df.iloc[cutoff_index:].copy()

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_valid = valid_df[features]
    y_valid = valid_df[target_col]

    return X_train, X_valid, y_train, y_valid, train_df, valid_df


def evaluate_single_model(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_splitter,
) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    """Train and evaluate one model."""
    cv_mse = -cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=cv_splitter,
    ).mean()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    residuals = y_test - preds

    result_row = {
        "model": model_name,
        "cv_mse": round(float(cv_mse), 6),
        "test_mse": round(float(mean_squared_error(y_test, preds)), 6),
        "test_r2": round(float(r2_score(y_test, preds)), 6),
        "n_features": X_train.shape[1],
    }

    prediction_df = pd.DataFrame(
        {
            "model": model_name,
            "actual": y_test.values,
            "predicted": preds,
            "residual": residuals.values,
        }
    )

    return result_row, prediction_df, model


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    """Train all models with a random split and compare performance."""
    features = select_features(df, target_col=target_col)
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        features=features,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    return _run_model_loop(X_train, X_test, y_train, y_test, kf, features, random_state)


def train_and_evaluate_time_based(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    valid_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str], pd.DataFrame, pd.DataFrame]:
    """Train all models using time-based validation."""
    features = select_features(df, target_col=target_col)
    X_train, X_valid, y_train, y_valid, train_df, valid_df = time_based_split(
        df=df,
        features=features,
        target_col=target_col,
        valid_fraction=valid_fraction,
    )

    n_splits = min(5, max(2, len(X_train) // 500))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results_df, predictions_df, trained_models, features = _run_model_loop(
        X_train, X_valid, y_train, y_valid, tscv, features, random_state
    )

    return results_df, predictions_df, trained_models, features, train_df, valid_df


def _run_model_loop(X_train, X_test, y_train, y_test, cv_splitter, features, random_state):
    """Internal helper to evaluate the full candidate model set."""
    rows = []
    prediction_frames = []
    trained_models = {}

    for model_name, model in get_model_dict(random_state=random_state).items():
        result_row, prediction_df, trained_model = evaluate_single_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_splitter=cv_splitter,
        )
        rows.append(result_row)
        prediction_frames.append(prediction_df)
        trained_models[model_name] = trained_model

    results_df = pd.DataFrame(rows).sort_values("test_mse").reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    return results_df, predictions_df, trained_models, features


def train_full_model(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COL,
    random_state: int = 42,
):
    """Train the best model on the full training dataset.

    The best model is chosen using time-based validation results, then
    refit on the full available training data.
    """
    results_df, _, _, features, _, _ = train_and_evaluate_time_based(
        df=df,
        target_col=target_col,
        random_state=random_state,
    )

    if feature_cols is not None:
        features = [col for col in feature_cols if col in df.columns]

    best_model_name = results_df.iloc[0]["model"]
    best_model = get_model_dict(random_state=random_state)[best_model_name]

    model_df = df.dropna(subset=features + [target_col]).copy()
    X = model_df[features]
    y = model_df[target_col]

    best_model.fit(X, y)

    return best_model_name, best_model, features, results_df


def build_summary_payload(results_df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact summary dictionary from model results."""
    if results_df.empty:
        return {
            "best_model": None,
            "best_test_mse": None,
            "best_test_r2": None,
        }

    best_row = results_df.iloc[0]
    return {
        "best_model": best_row["model"],
        "best_test_mse": float(best_row["test_mse"]),
        "best_test_r2": float(best_row["test_r2"]),
    }
