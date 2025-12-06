from typing import Any

from loguru import logger
from mlflow.pyfunc.model import PythonModel
import pandas as pd
from sklearn.model_selection import train_test_split

from itu_sdse_project.config import PROCESSED_DATA_DIR, RANDOM_STATE


def load_data():
    features_path = PROCESSED_DATA_DIR / "features.csv"
    labels_path = PROCESSED_DATA_DIR / "labels.csv"

    logger.info("Loading processed features from {}", features_path)
    logger.info("Loading processed labels from {}", labels_path)

    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    logger.info(
        "Loaded processed data. X shape: {}, y shape: {}. Performing train/test split.",
        X.shape,
        y.shape,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=RANDOM_STATE,
        test_size=0.15,
        stratify=y,
    )

    logger.info(
        "Completed split. X_train: {}, X_test: {}, y_train: {}, y_test: {}",
        x_train.shape,
        x_test.shape,
        y_train.shape,
        y_test.shape,
    )

    return x_train, x_test, y_train, y_test


class MLFlowWrapper(PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        import joblib

        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input: pd.DataFrame, params: dict[str, Any] | None = None)-> pd.DataFrame:
        return self.model.predict_proba(model_input)[:, 1]


def describe_numeric_col(x):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats.
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"],
    )


def impute_missing_values(x, method="mean"):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method == "mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x


def create_dummy_cols(df, col):
    logger.debug("Creating dummy variables for column '{}' (original shape: {})", col, df.shape)
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df
