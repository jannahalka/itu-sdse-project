from typing import Any

from mlflow.pyfunc.model import PythonModel
import pandas as pd
from sklearn.model_selection import train_test_split

from itu_sdse_project.config import PROCESSED_DATA_DIR, RANDOM_STATE


def load_data():
    X = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    y = pd.read_csv(PROCESSED_DATA_DIR / "labels.csv")

    return train_test_split(X, y, random_state=RANDOM_STATE, test_size=0.15, stratify=y)


class MLFlowWrapper(PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        import joblib

        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
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
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df
