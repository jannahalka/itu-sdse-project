from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import typer

from itu_sdse_project.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE

app = typer.Typer()

model_results = {}

X = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
y = pd.read_csv(PROCESSED_DATA_DIR / "labels.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)


@app.command()
def xgboost(output_path: Path = MODELS_DIR / "xgboost.json"):
    from scipy.stats import randint, uniform
    from xgboost import XGBRFClassifier

    model = XGBRFClassifier(RANDOM_STATE)
    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"],
    }
    model_grid = RandomizedSearchCV(
        model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10
    )
    model_grid.fit(X_train, y_train)

    y_pred_train = model_grid.predict(X_train)

    model_results[output_path] = classification_report(y_train, y_pred_train, output_dict=True)

    xgboost_model = model_grid.best_estimator_
    xgboost_model.save_model(output_path)


@app.command()
def log_reg(output_path: Path = MODELS_DIR / "logreg.pkl"):
    import joblib
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    params = {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["none", "l1", "l2", "elasticnet"],
        "C": [100, 10, 1.0, 0.1, 0.01],
    }
    model_grid = RandomizedSearchCV(model, param_distributions=params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)
    best_model = model_grid.best_estimator_

    joblib.dump(value=best_model, filename=output_path)

    y_pred_test = model_grid.predict(X_test)

    model_results[output_path] = classification_report(y_test, y_pred_test, output_dict=True)


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    pass


if __name__ == "__main__":
    app()
