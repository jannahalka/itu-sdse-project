import datetime
from pathlib import Path

import joblib
import mlflow
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
import typer

from itu_sdse_project.config import DATA_VERSION, EXPERIMENT_NAME, MODELS_DIR, RANDOM_STATE
from itu_sdse_project.helpers import MLFlowWrapper, load_data

app = typer.Typer()


@app.command()
def xgboost(output_path: Path = MODELS_DIR / "xgboost.pkl"):
    from scipy.stats import randint, uniform
    from xgboost import XGBRFClassifier

    X_train, X_test, y_train, y_test = load_data()

    # TODO: The parameters are defined incorrectly
    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["binary:logistic"],
        "eval_metric": ["aucpr", "error"],
    }

    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"xgboost_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        model = XGBRFClassifier(random_state=RANDOM_STATE)
        model_grid = RandomizedSearchCV(
            model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10
        )
        model_grid.fit(X_train, y_train)
        best_model = model_grid.best_estimator_

        y_pred_test = model_grid.predict(X_test)

        joblib.dump(value=best_model, filename=output_path)

        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test))
        mlflow.log_params(model_grid.best_params_)
        mlflow.log_param("data_version", DATA_VERSION)

        mlflow.pyfunc.log_model(
            name="xgb_model_tuned",
            python_model=MLFlowWrapper(best_model),
            artifacts={"model": str(output_path)},
            input_example= X_train.head(5)
        )


@app.command()
def log_reg(output_path: Path = MODELS_DIR / "logreg.pkl"):
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = load_data()

    params = {
        "solver": ["lbfgs", "saga"],
        "penalty": ["l2"],
        "C": [100, 10, 1.0, 0.1, 0.01],
    }

    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"log_reg_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        model = LogisticRegression()
        model_grid = RandomizedSearchCV(
            model, param_distributions=params, verbose=3, n_iter=10, cv=3
        )
        model_grid.fit(X_train, y_train)
        best_model = model_grid.best_estimator_

        y_pred_test = model_grid.predict(X_test)

        joblib.dump(value=best_model, filename=output_path)

        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test))
        mlflow.log_params(model_grid.best_params_)
        mlflow.log_param("data_version", DATA_VERSION)

        mlflow.pyfunc.log_model(
            name="lr_model_tuned",
            python_model=MLFlowWrapper(best_model),
            artifacts={"model": str(output_path)},
            input_example=X_train.head(5)
        )


if __name__ == "__main__":
    app()
