
import subprocess
import mlflow
import pytest

from mlflow import MlflowClient

from itu_sdse_project.config import EXPERIMENT_NAME, MODEL_NAME


@pytest.mark.parametrize("model_name", ["log-reg", "xgboost"])
def test_training_and_mlflow_logging(model_name):
    """
    Ensures 'train.py' runs successfully for both models and logs a run to MLFlow.
    """

    try:
        command = [
            "python",
            "itu_sdse_project/modeling/train.py",
            model_name
        ]

        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Training for {model_name} failed. Error: {e.stderr}")

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        assert len(runs) > 0, f"No MLFlow run logged for {model_name} training."

        latest_run = runs[0]
        assert 'f1_score' in latest_run.data.metrics, (
            f"F1-score metric not logged in MLFlow run for {model_name}."
        )

    except Exception as e:
        pytest.fail(f"MLFlow validation failed after training {model_name}. Error: {e}")


def test_model_selection_and_registration():
    """
    Verifies 'selection.py' chooses the model with the highest f1-score
    and sets the 'Staging' ALIAS on the correct model version.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )

    if not runs:
        pytest.skip("No runs available in MLFlow to test selection.")

    best_run_id_pre = runs[0].info.run_id
    best_f1_score = runs[0].data.metrics['f1_score']

    try:
        subprocess.run(
            ["python", "itu_sdse_project/modeling/selection.py"],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Model selection failed. Error: {e.stderr}")

    try:
        latest_version = client.get_model_version_by_alias(MODEL_NAME, "staging")

        assert "staging" in latest_version.aliases, (
            "The retrieved model version does not have the expected 'Staging' alias."
        )

        assert latest_version.run_id == best_run_id_pre, (
            f"Staging alias was set on Run ID {latest_version.run_id} but should have been "
            f"set on the best run ID {best_run_id_pre} (f1: {best_f1_score})."
        )

    except Exception as e:
        if "RESOURCE_NOT_FOUND" in str(e):
            pytest.fail(
                f"MLFlow selection verification failed: Model was not assigned the 'Staging' alias. "
                "The selection script likely failed to register the model or set the alias."
            )
        else:
            pytest.fail(f"MLFlow selection verification failed. Error: {e}")



