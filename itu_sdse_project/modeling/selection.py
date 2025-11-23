import mlflow
from mlflow.entities import Run
import typer

from itu_sdse_project.config import EXPERIMENT_NAME, MODEL_NAME

app = typer.Typer()


@app.command()
def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = mlflow.MlflowClient()

    run_id = None
    exp_ids: list[str] = [mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id]

    best_run_exp = mlflow.search_runs(
        experiment_ids=exp_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1,
        output_format="list",
    )[0]

    assert isinstance(best_run_exp, Run), "Type of `best_run_exp` should be `Run`"

    run_id = best_run_exp.info.run_id

    if run_id:
        models = mlflow.search_logged_models(
            output_format="list", filter_string=f"source_run_id='{run_id}'"
        )

        assert len(models) == 1, "There should be only one model"

        model_name = models[0].name
        model_uri = f"runs:/{run_id}/{model_name}"
        result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "staging", result.version)
    else:
        # TODO: Log err
        return


if __name__ == "__main__":
    app()
