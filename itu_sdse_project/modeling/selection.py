import mlflow
from mlflow.entities import Run
import typer
from loguru import logger

from itu_sdse_project.config import EXPERIMENT_NAME, MODEL_NAME

app = typer.Typer()


@app.command()
def main():
    logger.info(
        "Starting model selection. Experiment name='{}', model name='{}'",
        EXPERIMENT_NAME,
        MODEL_NAME,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.MlflowClient()

    run_id = None
    exp_ids: list[str] = [mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id]
    logger.info("Using experiment IDs: {}", exp_ids)

    logger.info("Searching for best run ordered by metrics.f1_score DESC")
    best_run_exp = mlflow.search_runs(
        experiment_ids=exp_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1,
        output_format="list",
    )[0]

    assert isinstance(best_run_exp, Run), "Type of `best_run_exp` should be `Run`"

    run_id = best_run_exp.info.run_id
    logger.info("Best run id selected: {}", run_id)

    if run_id:
        logger.info(
            "Searching for logged models for run_id='{}' to register under name='{}'",
            run_id,
            MODEL_NAME,
        )

        models = mlflow.search_logged_models(
            output_format="list", filter_string=f"source_run_id='{run_id}'"
        )

        logger.info(
            "Number of logged models found for run_id='{}': {}",
            run_id,
            len(models),
        )

        assert len(models) == 1, "There should be only one model"

        model_name = models[0].name
        model_uri = f"runs:/{run_id}/{model_name}"
        result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "staging", result.version)

        logger.success(
            "Alias 'staging' set for model='{}', version={}",
            MODEL_NAME,
            result.version,
        )
    else:
        logger.error(
            "No run_id was found while selecting best model for experiment='{}'",
            EXPERIMENT_NAME,
        )
        return


if __name__ == "__main__":
    app()
