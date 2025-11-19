from mlflow.tracking import MlflowClient
import typer

from itu_sdse_project.config import PROD_MODEL_NAME

app = typer.Typer()


@app.command()
def main(model_version: str):
    client = MlflowClient()
    model_version_details = dict(
        client.get_model_version(name=PROD_MODEL_NAME, version=model_version)
    )
    if model_version_details["current_stage"] != "Staging":
        client.transition_model_version_stage(
            name=PROD_MODEL_NAME,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True,
        )
    else:
        print("Model already in staging")


if __name__ == "__main__":
    app()
