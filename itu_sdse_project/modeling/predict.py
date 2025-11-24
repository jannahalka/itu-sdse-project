# TODO: move this to test, then delete this file
from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from itu_sdse_project.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "X_test.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "y_test.csv",
):
    logger.info("Starting inference.")
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    X = pd.read_csv(features_path)
    y = pd.read_csv(predictions_path)

    predictions = model.predict(X.head(5))
    logger.debug("Sample predictions: {}", predictions)
    logger.debug("Sample true labels: {}", y.head(5).values.tolist())
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
