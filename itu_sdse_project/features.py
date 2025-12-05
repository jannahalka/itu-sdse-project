import pandas as pd
import typer
from loguru import logger

from itu_sdse_project.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from itu_sdse_project.helpers import create_dummy_cols

app = typer.Typer()


@app.command()
def main():
    input_path = INTERIM_DATA_DIR / "cleaned_data.csv"
    logger.info("Starting feature engineering from {}", input_path)

    data = pd.read_csv(input_path)
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]
    other_vars = data.drop(cat_cols, axis=1)

    logger.info("Starting dummy encoding for categorical variables: {}", cat_cols)
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)
        logger.debug("Dummy creation finished for '{}'. Current cat_vars shape: {}", col, cat_vars.shape)

    data = pd.concat([other_vars, cat_vars], axis=1)
    logger.info("Recombined dataframe shape after dummy encoding: {}", data.shape)


    for col in data:
        data[col] = data[col].astype("float64")
        logger.debug("Converted column '{}' to float64", col)

    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    labels_path = PROCESSED_DATA_DIR / "labels.csv"
    features_path = PROCESSED_DATA_DIR / "features.csv"

    y.to_csv(labels_path, index=False)
    X.to_csv(features_path, index=False)

    logger.success("Saved processed labels to {} and features to {}", labels_path, features_path)


if __name__ == "__main__":
    app()
