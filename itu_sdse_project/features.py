import pandas as pd
import typer

from itu_sdse_project.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from itu_sdse_project.helpers import create_dummy_cols

app = typer.Typer()


@app.command()
def main():
    data = pd.read_csv(INTERIM_DATA_DIR / "cleaned_data.csv")

    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]
    other_vars = data.drop(cat_cols, axis=1)
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data:
        data[col] = data[col].astype("float64")
        print(f"Changed column {col} to float")

    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    y.to_csv(PROCESSED_DATA_DIR / "labels.csv", index=False)
    X.to_csv(PROCESSED_DATA_DIR / "features.csv", index=False)

    pass


if __name__ == "__main__":
    app()
