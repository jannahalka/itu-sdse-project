import pandas as pd

from itu_sdse_project.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
def test_training_data_integrity():
    true_df = pd.read_csv("tests/data/training_data.csv")
    generated_df = pd.read_csv(INTERIM_DATA_DIR / "cleaned_data.csv")

    assert true_df.equals(generated_df)


def test_training_data_split_integrity():
    true_x_df = pd.read_csv("tests/data/X.csv")
    true_y_df = pd.read_csv("tests/data/y.csv")
    generated_x_df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    generated_y_df = pd.read_csv(PROCESSED_DATA_DIR / "labels.csv")

    assert true_x_df.equals(generated_x_df)
    assert true_y_df.equals(generated_y_df)
