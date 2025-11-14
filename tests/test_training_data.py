import pandas as pd

from itu_sdse_project.config import PROCESSED_DATA_DIR


def test_training_data_integrity():
    true_df = pd.read_csv("tests/data/training_data.csv")
    generated_df = pd.read_csv(PROCESSED_DATA_DIR / "training_data.csv")

    assert true_df.equals(generated_df)
