import pandas as pd
import pytest

from itu_sdse_project.config import PROCESSED_DATA_DIR


def test_feature_label_row_consistency():
    try:
        X = pd.read_csv(f"{PROCESSED_DATA_DIR}/features.csv")
        y = pd.read_csv(f"{PROCESSED_DATA_DIR}/labels.csv")
    except FileNotFoundError as e:
        pytest.fail(f"Required processed files not found. Ensure 'features.py' was run. Error: {e}")

    assert len(X) == len(y), (
        f"Feature matrix X has {len(X)} rows, but label vector y has {len(y)} rows. "
        "They must be equal."
    )


def test_no_missing_values_in_processed_data():
    X = pd.read_csv(f"{PROCESSED_DATA_DIR}/features.csv")
    y = pd.read_csv(f"{PROCESSED_DATA_DIR}/labels.csv")

    X_null_count = X.isnull().sum().sum()
    assert X_null_count == 0, (
        f"Found {X_null_count} missing values in the features.csv. "
        "The feature engineering script should handle all missing data."
    )

    y_null_count = y.isnull().sum().sum()
    assert y_null_count == 0, (
        f"Found {y_null_count} missing values in the labels.csv. "
        "The label generation script should ensure a label for every row."
    )


def test_labels_are_binary():
    y = pd.read_csv(f"{PROCESSED_DATA_DIR}/labels.csv")

    label_column = y.iloc[:, 0]
    unique_labels = label_column.unique()

    assert len(unique_labels) == 2, (
        f"Expected exactly 2 unique labels for binary classification, but found {len(unique_labels)}: "
        f"{unique_labels}. Check data generation logic."
    )

    expected_values = {0, 1}
    actual_values = set(unique_labels)
    assert actual_values == expected_values, (
        f"Expected unique labels to be {expected_values} but found {actual_values}. "
        "Check how conversion status is encoded."
    )