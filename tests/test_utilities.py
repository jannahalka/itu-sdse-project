import pytest
import importlib
import numpy as np
from pathlib import Path
import pandas as pd

import itu_sdse_project.helpers as helpers
from itu_sdse_project import config

EXPECTED_CONFIG_SCHEMA = {
    # Core Settings
    "RANDOM_STATE": int,
    "DATA_VERSION": str,
    "EXPERIMENT_NAME": str,
    "MODEL_NAME": str,

    # Paths
    "PROJ_ROOT": Path,
    "DATA_DIR": Path,
    "RAW_DATA_DIR": Path,
    "INTERIM_DATA_DIR": Path,
    "PROCESSED_DATA_DIR": Path,
    "EXTERNAL_DATA_DIR": Path,
    "MODELS_DIR": Path,
    "REPORTS_DIR": Path,
    "FIGURES_DIR": Path,
}


def test_config_schema_integrity():
    """
    Verifies that all necessary configuration variables exist and have the correct data types.
    """
    for var_name, expected_type in EXPECTED_CONFIG_SCHEMA.items():

        assert hasattr(config, var_name), f"Config file is missing required variable: {var_name}"

        actual_value = getattr(config, var_name)
        assert isinstance(actual_value, expected_type), (
            f"Config variable {var_name} has type {type(actual_value)}, "
            f"but expected type(s): {expected_type}"
        )



def test_create_dummy_cols_basic():
    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Color': ['Red', 'Blue', 'Red']
    })

    df_expected = pd.DataFrame({
        'ID': [1, 2, 3],
        'Color_Red': [1, 0, 1]
    })

    df_result = helpers.create_dummy_cols(df.copy(), 'Color')

    assert set(df_result.columns) == set(df_expected.columns)



def test_impute_missing_values_numeric_mean():
    """
    Tests imputation using the 'mean' method for numeric data.
    """
    series = pd.Series([1.0, 2.0, np.nan, 3.0, np.nan])

    expected_series = pd.Series([1.0, 2.0, 2.0, 3.0, 2.0])

    imputed_series = helpers.impute_missing_values(series, method="mean")

    assert imputed_series.equals(expected_series), "Mean imputation failed for numeric data."