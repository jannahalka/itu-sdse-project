# Cleans and normalizes raw dataset

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from itu_sdse_project.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from itu_sdse_project.helpers import impute_missing_values

output_path: Path = INTERIM_DATA_DIR / "cleaned_data.csv"
input_path: Path = RAW_DATA_DIR / "raw_data.csv"

if __name__ == "__main__":
    data = pd.read_csv(input_path)

    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data.drop(
        ["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"],
        axis=1,
    )
    data = data.drop(
        ["domain", "country", "visited_learn_more_before_booking", "visited_faq"], axis=1
    )
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)
    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])
    data = data[data.source == "signup"]
    vars = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]

    for col in vars:
        data[col] = data[col].astype("object")

    cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]
    cat_vars = data.loc[:, (data.dtypes == "object")]

    cont_vars = cont_vars.apply(
        lambda x: x.clip(lower=(x.mean() - 2 * x.std()), upper=(x.mean() + 2 * x.std()))
    )
    cont_vars = cont_vars.apply(impute_missing_values)
    cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"
    cat_vars = cat_vars.apply(impute_missing_values)

    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)

    mapping = {"li": "socials", "fb": "socials", "organic": "group1", "signup": "group1"}

    data["bin_source"] = data["source"].map(mapping)
    data.to_csv(output_path, index=False)
