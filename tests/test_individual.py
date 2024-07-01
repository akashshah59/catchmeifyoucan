import pytest
import polars as pl
import numpy as np

from catchmeifyoucan.config import RAW_DATA_DIR
from catchmeifyoucan.modeling.unsupervised.anomaly import ForestBased


@pytest.fixture
def fetch_ecommerce_dataset():
    from category_encoders import CatBoostEncoder

    enc = CatBoostEncoder()
    df = pl.read_csv(RAW_DATA_DIR/ "e_commerce" / "Fraud_data.csv")

    time_transformed_df = df.with_columns(
        pl.col("user_id").cast(str).alias("user_id"),
        pl.col("ip_address").cast(str).alias("ip_address"),
        df["signup_time"].str.to_datetime().dt.year().alias("signup_year"),
        df["signup_time"].str.to_datetime().dt.month().alias("signup_month"),
        df["signup_time"].str.to_datetime().dt.day().alias("signup_day"),
        df["purchase_time"].str.to_datetime().dt.year().alias("purchase_year"),
        df["purchase_time"].str.to_datetime().dt.month().alias("purchase_month"),
        df["purchase_time"].str.to_datetime().dt.day().alias("purchase_day"),
    ).drop(["signup_time", "purchase_time"])

    y = df["class"]
    x = enc.fit_transform(
        time_transformed_df.drop("class").to_pandas(),
        time_transformed_df["class"].to_pandas(),
    )
    return (x, y)


def test_forest_based_no_y():
    assert True


def test_forest_based_with_y(fetch_ecommerce_dataset):
    model = ForestBased()
    x, y = fetch_ecommerce_dataset

    model.fit(X=x, y=y)
    preds = model.predict(x)

    assert np.max(preds) == 1. and np.min(preds) == -1.
    assert x.shape[0] == len(preds)
