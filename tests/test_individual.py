import pytest
import polars as pl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from torch.nn import Module

from catchmeifyoucan.config import RAW_DATA_DIR
from catchmeifyoucan.modeling.metrics import return_cutoff_matrix, return_metrics
from catchmeifyoucan.modeling.unsupervised.anomaly import ForestBased, AutoEncoderBased

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

    y = time_transformed_df["class"].to_numpy()
    x = enc.fit_transform(
        time_transformed_df.drop("class").to_numpy(),
        time_transformed_df["class"].to_numpy(),
    )

    train_x, test_x, train_y , test_y = train_test_split(x,y,
                                                            shuffle= True,
                                                            random_state=42)
    return train_x, test_x, train_y, test_y


def test_forest_based_no_y():
    assert True


def test_forest_based_with_y(fetch_ecommerce_dataset):
    train_x, test_x, train_y , test_y = fetch_ecommerce_dataset
    
    model = ForestBased()
    model.fit(X=train_x, y=train_y)
    preds = model.predict(test_x).ravel()

    assert np.max(preds) == 1. and np.min(preds) == -1.
    assert test_x.shape[0] == len(preds)

def test_autoencoder_train(fetch_ecommerce_dataset):
    train_x, _,_,_ = fetch_ecommerce_dataset
    
    model = AutoEncoderBased()
    model.fit(X=train_x)

    assert isinstance(model, Module)


def test_metrics(fetch_ecommerce_dataset):
    lr = LogisticRegressionCV()
    train_x,test_x, train_y, test_y = fetch_ecommerce_dataset

    lr.fit(X = train_x, y = train_y)
    
    preds = lr.predict_proba(test_x)[:,1]
    
    basic_metrics = return_metrics(preds, test_y)
    cutoff_matrix = return_cutoff_matrix(preds, test_y)
    
    assert all([metric in basic_metrics.keys() for metric in ["roc_auc_score", 
                                                                "brier_score",
                                                                "average_precision_score"]])
    assert isinstance(cutoff_matrix,  pd.DataFrame)
