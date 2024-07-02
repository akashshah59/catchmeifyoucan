from pathlib import Path

import typer
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from catchmeifyoucan.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)


def return_metrics(score_array,
                   true_array):
    return {
        "roc_auc_score": roc_auc_score(true_array, score_array),
        "average_precision_score": average_precision_score(true_array, score_array),
        "brier_score": brier_score_loss(true_array, score_array),
    }


def return_cutoff_matrix(score_array, true_array):
    assert (np.dtype(score_array) == np.array) & (np.dtype(true_array) == np.array)

    precisions, recalls, f1_scores, percentiles, tps, tns, fns, fps = ([],) * 8

    for cutoffs in np.linspace(0, 100, 21):
        # 20 cutoffs taken only for interpretability.
        predictions = (score_array >= np.percentile(score_array, cutoffs)).astype(int)
        percentiles.append(np.percentile(score_array, cutoffs))
        recalls.append(recall_score(true_array, predictions))
        precisions.append(precision_score(true_array, predictions))
        f1_scores.append(f1_score(true_array, predictions))
        tn, fp, fn, tp = confusion_matrix(true_array, predictions).ravel()
        tns.append(tn)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)

    cutoff_table = pd.DataFrame(
        {
            "percentile": np.linspace(0, 100, 21),
            "thresholds": percentiles,
            "recall": recalls,
            "precision": precisions,
            "f1_scores": f1_scores,
            "true_positive": tps,
            "true_negatives": tns,
            "false_positives": fps,
            "false_negatives": fns,
        }
    ).sort_values("percentile", ascending=False)

    return cutoff_table
