import numpy as np
from sklearn import metrics


def recall_p(tpr: np.ndarray, fpr: np.ndarray, fp_p: float) -> float:
    """Calculate recall for a given False Positive Rate

    Args:
        tpr (np.ndarray): True Positive Rates array
        fpr (np.ndarray): False Positive Rates array
        fp_p (float): False Positive Rate threshold

    Returns:
        float: recall
    """
    index = np.argmin(abs(np.flip(fpr[:]) - fp_p))
    return np.flip(tpr[:])[index]


def fpr_p(tpr: np.ndarray, fpr: np.ndarray, tp_p: float) -> float:
    """Calculate False Positive Rate for a given True Positive Rate

    Args:
        tpr (np.ndarray): True Positive Rates array
        fpr (np.ndarray): False Positive Rates array
        tp_p (float): True Positive Rate threshold

    Returns:
        float: fpr
    """
    index = np.argmin(abs(tpr[:] - tp_p))
    return fpr[index]


def compute_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds_roc = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels,
        anomaly_prediction_weights,
    )
    precision, recall, thresholds_pr = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    aupr = metrics.average_precision_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    recalls = {
        "1p": recall_p(tpr, fpr, 0.01),
        "2p": recall_p(tpr, fpr, 0.02),
        "5p": recall_p(tpr, fpr, 0.05),
        "10p": recall_p(tpr, fpr, 0.1),
        "20p": recall_p(tpr, fpr, 0.2),
    }

    fprs = {
        "99p": fpr_p(tpr, fpr, 0.99),
        "95p": fpr_p(tpr, fpr, 0.95),
        "90p": fpr_p(tpr, fpr, 0.90),
    }

    f1_scores = np.divide(  # noqa
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "fprs": fprs,
        "F1": f1_scores,
        "threshold_roc": thresholds_roc,
        "aupr": aupr,
        "precision": precision,
        "recall": recall,
        "recalls": recalls,
        "thresholds_pr": thresholds_pr,
    }
