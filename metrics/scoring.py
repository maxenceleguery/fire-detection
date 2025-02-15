import os
from typing import Callable

import matplotlib.pyplot as plt
import mlflow
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


class Scoring:
    def __init__(
        self,
        anomaly_score: np.ndarray,
        anomaly_labels: np.ndarray,
        fraction: float = 1.0,
    ) -> None:
        """Scoring class

        Args:
            anomaly_score (np.ndarray): List of all score (higher means anomaly, lower means normal).
            anomaly_labels (np.ndarray): Labels (0 : True label, 1 : anomaly label).
        """
        self.anomaly_score = anomaly_score
        self.anomaly_labels = anomaly_labels

        self.scores = np.array(anomaly_score)
        self.labels = np.array(anomaly_labels)
        self.labels = 1 - self.labels

        self.true_label = 1
        self.anomaly_label = 0

        indexes = np.random.choice(
            self.labels.shape[0], int(fraction * self.labels.shape[0]), replace=False
        )
        self.scores = self.scores[indexes]
        self.labels = self.labels[indexes]

        self.fpr, self.tpr, self.thresholds_roc = metrics.roc_curve(
            self.labels, -self.scores
        )

        self.fpr2, self.fnr2, self.thresholds_det = metrics.det_curve(
            self.labels, -self.scores
        )

    @staticmethod
    def _register(metric_function):
        """Decorator : Register a method as a function which returns a metric. Those methods have now a attribute is_metric set as True.

        Args:
            metric_function (function): Method that returns a metric.

        Returns:
            function: function registred.
        """
        setattr(metric_function, "is_metric", True)
        return metric_function

    def _get_best_metric(
        self, get_metric_func: Callable[..., float], num_trials: int
    ) -> tuple[float, float, np.ndarray]:
        """Method to retrieve the best metric and threshold from a mapping (threshold -> metric).

        Args:
            get_metric_func (Callable[..., float]): Metric function.
            num_trials (int): Number of thresholds to test.

        Returns:
            tuple[float, float, np.ndarray]: Tuple with : the best metric, the best threshold, the confusion matrix associated.
        """
        thresholds = np.linspace(self.scores.min(), self.scores.max(), num_trials)
        metrics = np.vectorize(get_metric_func)(thresholds)
        best_metric_ind = np.argmax(metrics)
        return (
            metrics[best_metric_ind],
            thresholds[best_metric_ind],
            self.get_confusion_matrix(thresholds[best_metric_ind]),
        )

    @_register
    def get_auroc(self) -> float:
        """Get AUROC metric.

        Returns:
            float: AUROC.
        """
        return float(
            metrics.roc_auc_score(
                self.labels,
                -self.scores,
            )
        )

    @_register
    def get_auroc_10p(self) -> float:
        """Get AUROC metric.

        Returns:
            float: AUROC.
        """
        return float(
            metrics.roc_auc_score(
                self.labels,
                -self.scores,
                max_fpr=0.1,
            )
        )

    @_register
    def get_aupr(self) -> float:
        """Get AUPR metric.

        Returns:
            float: AUPR.
        """
        return float(
            metrics.average_precision_score(
                self.labels,
                -self.scores,
            )
        )

    @_register
    def get_audet(self) -> float:
        """Get AUDET metric.

        Returns:
            float: AUDET.
        """
        return float(np.trapz(self.fnr2, x=np.flip(self.fpr2)))

    def get_fpr_at_tp(self, true_positive_p: float) -> float:
        """See :func:`~adw.metrics.fpr_p`."""
        return fpr_p(self.tpr, self.fpr, true_positive_p)

    def get_tpr_at_fp(self, false_positive_p: float) -> float:
        """See :func:`~adw.metrics.recall_p`."""
        return recall_p(self.tpr, self.fpr, false_positive_p)

    def get_confusion_matrix(self, threshold: float) -> np.ndarray:
        """Get the confusion matrix given a threshold.

        By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
        is equal to the number of observations known to be in group :math:`i` and
        predicted to be in group :math:`j`.

        Thus in binary classification, the count of true negatives is
        :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
        :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

        Here negative means anomaly and positive means ok.

        Args:
            threshold (float): Threshold

        Returns:
            np.ndarray: Confusion matrix.
        """
        labels_pred = np.where(
            self.scores < threshold, self.true_label, self.anomaly_label
        )
        return metrics.confusion_matrix(self.labels, labels_pred)

    def get_accuracy(self, threshold: float) -> float:
        """Get accuracy.

        Args:
            threshold (float): threshold.

        Returns:
            float: accuracy.
        """
        labels_pred = np.where(
            self.scores < threshold, self.true_label, self.anomaly_label
        )
        return float(metrics.accuracy_score(self.labels, labels_pred))

    def get_precision(self, threshold: float) -> float:
        """Get precision.

        Args:
            threshold (float): threshold.

        Returns:
            float: precision.
        """
        labels_pred = np.where(
            self.scores < threshold, self.true_label, self.anomaly_label
        )
        return float(metrics.precision_score(self.labels, labels_pred))

    def get_recall(self, threshold: float) -> float:
        """Get recall.

        Args:
            threshold (float): threshold.

        Returns:
            float: recall.
        """
        labels_pred = np.where(
            self.scores < threshold, self.true_label, self.anomaly_label
        )
        return float(metrics.recall_score(self.labels, labels_pred))

    def get_f1_score(self, threshold: float) -> float:
        """Get F1-score.

        Args:
            threshold (float): threshold.

        Returns:
            float: F1-score.
        """
        labels_pred = np.where(
            self.scores < threshold, self.true_label, self.anomaly_label
        )
        return float(metrics.f1_score(self.labels, labels_pred))

    def get_matthews_correlation_coef(self, threshold: float) -> float:
        """Get Matthews correlation coefficient.

        Args:
            threshold (float): threshold.

        Returns:
            float: Matthews correlation coefficient.
        """
        (tp, fp), (fn, tn) = self.get_confusion_matrix(threshold).tolist()
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if abs(denom) < 1e-6:
            c = np.where(
                np.array([tp + fp, tp + fn, tn + fp, tn + fn]) == 0, 1, 0
            ).sum()
            if c == 1:
                denom = 1
            else:
                denom = np.nan
        return (tp * tn - fp * fn) / denom

    def get_jaccard_index(self, threshold: float) -> float:
        """Get jaccard index.

        Args:
            threshold (float): threshold.

        Returns:
            float: Jaccard index.
        """
        (tp, fp), (fn, _) = self.get_confusion_matrix(threshold).tolist()
        return tp / (tp + fn + fp)

    @_register
    def get_best_accuracy(
        self, num_trials: int = 500
    ) -> tuple[float, float, np.ndarray]:
        return self._get_best_metric(self.get_accuracy, num_trials)

    @_register
    def get_best_f1_score(
        self, num_trials: int = 500
    ) -> tuple[float, float, np.ndarray]:
        return self._get_best_metric(self.get_f1_score, num_trials)

    @_register
    def get_best_matthews_correlation_coef(
        self, num_trials: int = 500
    ) -> tuple[float, float, np.ndarray]:
        return self._get_best_metric(self.get_matthews_correlation_coef, num_trials)

    @_register
    def get_best_jaccard_index(
        self, num_trials: int = 500
    ) -> tuple[float, float, np.ndarray]:
        return self._get_best_metric(self.get_jaccard_index, num_trials)

    @_register
    def get_best_precision(self) -> float:
        return self.get_precision(self.get_best_accuracy()[1])

    @_register
    def get_best_recall(self) -> float:
        return self.get_recall(self.get_best_accuracy()[1])

    def plot_roc_curve(self, savename: str = None):
        """Plot the ROC curve.

        Args:
            savename (str, optional): If not None, saves the plot using the path provided else only shows it. Defaults to None.
        """
        fig = plt.figure()
        plt.title("ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot(self.fpr, self.tpr)
        if savename is None:
            plt.show()
        else:
            os.makedirs(os.path.split(savename)[0], exist_ok=True)
            plt.savefig(savename)
            mlflow.log_figure(fig, savename)
        plt.close()

    def plot_det_curve(self, savename: str = None):
        """Plot the DET curve.

        Args:
            savename (str, optional): If not None, saves the plot using the path provided else only shows it. Defaults to None.
        """
        fig = plt.figure()
        plt.title("DET curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("False Negative Rate")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot(self.fpr2, self.fnr2)
        if savename is None:
            plt.show()
        else:
            os.makedirs(os.path.split(savename)[0], exist_ok=True)
            plt.savefig(savename)
            mlflow.log_figure(fig, savename)
        plt.close()

    def plot_distribution(self, savename: str = None):
        """Plot the distribution of scores and the best accuracy threshold.

        Args:
            savename (str, optional): If not None, saves the plot using the path provided else only shows it. Defaults to None.
        """
        threshold = self.get_best_accuracy()[1]
        fig = plt.figure()
        plt.xlabel("Scores")
        plt.ylabel(f"Labels (Anomaly : {self.anomaly_label})")
        plt.scatter(self.scores, self.labels)
        plt.plot([threshold, threshold], [0, 1])
        if savename is None:
            plt.show()
        else:
            mlflow.log_figure(fig, savename)
        plt.close()

    def get_available_metrics(self) -> dict[str, float]:
        """Get a dictionary with all available metrics (the registered ones).

        Returns:
            dict: Metrics dictionary.
        """
        metrics = {}
        for attr in dir(self):
            func = getattr(self, attr)
            if hasattr(func, "is_metric"):
                result = func()
                if isinstance(result, tuple):
                    result = result[0]
                metrics[attr[4:]] = result
        return metrics

    def mlflow_log(self, verbose: bool = False):
        """Log all the registred metrics into MlFlow.

        Args:
            verbose (bool, optional): _description_. Defaults to False.
        """
        metrics = self.get_available_metrics()
        mlflow.log_metrics(metrics)
        if verbose:
            print(metrics)

    def __str__(self) -> str:
        metrics = {
            "acc": {"name": "Accuracy", "func": self.get_best_accuracy},
            "f1": {"name": "F1 score", "func": self.get_best_f1_score},
            "matthews": {
                "name": "Matthews correlation coefficient",
                "func": self.get_best_matthews_correlation_coef,
            },
            "jaccard": {"name": "Jaccard index", "func": self.get_best_jaccard_index},
        }
        string = "----------\n"
        string += f"AUROC : {self.get_auroc():.3f}\n"
        string += f"AUROC 10% : {self.get_auroc_10p():.3f}\n"
        string += f"AUPR : {self.get_aupr():.3f}\n"
        string += f"AUDET : {self.get_audet():.3f}\n\n"

        for _, metric in metrics.items():
            best_acc, best_threshold, matrix = metric["func"]()
            string += f"{metric['name']} : {best_acc:.3f}\n"
            string += f"Threshold : {best_threshold:.3f}\n"
            string += f"Precision : {self.get_precision(best_threshold):.3f}\n"
            string += f"Recall : {self.get_recall(best_threshold):.3f}\n"
            string += f"Confusion matrix (maximazing {metric['name']}) :\n"
            string += f"|{matrix[0,0]}\t{matrix[0,1]}|\n"
            string += f"|{matrix[1,0]}\t{matrix[1,1]}|\n\n"

        #string += "|NOK\tNOD|\n"
        #string += "|FA\tOK|\n\n"
        string += "----------"
        return string

    def save_to_file(self, savename: str, mlflow_log_path: str = None):
        np.savez(
            savename,
            anomaly_score=self.anomaly_score,
            anomaly_labels=self.anomaly_labels,
            allow_pickle=False,
        )
        if mlflow_log_path is not None:
            mlflow.log_artifact(savename, mlflow_log_path)

    def load_from_file(self, savename: str, fraction: float = 1.0):
        with np.load(savename, allow_pickle=False) as data:
            self.__init__(data["anomaly_score"], data["anomaly_labels"], fraction)


def get_scoring_from_file(savename: str, fraction: float = 1.0) -> Scoring:
    with np.load(savename, allow_pickle=False) as data:
        return Scoring(data["anomaly_score"], data["anomaly_labels"], fraction)
