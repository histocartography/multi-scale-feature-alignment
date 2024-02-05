from typing import Any, Callable, Dict, List
import numpy as np
import sklearn.metrics as metrics
from dataclasses import dataclass


@dataclass
class Metric:
    name: str
    fun: Callable
    probs: bool
    kwargs: Dict[str, Any]

    def call(
        self, y_true: np.ndarray, y_probs: np.ndarray, binary: bool = False
    ) -> float:
        kwargs = self.kwargs.copy()
        # if f1 or kappa and it is not binary -> get the prediction as argmax
        y = y_probs if binary else np.argmax(y_probs, axis=1).astype(np.int32)
        if binary:
            # binary case -> get pos-class only (if probs, ndim is 2)
            if y.ndim == 2:
                y = y[:, 1]
            kwargs.pop("average", None)
            kwargs.pop("multi_class", None)
        return self.fun(y_true, y, **kwargs)


class Metrics:
    METRICS = [
        Metric(
            name="f1", fun=metrics.f1_score, probs=False, kwargs={"average": "weighted"}
        ),
        Metric(
            name="kappa",
            fun=metrics.cohen_kappa_score,
            probs=False,
            kwargs={"weights": "quadratic"},
        ),
    ]

    def __init__(self, metrics: List[str]) -> None:
        self.metrics: List[Metric] = list(
            filter(lambda x: x.name in metrics, __class__.METRICS)
        )

    def get_scores(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        scores = {}
        for metric in self.metrics:
            scores[metric.name] = metric.call(y_true.astype(int), y_probs)
            for j in np.unique(y_true.astype(int)):
                true_cls = (y_true == j).astype(np.int_)
                probs_cls = (np.argmax(y_probs, axis=1) == j).astype(np.int_)
                scores[f"{metric.name} cl{j}"] = metric.call(true_cls, probs_cls, True)
        if not scores:
            print("No validation score??")
        return scores
