"""Source model: preprocessing, classifier, kNN reference set, OOD scorers and gate calibration.

Spaces:
- input space:   x~ = l2(x_titan). All shift estimates (mu_S, mu_S,c, delta) live here.
- features:      u = l2(x~ - delta). Classifier, kNN reference set, gate and detectors live in the features.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

KNN_K = 10
LOGREG_C = 1.0
GATE_QUANTILE = 0.01  # gate g: 1st percentile of the CPTAC-val nearest-neighbour similarity


def l2(z: np.ndarray) -> np.ndarray:
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def to_input_space(x_titan: np.ndarray) -> np.ndarray:
    return l2(np.asarray(x_titan, dtype=np.float64))


class SourceModel:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        """x_train: input-space source-train embeddings; y_train: ID classes 0..K-1."""
        self.classes = np.unique(y_train)
        if not np.array_equal(self.classes, np.arange(len(self.classes))):
            raise ValueError(f"classes must be 0..K-1, got {self.classes}")
        self.mu = x_train.mean(0)
        self.class_means = np.stack([x_train[y_train == c].mean(0) for c in self.classes])
        self.cov = np.cov(x_train.T)  # input-space covariance (used by the CORAL baseline)
        self.x_train = x_train        # input space (used by the matched-neighbour shift estimator)
        self.bank = self.features(x_train)
        self.bank_labels = np.asarray(y_train)
        self.clf = LogisticRegression(C=LOGREG_C, max_iter=10_000).fit(self.bank, self.bank_labels)
        if self.clf.n_iter_.max() >= self.clf.max_iter:
            raise RuntimeError("logistic regression did not converge")
        self.gate = None

    # ------------------------------------------------------------ representation
    def features(self, x: np.ndarray, delta: np.ndarray | None = None) -> np.ndarray:
        """u = l2(x~ - delta) for input-space x~."""
        return l2(x if delta is None else x - delta)

    def similarity(self, u: np.ndarray, cls: int | None = None) -> np.ndarray:
        """Cosine similarity to the reference set (optionally restricted to one source-train class)."""
        bank = self.bank if cls is None else self.bank[self.bank_labels == cls]
        return u @ bank.T

    @staticmethod
    def topk_mean(sim: np.ndarray, k: int) -> np.ndarray:
        k = min(k, sim.shape[1])
        return np.partition(sim, -k, axis=1)[:, -k:].mean(1)

    # ------------------------------------------------------------ prediction and scores (higher = more OOD)
    def predict(self, u: np.ndarray) -> np.ndarray:
        return self.clf.predict(u)

    def score(self, u: np.ndarray, scorer: str) -> np.ndarray:
        if scorer == "knn":  # Sun et al. (2022): Euclidean distance to the k-th nearest reference feature (unit vectors)
            kth = np.partition(self.similarity(u), -KNN_K, 1)[:, -KNN_K]
            return np.sqrt(np.maximum(2 - 2 * kth, 0))
        if scorer == "msp":
            return -self.clf.predict_proba(u).max(1)
        raise ValueError(scorer)

    # ------------------------------------------------------------ calibration on source val (ID only)
    def calibrate(self, x_val: np.ndarray) -> None:
        """Gate g of CHORD's candidate selection, calibrated on source-val ID slides."""
        u = self.features(x_val)
        self.gate = float(np.quantile(self.similarity(u).max(1), GATE_QUANTILE))
