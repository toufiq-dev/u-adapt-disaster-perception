"""Mode B — logistic-regression gate (PRIMARY Mode B claim).

A 6-parameter logistic gate (5 inputs + bias) is trained on a small held-out
calibration set (20 labeled boxes per class, disjoint from the k support
examples and the test set) using cached features and a frozen backbone.

Input vector (normalized to [0, 1], see docs/pre_registration.md):

    x = [S_tilde_text, S_tilde_visual, sigma_tilde^2_text,
         sigma_tilde^2_visual, a_tilde_visual]

Soft targets (per proposal Section 5.4.2):

    w* = 1                                    if visual top-1 correct & text not
         0                                    if text top-1 correct & visual not
         sigma(S_visual - S_text)             if both or neither are correct

The gate is trained with binary cross-entropy against these soft targets via
plain gradient descent in numpy (no sklearn/torch required). L2 weight decay
1e-4 is applied; 5-fold CV on the calibration set reports mean + std.

With 6 parameters and ~20 boxes/class the parameter-to-sample ratio is ~0.3,
keeping overfitting risk low (pre-registered overfitting mitigation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_L2 = 1e-4
DEFAULT_EPOCHS = 2000
DEFAULT_LR = 0.1
DEFAULT_N_FOLDS = 5


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def soft_targets(
    s_text: np.ndarray,
    s_visual: np.ndarray,
    text_correct: np.ndarray,
    visual_correct: np.ndarray,
) -> np.ndarray:
    """Pre-registered soft target w* (see module docstring)."""
    t = np.zeros_like(np.asarray(s_text, dtype=np.float64))
    both = text_correct & visual_correct
    neither = ~text_correct & ~visual_correct
    t[text_correct & ~visual_correct] = 1.0
    t[visual_correct & ~text_correct] = 0.0
    t[both | neither] = _sigmoid(
        np.asarray(s_visual, dtype=np.float64)[both | neither]
        - np.asarray(s_text, dtype=np.float64)[both | neither]
    )
    return t


@dataclass
class LogRegGate:
    """6-parameter logistic gate: w = sigmoid(x @ theta + b)."""

    l2: float = DEFAULT_L2
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    n_folds: int = DEFAULT_N_FOLDS
    _theta: Optional[np.ndarray] = None
    _bias: Optional[float] = None
    _cv_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y_soft: np.ndarray,
        verbose: bool = False,
    ) -> "LogRegGate":
        """Fit the gate on normalized features X (N, 5) with soft targets.

        Uses full-batch gradient descent with L2 regularization on the weight
        vector only (bias unregularized).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_soft, dtype=np.float64)
        n, d = X.shape
        theta = np.zeros(d)
        bias = 0.0
        for epoch in range(self.epochs):
            z = X @ theta + bias
            p = _sigmoid(z)
            grad_theta = X.T @ (p - y) / n + self.l2 * theta
            grad_bias = float(np.mean(p - y))
            theta -= self.lr * grad_theta
            bias -= self.lr * grad_bias
            if verbose and epoch % 200 == 0:
                loss = float(
                    np.mean(-(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
                )
                logger.info("epoch %d loss %.4f", epoch, loss)
        self._theta, self._bias = theta, bias
        return self

    def fit_cv(
        self, X: np.ndarray, y_soft: np.ndarray
    ) -> "LogRegGate":
        """5-fold CV on the calibration set; stores per-fold scores."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_soft, dtype=np.float64)
        n = len(X)
        perm = np.random.default_rng(0).permutation(n)
        X, y = X[perm], y[perm]
        scores: list[float] = []
        for fold in range(self.n_folds):
            test_idx = np.arange(fold, n, self.n_folds)
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            gate = LogRegGate(
                l2=self.l2, epochs=self.epochs, lr=self.lr, n_folds=self.n_folds
            ).fit(X[train_idx], y[train_idx])
            pred = gate.predict(X[test_idx])
            scores.append(float(np.mean((pred - y[test_idx]) ** 2)))
        self._cv_scores = np.asarray(scores)
        logger.info("LogReg 5-fold MSE: mean %.4f std %.4f", scores.mean(), scores.std())
        return self.fit(X, y)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gate weights w in (0, 1) for normalized features X (N, 5)."""
        if self._theta is None or self._bias is None:
            raise RuntimeError("LogRegGate must be fit() before predict()")
        X = np.asarray(X, dtype=np.float64)
        return _sigmoid(X @ self._theta + self._bias)

    @property
    def cv_scores(self) -> Optional[np.ndarray]:
        return self._cv_scores
