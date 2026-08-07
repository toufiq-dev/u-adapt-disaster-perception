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

``w`` is the weight on the VISUAL score, so w* = 1 means "trust visual" and
w* = 0 means "trust text". The gate is trained with binary cross-entropy
against these soft targets via plain gradient descent in numpy (no
sklearn/torch required). L2 weight decay 1e-4 is applied; 5-fold CV on the
calibration set reports mean + std.

NOTE (2026-08-07): ``soft_targets`` previously had the two disagreement
branches SWAPPED (``w* = 1`` for text-only-correct), contradicting the
pre-registered formula; fixed to match proposal §5.4.2 and
``tests/test_mode_b_calibration.py``. Runs produced before this fix learned
the inverse directional mapping and should not be compared with post-fix
Mode B numbers.

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
    # Pre-registered formula (proposal §5.4.2): w* = 1 (trust visual) when
    # the VISUAL modality alone is correct; w* = 0 (trust text) when TEXT
    # alone is correct. Fixed 2026-08-07 — the two branches were previously
    # swapped (see module docstring).
    t[visual_correct & ~text_correct] = 1.0
    t[text_correct & ~visual_correct] = 0.0
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
    def set_params(self, theta, bias) -> "LogRegGate":
        """Inject pretrained weights (COCO/LVIS init ablation, former Mode C).

        The subsequent ``fit()`` warm-starts from these values instead of zero.
        """
        self._theta = np.asarray(theta, dtype=np.float64)
        self._bias = float(bias)
        return self

    def fit(
        self,
        X: np.ndarray,
        y_soft: np.ndarray,
        verbose: bool = False,
    ) -> "LogRegGate":
        """Fit the gate on normalized features X (N, 5) with soft targets.

        Uses full-batch gradient descent with L2 regularization on the weight
        vector only (bias unregularized). Warm-starts from ``set_params``
        weights when provided (COCO/LVIS-pretrained init ablation).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_soft, dtype=np.float64)
        n, d = X.shape
        if self._theta is not None and self._theta.shape[0] != d:
            raise ValueError(
                f"injected theta has {self._theta.shape[0]} entries but X has {d} features"
            )
        theta = self._theta.copy() if self._theta is not None else np.zeros(d)
        bias = self._bias if self._bias is not None else 0.0
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
        """5-fold CV on the calibration set; stores per-fold scores.

        Folds with an empty train or test partition are skipped (the
        pre-registered 5-fold design assumes >= 5 samples; on tiny pilot
        calibration sets, e.g. the n=100 pilot's 1-6 boxes, fewer folds are
        actually evaluated and ``cv_scores`` is None when none qualify).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_soft, dtype=np.float64)
        n = len(X)
        perm = np.random.default_rng(0).permutation(n)
        X, y = X[perm], y[perm]
        scores: list[float] = []
        for fold in range(self.n_folds):
            test_idx = np.arange(fold, n, self.n_folds)
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            gate = LogRegGate(
                l2=self.l2, epochs=self.epochs, lr=self.lr, n_folds=self.n_folds
            ).fit(X[train_idx], y[train_idx])
            pred = gate.predict(X[test_idx])
            scores.append(float(np.mean((pred - y[test_idx]) ** 2)))
        self._cv_scores = np.asarray(scores) if scores else None
        if scores:
            logger.info(
                "LogReg %d-fold MSE (of %d): mean %.4f std %.4f",
                len(scores), self.n_folds,
                float(np.mean(scores)), float(np.std(scores)),
            )
        else:
            logger.info(
                "LogReg %d-fold CV skipped: calibration set has only %d sample(s)",
                self.n_folds, n,
            )
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
