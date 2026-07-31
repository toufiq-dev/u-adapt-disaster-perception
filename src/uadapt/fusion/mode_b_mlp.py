"""Mode B — gating MLP (SECONDARY variant; logistic regression is primary).

Small MLP 5 -> 128 -> 1 (~650 parameters) trained on the calibration set
(20 boxes/class) with:

  * ReLU hidden activation, dropout p=0.3 during training only,
  * L2 weight decay 1e-4,
  * early stopping (patience 10) on a held-out validation fold
    (5 images per class from the calibration set),
  * 5-fold cross-validation on the calibration set (reports mean + std).

The gate predicts w in (0,1) via sigmoid; S_final = (1-w) S_text + w S_visual.
Implemented in numpy (no torch) so it runs anywhere and is unit-testable.

Pre-registered expectation: if this MLP fails to beat the logistic gate, that
finding is reported honestly (docs/pre_registration.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HIDDEN = 128
DEFAULT_DROPOUT_P = 0.3
DEFAULT_L2 = 1e-4
DEFAULT_LR = 0.01
DEFAULT_EPOCHS = 500
DEFAULT_PATIENCE = 10
DEFAULT_N_FOLDS = 5


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class MLPGate:
    """5 -> hidden -> 1 MLP gate trained with dropout + L2 + early stopping."""

    hidden_dim: int = DEFAULT_HIDDEN
    dropout_p: float = DEFAULT_DROPOUT_P
    l2: float = DEFAULT_L2
    lr: float = DEFAULT_LR
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    n_folds: int = DEFAULT_N_FOLDS
    rng_seed: int = 0

    _w1: Optional[np.ndarray] = None
    _b1: Optional[np.ndarray] = None
    _w2: Optional[np.ndarray] = None
    _b2: Optional[np.ndarray] = None
    _cv_scores: Optional[np.ndarray] = None
    _history: list = field(default_factory=list)

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y_soft: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "MLPGate":
        """Train with dropout + L2. If a validation split is given, apply
        early stopping (patience) and keep the best-validated parameters."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_soft, dtype=np.float64)
        rng = np.random.default_rng(self.rng_seed)
        d_in = X.shape[1]

        w1 = rng.normal(0.0, 0.1, (d_in, self.hidden_dim))
        b1 = np.zeros(self.hidden_dim)
        w2 = rng.normal(0.0, 0.1, (self.hidden_dim, 1))
        b2 = 0.0

        use_val = X_val is not None and y_val is not None
        best_val = np.inf
        best_params = None
        bad_epochs = 0

        for epoch in range(self.epochs):
            # Forward with dropout (training)
            mask = rng.binomial(1, 1.0 - self.dropout_p, size=(len(X), self.hidden_dim))
            z1 = X @ w1 + b1
            h1 = _relu(z1) * mask / (1.0 - self.dropout_p)
            z2 = h1 @ w2 + b2
            p = _sigmoid(z2).ravel()

            # Gradients
            dz2 = (p - y) / len(X)
            dw2 = h1.T @ dz2[:, None] + self.l2 * w2
            db2 = float(dz2.sum())
            dh1 = dz2[:, None] * w2.T * mask / (1.0 - self.dropout_p)
            dz1 = dh1 * (z1 > 0)
            dw1 = X.T @ dz1 + self.l2 * w1
            db1 = dz1.sum(axis=0)

            w1 -= self.lr * dw1
            b1 -= self.lr * db1
            w2 -= self.lr * dw2
            b2 -= self.lr * db2
            self._history.append(float(_bce(p, y)))

            if use_val:
                val_loss = float(_bce(self._forward(X_val, w1, b1, w2, b2), y_val))
                if val_loss < best_val:
                    best_val = val_loss
                    best_params = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        if verbose:
                            logger.info("early stop at epoch %d (val %.4f)", epoch, best_val)
                        break

        if use_val and best_params is not None:
            w1, b1, w2, b2 = best_params
        self._w1, self._b1, self._w2, self._b2 = w1, b1, w2, b2
        return self

    def fit_cv(self, X: np.ndarray, y_soft: np.ndarray) -> "MLPGate":
        """5-fold CV on the calibration set (mean + std MSE), then refit on
        the full calibration set."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_soft, dtype=np.float64)
        n = len(X)
        perm = np.random.default_rng(0).permutation(n)
        X, y = X[perm], y[perm]
        scores: list[float] = []
        for fold in range(self.n_folds):
            test_idx = np.arange(fold, n, self.n_folds)
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            gate = MLPGate(
                hidden_dim=self.hidden_dim,
                dropout_p=self.dropout_p,
                l2=self.l2,
                lr=self.lr,
                epochs=self.epochs,
                patience=self.patience,
                n_folds=self.n_folds,
            ).fit(X[train_idx], y[train_idx])
            pred = gate.predict(X[test_idx])
            scores.append(float(np.mean((pred - y[test_idx]) ** 2)))
        self._cv_scores = np.asarray(scores)
        logger.info("MLP 5-fold MSE: mean %.4f std %.4f", scores.mean(), scores.std())
        return self.fit(X, y)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._w1 is None:
            raise RuntimeError("MLPGate must be fit() before predict()")
        return self._forward(
            np.asarray(X, dtype=np.float64), self._w1, self._b1, self._w2, self._b2
        )

    def _forward(self, X, w1, b1, w2, b2) -> np.ndarray:
        return _sigmoid(_relu(X @ w1 + b1) @ w2 + b2).ravel()

    @property
    def cv_scores(self) -> Optional[np.ndarray]:
        return self._cv_scores


def _bce(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
