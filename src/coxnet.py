"""
CoxNet: Cox proportional-hazards model with an elastic-net penalty.

Minimises the same objective as glmnet's Cox family and scikit-survival's
CoxnetSurvivalAnalysis:

    -(1/n) * log partial likelihood(beta)
        + alpha * ( l1_ratio * ||beta||_1 + (1 - l1_ratio) / 2 * ||beta||_2^2 )

with Breslow's handling of tied event times. The features are standardised
inside the estimator. Fitted by accelerated proximal gradient (FISTA) along a
decreasing path of alphas with warm starts; alpha and l1_ratio are chosen by
cross-validation over patients on Harrell's C-index.

`CoxPH` is the same model without penalty. Both are direct survival models on patients, used as baselines for the pairwise
approach in 08_mwfas_ranking_test_set.ipynb.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.func_utils import concordance_index


# =============================================================================
# Partial likelihood
# =============================================================================
def _risk_set_index(time):
    """
    Order of the patients by decreasing time, and for each position in that order the index
    of the last patient with the same time (so that cumulative sums cover the whole risk set
    {j : t_j >= t_i}, ties included, as in Breslow's approximation).
    """
    order = np.argsort(-time, kind="stable")
    t_sorted = time[order]
    # last position of each block of equal times in the decreasing order
    is_last = np.r_[t_sorted[1:] != t_sorted[:-1], True]
    # for each position, the first "last of its block" position at or after it
    last_pos = np.minimum.accumulate(np.where(is_last, np.arange(len(t_sorted)), len(t_sorted))[::-1])[::-1]
    return order, last_pos


def cox_loss_and_grad(beta, X, time, event, order=None, last_pos=None):
    """Negative log partial likelihood divided by n (Breslow ties), and its gradient."""
    if order is None:
        order, last_pos = _risk_set_index(time)
    n = len(time)
    Xs, ds = X[order], event[order].astype(bool)
    eta = Xs @ beta
    shift = eta.max()
    w = np.exp(eta - shift)
    S0 = np.cumsum(w)[last_pos]                       # sum of w over the risk set
    S1 = np.cumsum(w[:, None] * Xs, axis=0)[last_pos]  # sum of w * x over the risk set
    loss = -(eta[ds] - shift - np.log(S0[ds])).sum() / n
    grad = -(Xs[ds] - S1[ds] / S0[ds, None]).sum(axis=0) / n
    return loss, grad


def _soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def _fit_one_alpha(X, time, event, alpha, l1_ratio, beta0, order, last_pos,
                   max_iter=2000, tol=1e-7):
    """FISTA with backtracking for one value of alpha."""
    l1 = alpha * l1_ratio
    l2 = alpha * (1 - l1_ratio)

    def smooth(b):
        loss, grad = cox_loss_and_grad(b, X, time, event, order, last_pos)
        return loss + 0.5 * l2 * b @ b, grad + l2 * b

    beta = beta0.copy()
    y, t_k, step = beta.copy(), 1.0, 1.0
    f_y, g_y = smooth(y)
    for _ in range(max_iter):
        while True:  # backtracking line search
            candidate = _soft_threshold(y - step * g_y, step * l1)
            f_c, _ = smooth(candidate)
            diff = candidate - y
            if f_c <= f_y + g_y @ diff + diff @ diff / (2 * step) + 1e-12:
                break
            step *= 0.5
        t_next = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        y = candidate + (t_k - 1) / t_next * (candidate - beta)
        converged = np.max(np.abs(candidate - beta)) < tol * max(1.0, np.max(np.abs(beta)))
        beta, t_k = candidate, t_next
        if converged:
            break
        f_y, g_y = smooth(y)
    return beta


# =============================================================================
# Estimator
# =============================================================================
class CoxNet:
    """
    Elastic-net Cox model with standardisation and cross-validated penalty.

    :param l1_ratios: candidate mixing parameters (1 = lasso, close to 0 = ridge)
    :param n_alphas: length of the alpha path
    :param alpha_min_ratio: smallest alpha as a fraction of the largest one
    :param n_splits: folds for the cross-validation over patients
    """

    def __init__(self, l1_ratios=(0.1, 0.5, 0.9), n_alphas=40, alpha_min_ratio=1e-3,
                 n_splits=5, random_state=42):
        self.l1_ratios = l1_ratios
        self.n_alphas = n_alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.n_splits = n_splits
        self.random_state = random_state

    # -- helpers --------------------------------------------------------------
    def _alpha_path(self, X, time, event, l1_ratio):
        _, grad0 = cox_loss_and_grad(np.zeros(X.shape[1]), X, time, event)
        alpha_max = np.max(np.abs(grad0)) / max(l1_ratio, 1e-3)
        return np.geomspace(alpha_max, alpha_max * self.alpha_min_ratio, self.n_alphas)

    @staticmethod
    def _path(X, time, event, alphas, l1_ratio):
        order, last_pos = _risk_set_index(time)
        beta = np.zeros(X.shape[1])
        coefs = []
        for a in alphas:
            beta = _fit_one_alpha(X, time, event, a, l1_ratio, beta, order, last_pos)
            coefs.append(beta.copy())
        return np.array(coefs)

    # -- API ------------------------------------------------------------------
    def fit(self, X, time, event):
        X = np.asarray(X, dtype=float)
        time = np.asarray(time, dtype=float)
        event = np.asarray(event).astype(int)
        self.mean_, self.scale_ = X.mean(axis=0), X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = (X - self.mean_) / self.scale_

        folds = list(StratifiedKFold(self.n_splits, shuffle=True,
                                     random_state=self.random_state).split(Z, event))
        self.cv_results_ = []
        best = (-np.inf, None, None)
        for l1_ratio in self.l1_ratios:
            alphas = self._alpha_path(Z, time, event, l1_ratio)
            scores = np.zeros((len(folds), len(alphas)))
            for k, (tr, va) in enumerate(folds):
                coefs = self._path(Z[tr], time[tr], event[tr], alphas, l1_ratio)
                for a_idx, b in enumerate(coefs):
                    scores[k, a_idx] = concordance_index(time[va], event[va], Z[va] @ b)
            mean = scores.mean(axis=0)
            for a, s in zip(alphas, mean):
                self.cv_results_.append({"l1_ratio": l1_ratio, "alpha": a, "cv_c_index": s})
            i = int(np.argmax(mean))
            if mean[i] > best[0]:
                best = (mean[i], l1_ratio, alphas[: i + 1])

        self.cv_score_, self.l1_ratio_, path = best
        self.alpha_ = path[-1]
        # refit on all patients along the path up to the chosen alpha (warm starts)
        self.coef_ = self._path(Z, time, event, path, self.l1_ratio_)[-1]
        return self

    def predict(self, X):
        """Linear risk score (higher = higher hazard = earlier predicted death)."""
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return Z @ self.coef_


class CoxPH:
    """
    Unpenalised Cox proportional-hazards model (Breslow ties), with standardised features.
    Fitted by L-BFGS on the partial likelihood. Baseline for CoxNet: same model, no penalty.
    """

    def fit(self, X, time, event):
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=float)
        time = np.asarray(time, dtype=float)
        event = np.asarray(event).astype(int)
        self.mean_, self.scale_ = X.mean(axis=0), X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = (X - self.mean_) / self.scale_
        order, last_pos = _risk_set_index(time)
        res = minimize(lambda b: cox_loss_and_grad(b, Z, time, event, order, last_pos),
                       np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B",
                       options={"gtol": 1e-9, "maxiter": 5000})
        if not res.success:
            raise RuntimeError(f"Cox model did not converge: {res.message}")
        self.coef_ = res.x
        return self

    def predict(self, X):
        """Linear risk score (higher = higher hazard = earlier predicted death)."""
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return Z @ self.coef_
