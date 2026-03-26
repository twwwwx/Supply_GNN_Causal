import numpy as np
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path

DEFAULT_MODEL_SPECS = [
    {"model_name": "linear_v1", "intercept": 0.5, "beta": 1.2, "beta2": 0.0},
    {"model_name": "quadratic_v1", "intercept": 0.2, "beta": 1.0, "beta2": 0.15},
]


from .utils import (
    clip_probs,
    doubly_robust_scores,
    to_2d_float,
)




def estimate_tau_hat_dr_linear(data, feature_key="X", outcome_key="Y", treatment_key="D", clip=1e-3):
    """
    End-to-end DR ATE estimator:
    1) Outcome regressions Y~X by treatment arm
    2) Propensity model D~X
    3) DR aggregation to tau_hat
    """
    X = data[feature_key]
    Y = data[outcome_key]
    D = data[treatment_key] if treatment_key in data else data["T"]

    x = np.asarray(to_2d_float(X), dtype=float)
    y = np.asarray([float(v) for v in Y], dtype=float)
    d = np.asarray([1 if float(v) > 0.5 else 0 for v in D], dtype=int)

    pooled_model = LinearRegression().fit(x, y)

    if np.any(d == 1):
        outcome_treated = LinearRegression().fit(x[d == 1], y[d == 1])
    else:
        outcome_treated = pooled_model

    if np.any(d == 0):
        outcome_control = LinearRegression().fit(x[d == 0], y[d == 0])
    else:
        outcome_control = pooled_model

    mu1_hat = outcome_treated.predict(x).astype(float).tolist()
    mu0_hat = outcome_control.predict(x).astype(float).tolist()

    propensity_model = LinearRegression().fit(x, d.astype(float))
    p_hat = clip_probs(propensity_model.predict(x).astype(float).tolist(), clip=clip)
    tau_hat = float(np.mean(np.asarray(doubly_robust_scores(Y, D, mu1_hat, mu0_hat, p_hat), dtype=float)))

    return {
        "tau_hat": tau_hat,
        "mu1_hat": mu1_hat,
        "mu0_hat": mu0_hat,
        "p_hat": p_hat,
        "beta_outcome_treated": [float(outcome_treated.intercept_)]
        + outcome_treated.coef_.astype(float).tolist(),
        "beta_outcome_control": [float(outcome_control.intercept_)]
        + outcome_control.coef_.astype(float).tolist(),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    true_tau = 2.0
    n = 500

    print(f"true_tau={true_tau:.6f}")
    for i in range(10):
        x = rng.uniform(0.0, 1.0, size=n)
        p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x)))
        d = rng.binomial(1, p, size=n)
        eps = rng.normal(0.0, 1.0, size=n)
        y = true_tau * d + 0.6 * x + eps

        fit = estimate_tau_hat_dr_linear(
            {"X": x.tolist(), "D": d.tolist(), "Y": y.tolist()},
            feature_key="X",
            treatment_key="D",
            outcome_key="Y",
        )
        print(f"pred_{i + 1}={fit['tau_hat']:.6f}")
