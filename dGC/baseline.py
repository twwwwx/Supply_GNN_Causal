DEFAULT_MODEL_SPECS = [
    {"model_name": "linear_v1", "intercept": 0.5, "beta": 1.2, "beta2": 0.0},
    {"model_name": "quadratic_v1", "intercept": 0.2, "beta": 1.0, "beta2": 0.15},
]

from .gen_data import sample_data
from .utils import (
    add_intercept,
    clip_probs,
    doubly_robust_scores,
    fit_ols,
    mean,
    mse,
    predict_linear,
    to_2d_float,
)


def fit_outcome_regressions(X, Y, D):
    """
    Fit two linear outcome regressions:
    mu1(x) from treated subsample and mu0(x) from control subsample.
    """
    x = add_intercept(to_2d_float(X))
    y = [float(v) for v in Y]
    d = [1 if float(v) > 0.5 else 0 for v in D]

    x_treat = [row for row, di in zip(x, d) if di == 1]
    y_treat = [yi for yi, di in zip(y, d) if di == 1]
    x_ctrl = [row for row, di in zip(x, d) if di == 0]
    y_ctrl = [yi for yi, di in zip(y, d) if di == 0]

    # Fallback to pooled fit if one arm is empty.
    if len(x_treat) == 0:
        beta_1 = fit_ols(x, y)
    else:
        beta_1 = fit_ols(x_treat, y_treat)

    if len(x_ctrl) == 0:
        beta_0 = fit_ols(x, y)
    else:
        beta_0 = fit_ols(x_ctrl, y_ctrl)

    return beta_1, beta_0


def fit_propensity_linear(X, D, clip=1e-3):
    """
    Fit a simple linear probability model for p(x)=P(D=1|X).
    """
    x = add_intercept(to_2d_float(X))
    d = [1.0 if float(v) > 0.5 else 0.0 for v in D]
    beta_p = fit_ols(x, d)
    p_hat = clip_probs(predict_linear(x, beta_p), clip=clip)
    return beta_p, p_hat

def estimate_tau_hat_dr(data, feature_key="X", outcome_key="Y", treatment_key="D", clip=1e-3):
    """
    End-to-end DR ATE estimator:
    1) Outcome regressions Y~X by treatment arm
    2) Propensity model D~X
    3) DR aggregation to tau_hat
    """
    X = data[feature_key]
    Y = data[outcome_key]
    D = data[treatment_key] if treatment_key in data else data["T"]

    x_with_intercept = add_intercept(to_2d_float(X))
    beta_1, beta_0 = fit_outcome_regressions(X, Y, D)
    _, p_hat = fit_propensity_linear(X, D, clip=clip)

    mu1_hat = predict_linear(x_with_intercept, beta_1)
    mu0_hat = predict_linear(x_with_intercept, beta_0)
    tau_hat = mean(doubly_robust_scores(Y, D, mu1_hat, mu0_hat, p_hat))

    return {
        "tau_hat": tau_hat,
        "mu1_hat": mu1_hat,
        "mu0_hat": mu0_hat,
        "p_hat": p_hat,
        "beta_outcome_treated": beta_1,
        "beta_outcome_control": beta_0,
    }


def report_tau_mse(
    num_replications=200,
    sample_size=500,
    seed=123,
    graph_model="rgg",
    true_tau=0.0,
    feature_key="X",
):
    """
    Run Monte Carlo draws and report MSE of tau_hat.
    """
    tau_hats = []
    for r in range(num_replications):
        draw = sample_data(
            sample_size=sample_size,
            seed=seed + r,
            graph_model=graph_model,
        )
        result = estimate_tau_hat_dr(draw, feature_key=feature_key)
        tau_hats.append(result["tau_hat"])

    return {
        "num_replications": int(num_replications),
        "sample_size": int(sample_size),
        "graph_model": graph_model,
        "true_tau": float(true_tau),
        "mean_tau_hat": mean(tau_hats),
        "mse_tau_hat": mse(tau_hats, true_tau),
    }
