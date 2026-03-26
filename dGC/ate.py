from .utils import (
    clip_probs,
    doubly_robust_scores,
    mean,
    standard_error,
    to_1d_float,
)


def tau_hat_from_gnn(
    gnn_outputs: dict,
    data: dict,
    outcome_key: str = "Y",
    treatment_key: str = "T",
    clip: float = 1e-3,
    return_details: bool = False,
):
    """
    Estimate ATE tau_hat from GNN nuisance predictions and observed data.

    Required keys in gnn_outputs:
    - "mu1": E[Y | T=1, X, A] predictions
    - "mu0": E[Y | T=0, X, A] predictions

    Optional key in gnn_outputs:
    - "p" or "propensity": P(T=1 | X, A) predictions.
      If present, uses DR-AIPW. Otherwise uses plug-in mean(mu1 - mu0).

    Required keys in data:
    - outcome_key (default "Y")
    - treatment_key (default "T"), with fallback to "D" when "T" missing.
    """
    y = to_1d_float(data[outcome_key])
    if treatment_key in data:
        t = to_1d_float(data[treatment_key])
    elif "D" in data:
        t = to_1d_float(data["D"])
    else:
        raise KeyError(f"Data must include '{treatment_key}' or 'D'.")

    mu1 = to_1d_float(gnn_outputs["mu1"])
    mu0 = to_1d_float(gnn_outputs["mu0"])

    n = len(y)
    for name, arr in (("T", t), ("mu1", mu1), ("mu0", mu0)):
        if len(arr) != n:
            raise ValueError(f"{name} length does not match Y length.")

    if "p" in gnn_outputs or "propensity" in gnn_outputs:
        p_raw = gnn_outputs["p"] if "p" in gnn_outputs else gnn_outputs["propensity"]
        p = clip_probs(to_1d_float(p_raw), clip=clip)
        if len(p) != n:
            raise ValueError("p length does not match Y length.")
        psi = doubly_robust_scores(y, t, mu1, mu0, p)
        tau_hat = mean(psi)
        if return_details:
            return {
                "tau_hat": tau_hat,
                "se": standard_error(psi),
                "estimator": "dr_aipw",
                "n": int(n),
            }
        return tau_hat

    tau_hat = mean([m1 - m0 for m1, m0 in zip(mu1, mu0)])
    if return_details:
        return {
            "tau_hat": tau_hat,
            "se": None,
            "estimator": "plugin",
            "n": int(n),
        }
    return tau_hat
