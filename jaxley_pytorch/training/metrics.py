# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""How close a student's activity and parameters are to its teacher's (the paper's
Fig. 1c-e), the shuffled-neuron baselines these errors are judged against, and the
dimensionality of a network's activity. Rates are (..., n): every sample (a solver step
of a trial, or a token's window) of every neuron."""

from typing import Dict

import torch

from jaxley_pytorch.training.rate_network import RateNetwork


def corr_columns(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of every column of x with the same column of y (nan if one is
    constant)."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    return (x * y).sum(0) / torch.sqrt((x * x).sum(0) * (y * y).sum(0))


@torch.no_grad()
def activity_stats(a: torch.Tensor, basis: torch.Tensor) -> Dict:
    """Dimensionality of rates `a` (..., n): participation ratio, components for 90% and
    95% of the variance, and the fraction of the variance in the readout's directions
    (`basis`, orthonormal (n, rank))."""
    x = a.reshape(-1, a.shape[-1]).double()
    x = x - x.mean(0)
    cov = x.T @ x / (x.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)
    frac = torch.cumsum(ev, 0) / ev.sum()
    return {
        "participation_ratio": float(ev.sum() ** 2 / (ev**2).sum()),
        "components_90": int((frac < 0.9).sum()) + 1,
        "components_95": int((frac < 0.95).sum()) + 1,
        "readout_rank": int(basis.shape[1]),
        "readout_variance_fraction": float(
            torch.trace(basis.T @ cov @ basis) / ev.sum()
        ),
        "mean_rate": float(a.mean()),
    }


@torch.no_grad()
def baselines(
    a_t: torch.Tensor,
    teacher: RateNetwork,
    basis: torch.Tensor,
    n_perm: int = 100,
    seed: int = 0,
) -> Dict:
    """The paper's shuffled baselines: the errors of the teacher's own rates and
    parameters against a random permutation of its neurons (mean over `n_perm`
    permutations), with the keys of `compare` + "_shuffled"."""
    gen = torch.Generator().manual_seed(seed)
    n = a_t.shape[-1]
    x = a_t.reshape(-1, n).double()
    g, b = torch.relu(teacher.g).detach().double(), teacher.b.detach().double()
    keys = ["act_mse", "act_mse_readout", "act_corr_err", "gain_rmse", "bias_rmse"]
    acc = {k: 0.0 for k in keys}
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=gen).to(x.device)
        e = x[:, perm] - x
        acc["act_mse"] += float((e**2).mean())
        acc["act_mse_readout"] += float(((e @ basis) ** 2).sum(-1).mean() / n)
        acc["act_corr_err"] += float(
            1 - torch.nanmean(corr_columns(x[:, perm], x).abs())
        )
        acc["gain_rmse"] += float(torch.sqrt(((g[perm] - g) ** 2).mean()))
        acc["bias_rmse"] += float(torch.sqrt(((b[perm] - b) ** 2).mean()))
    out = {f"{k}_shuffled": v / n_perm for k, v in acc.items()}
    out["act_mse_other_shuffled"] = (
        out["act_mse_shuffled"] - out["act_mse_readout_shuffled"]
    )
    return out


@torch.no_grad()
def compare(
    a_s: torch.Tensor,
    a_t: torch.Tensor,
    student: RateNetwork,
    teacher: RateNetwork,
    basis: torch.Tensor,
) -> Dict:
    """Errors of the student against the teacher: rates (mean squared error, split into
    the readout's directions and the rest; 1 - mean |Pearson r| over neurons), gains
    (RMS of relu(g)) and biases (RMS)."""
    n = a_t.shape[-1]
    xs, xt = a_s.reshape(-1, n).double(), a_t.reshape(-1, n).double()
    e = xs - xt
    gain = torch.relu(student.g) - torch.relu(teacher.g)
    out = {
        "act_mse": float((e**2).mean()),
        "act_mse_readout": float(((e @ basis) ** 2).sum(-1).mean() / n),
        "act_corr_err": float(1 - torch.nanmean(corr_columns(xs, xt).abs())),
        "gain_rmse": float(torch.sqrt((gain.double() ** 2).mean())),
        "bias_rmse": float(torch.sqrt(((student.b - teacher.b).double() ** 2).mean())),
    }
    out["act_mse_other"] = out["act_mse"] - out["act_mse_readout"]
    return out
