# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""How close a student's activity and parameters are to its teacher's (the paper's
Fig. 1c-e), the shuffled-neuron baselines these errors are judged against, and the
dimensionality of a network's activity. The activity is (..., n): every sample (a solver
step of a trial, or a token's window) of every neuron (rates, or the features a spiking
network's task reads). The parameters are a network's single-neuron parameters
(`neuron_params()`: a rate network's gains and biases, a spiking network's log
conductances and biases)."""

from typing import Dict

import torch


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
    teacher: torch.nn.Module,
    basis: torch.Tensor,
    n_perm: int = 100,
    seed: int = 0,
) -> Dict:
    """The paper's shuffled baselines: the errors of the teacher's own activity and
    parameters against a random permutation of its neurons (mean over `n_perm`
    permutations; a parameter that only some neurons have is permuted among those),
    with the keys of `compare` + "_shuffled"."""
    gen = torch.Generator().manual_seed(seed)
    n = a_t.shape[-1]
    x = a_t.reshape(-1, n).double()
    params = {k: v.detach().double() for k, v in teacher.neuron_params().items()}
    keys = ["act_mse", "act_mse_readout", "act_corr_err"]
    keys += [f"{name}_rmse" for name in params]
    acc = {k: 0.0 for k in keys}
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=gen).to(x.device)
        e = x[:, perm] - x
        acc["act_mse"] += float((e**2).mean())
        acc["act_mse_readout"] += float(((e @ basis) ** 2).sum(-1).mean() / n)
        acc["act_corr_err"] += float(
            1 - torch.nanmean(corr_columns(x[:, perm], x).abs())
        )
        for name, p in params.items():
            if len(p) != n:
                perm_p = torch.randperm(len(p), generator=gen).to(p.device)
            else:
                perm_p = perm
            acc[f"{name}_rmse"] += float(torch.sqrt(((p[perm_p] - p) ** 2).mean()))
    out = {f"{k}_shuffled": v / n_perm for k, v in acc.items()}
    out["act_mse_other_shuffled"] = (
        out["act_mse_shuffled"] - out["act_mse_readout_shuffled"]
    )
    return out


@torch.no_grad()
def compare(
    a_s: torch.Tensor,
    a_t: torch.Tensor,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    basis: torch.Tensor,
) -> Dict:
    """Errors of the student against the teacher: activity (mean squared error, split
    into the readout's directions and the rest; 1 - mean |Pearson r| over neurons) and
    every single-neuron parameter (RMS, "<name>_rmse": a rate network's "gain_rmse" of
    relu(g) and "bias_rmse")."""
    n = a_t.shape[-1]
    xs, xt = a_s.reshape(-1, n).double(), a_t.reshape(-1, n).double()
    e = xs - xt
    out = {
        "act_mse": float((e**2).mean()),
        "act_mse_readout": float(((e @ basis) ** 2).sum(-1).mean() / n),
        "act_corr_err": float(1 - torch.nanmean(corr_columns(xs, xt).abs())),
    }
    s_params = student.neuron_params()
    for name, t in teacher.neuron_params().items():
        out[f"{name}_rmse"] = float(
            torch.sqrt(((s_params[name] - t).double() ** 2).mean())
        )
    out["act_mse_other"] = out["act_mse"] - out["act_mse_readout"]
    return out


@torch.no_grad()
def spike_compare(c_s: torch.Tensor, c_t: torch.Tensor) -> Dict:
    """A spiking student's spike counts per token (..., n) against its teacher's: mean
    squared difference and 1 - mean |Pearson r| over neurons (silent neurons left out)."""
    n = c_t.shape[-1]
    xs, xt = c_s.reshape(-1, n).double(), c_t.reshape(-1, n).double()
    return {
        "spike_mse": float(((xs - xt) ** 2).mean()),
        "spike_corr_err": float(1 - torch.nanmean(corr_columns(xs, xt).abs())),
    }


@torch.no_grad()
def spike_baselines(c_t: torch.Tensor, n_perm: int = 100, seed: int = 0) -> Dict:
    """`spike_compare` of the teacher's own spike counts against a random permutation
    of its neurons (mean over `n_perm` permutations), keys + "_shuffled"."""
    gen = torch.Generator().manual_seed(seed)
    n = c_t.shape[-1]
    x = c_t.reshape(-1, n).double()
    acc = {"spike_mse": 0.0, "spike_corr_err": 0.0}
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=gen).to(x.device)
        for k, v in spike_compare(x[:, perm], x).items():
            acc[k] += v
    return {f"{k}_shuffled": v / n_perm for k, v in acc.items()}
