"""Paper Fig. 2e-i: fit 1,390 parameters of an L5PC to voltages recorded in every branch,
and infer their posterior with Hamiltonian Monte Carlo.

Port of `nex/fig3_l5pc/notebooks/02_voltage_imaging.ipynb` and
`02_voltage_imaging_mcmc.py` of https://github.com/mackelab/jaxley_experiments. The
L5PC of Fig. 2a-c gets one maximal conductance per branch for the three active apical
channels and the seven active axonal channels (plus 9 somatic/axonal pump and channel
parameters): 1,390 parameters. Ground truth: the paper's Gaussian-process conductance
profiles (as a function of the distance from the soma, loaded from Zenodo 14166049,
`fig3_results.zip`; resampling them with numpy is not reproducible across LAPACK builds).
Observation: the voltage in the first compartment of every branch (339) in response to a
3 ms, 3 nA somatic step current, from 1 to 5 ms.

Modes:
  fit  Gradient descent as in the paper: initial values 2 * N(0, 1) in the unconstrained
       space (numpy seed 1), SGD with momentum 0.9 and learning rate 0.1 on the gradient
       divided by its norm**0.8, loss = mean absolute error + 0.001 * sum over branches of
       (g_b - g_parent(b))**2 (in the unconstrained space), stopping at loss < 0.5 or after
       500 iterations.
  hmc  200 HMC chains (5 leapfrog steps): 2,000 steps (step size 0.1) on the prior
       (uniform within the bounds, with the regularizer), then 500 steps (step size 0.01)
       on the posterior with likelihood exp(-loss / 0.001).
  plot Panels e-i.

`--legacy 1` (default) simulates as the Jaxley version of the paper did (see
`legacy_jaxley.py`).

Usage: python fig2_l5pc_voltage_imaging.py --out <dir> --mode fit|hmc|plot
"""
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jaxley_pytorch as jx
from jaxley_pytorch.optimize.transforms import SigmoidTransform
from fig2_l5pc_synthetic import PARAMS, SWC, build_cell
from legacy_jaxley import legacy_distance, use_legacy_axial_conductances

REFERENCE = "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/data/zenodo_14166049/fig3_results.zip"
DT, T_MAX = 0.025, 5.0
CHECKPOINT_LENGTHS = [101, 2]  # As in the paper's HMC script.
APICAL = ["apical_NaTs2T_gNaTs2T", "apical_SKv3_1_gSKv3_1", "apical_M_gM"]
SOMA = [
    "somatic_NaTs2T_gNaTs2T",
    "somatic_SKv3_1_gSKv3_1",
    "somatic_SKE2_gSKE2",
    "somatic_CaPump_gamma",
    "somatic_CaPump_decay",
    "somatic_CaHVA_gCaHVA",
    "somatic_CaLVA_gCaLVA",
]
AXONAL = [
    "axonal_NaTaT_gNaTaT",
    "axonal_KPst_gKPst",
    "axonal_KTst_gKTst",
    "axonal_SKE2_gSKE2",
    "axonal_SKv3_1_gSKv3_1",
    "axonal_CaHVA_gCaHVA",
    "axonal_CaLVA_gCaLVA",
]
AXONAL_PUMP = ["axonal_CaPump_gamma", "axonal_CaPump_decay"]
BOUNDS = {key: (lo, hi) for (_, key), (lo, hi, _) in PARAMS.items()}
GT = {key: gt for (_, key), (_, _, gt) in PARAMS.items()}


def reference(name):
    sys.path.insert(0, "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools")
    from safe_pickle import load_from_zip

    return load_from_zip(REFERENCE, f"results/03_voltage_imaging/{name}.pkl")


def branch_indices(cell, group):
    return np.unique(getattr(cell, group).nodes["global_branch_index"]).tolist()


def build(args, device):
    """The cell with the ground-truth profiles, stimulus and recordings, its trainable
    parameters (paper order), and the distance of every branch from the soma."""
    cell = build_cell(args.swc, device, "legacy" if args.legacy else "graph")
    apical, axonal = branch_indices(cell, "apical"), branch_indices(cell, "axon")
    soma = branch_indices(cell, "soma")

    def distance(b):
        if args.legacy:
            return legacy_distance(cell, (b, 0), (0, 0), nseg=4)
        from jaxley_pytorch.morphology import distance_direct

        return float(distance_direct(cell.branch(0).loc(0.0), cell.branch(b).loc(0.0))[0])

    evals = np.asarray(reference("profile_evals"))
    profiles = {**reference("apical_profile"), **reference("axonal_profile")}
    profiles = {k: np.asarray(v) for k, v in profiles.items()}
    distances = {b: distance(b) for b in range(cell.total_nbranches)}
    for b in apical:
        for key in APICAL:
            cell.branch(b).set(key, np.interp(distances[b], evals, profiles[key]) * BOUNDS[key][1])
    for key in SOMA:
        cell.soma.set(key, GT[key])
    for b in axonal:
        for key in AXONAL:
            cell.branch(b).set(key, np.interp(distances[b], evals, profiles[key]) * BOUNDS[key][1])
    # As in the paper's notebook, the axonal conductances are then reset to the uniform
    # values of Fig. 2a-c (so only the apical ground truth varies along the tree).
    for key in AXONAL + AXONAL_PUMP:
        cell.axon.set(key, GT[key])

    cell.delete_trainables()
    for key in APICAL:
        cell.branch(apical).make_trainable(key, verbose=False)
    for key in SOMA:
        cell.soma.make_trainable(key, verbose=False)
    for key in AXONAL:
        cell.branch(axonal).make_trainable(key, verbose=False)
    for key in AXONAL_PUMP:
        cell.axon.make_trainable(key, verbose=False)

    cell.delete_stimuli()
    cell.delete_recordings()
    amp = torch.tensor(3.0, device=device)
    cell.branch(soma[0]).loc(0.0).stimulate(jx.step_current(0.0, 3.0, amp, DT, T_MAX), verbose=False)
    cell.branch("all").comp(0).record(verbose=False)
    cell.set("v", -70.0)
    cell.init_states()
    return cell, apical, axonal, distances


def regularizer_pairs(cell, inds):
    """(child, parent) positions within `inds` of branches whose parent is also in `inds`."""
    inds = np.asarray(inds)
    parents = np.asarray(cell.comb_parents)[inds]
    has = np.isin(parents, inds)
    child = np.arange(len(inds))[has]
    parent = np.asarray([np.where(inds == p)[0][0] for p in parents[has]])
    return child, parent


class Problem:
    """Loss, regularizer and transforms, for parameters as a list of dicts (paper order)
    or as flat vectors of shape (batch, 1390)."""

    def __init__(self, cell, apical, axonal, x_o, compile):
        self.cell = cell
        self.compile = compile
        params = cell.get_parameters()
        self.keys = [next(iter(p)) for p in params]
        self.sizes = [p[k].shape[-1] for p, k in zip(params, self.keys)]
        assert sum(self.sizes) == 1390, sum(self.sizes)
        self.transforms = [SigmoidTransform(*BOUNDS[k]) for k in self.keys]
        self.pairs = {
            "apical": regularizer_pairs(cell, apical),
            "axonal": regularizer_pairs(cell, axonal),
        }
        self.x_o = x_o

    def unflatten(self, flat):
        return [{k: x} for k, x in zip(self.keys, torch.split(flat, self.sizes, dim=-1))]

    def forward(self, opt_params):
        return [{k: t.forward(p[k])} for p, k, t in zip(opt_params, self.keys, self.transforms)]

    def log_det(self, flat):
        widths = torch.cat([torch.full((n,), t.width, dtype=flat.dtype, device=flat.device) for n, t in zip(self.sizes, self.transforms)])
        return torch.sum(torch.log(widths) + torch.nn.functional.logsigmoid(flat) + torch.nn.functional.logsigmoid(-flat), dim=-1)

    def regularizer(self, opt_params):
        reg = 0.0
        for p, k in zip(opt_params, self.keys):
            group = "apical" if k in APICAL else "axonal" if k in AXONAL else None
            if group is not None:
                child, parent = self.pairs[group]
                reg = reg + torch.sum((p[k][..., child] - p[k][..., parent]) ** 2, dim=-1)
        return reg

    def simulate(self, params, checkpoint_lengths=CHECKPOINT_LENGTHS):
        return jx.integrate(self.cell, params=params, checkpoint_lengths=checkpoint_lengths, compile=self.compile)

    def data_loss(self, v):
        return torch.mean(torch.abs(v[..., 40:200:5] - self.x_o[..., 40:200:5]), dim=(-1, -2))


def fit(args, device):
    cell, apical, axonal, distances = build(args, device)
    with torch.no_grad():
        x_o = jx.integrate(cell, compile=bool(args.compile))
    ref_x_o = np.asarray(reference("recording_xo"))
    n = min(ref_x_o.shape[-1], x_o.shape[-1])
    diff = np.abs(x_o.cpu().numpy()[:, :n] - ref_x_o[:, :n])
    print(f"x_o {tuple(x_o.shape)} vs paper {ref_x_o.shape}: max |dv| {diff.max():.3g} mV, mean {diff.mean():.3g} mV", flush=True)
    prob = Problem(cell, apical, axonal, x_o, bool(args.compile))

    np.random.seed(1)
    opt = [{k: torch.as_tensor(2.0 * np.random.randn(n_), device=device).requires_grad_(True)}
           for k, n_ in zip(prob.keys, prob.sizes)]
    leaves = [p[k] for p, k in zip(opt, prob.keys)]
    optimizer = torch.optim.SGD(leaves, lr=0.1, momentum=0.9)
    beta, losses, times = 0.8, [], []
    best_loss, best = float("inf"), None
    for epoch in range(args.epochs):
        t0 = time.time()
        optimizer.zero_grad()
        v = prob.simulate(prob.forward(opt))
        loss = prob.data_loss(v) + 0.001 * prob.regularizer(opt)
        loss.backward()
        loss_val = loss.item()
        times.append(time.time() - t0)
        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss, best = loss_val, [x.detach().clone() for x in leaves]
        if epoch % 10 == 0:
            print(f"epoch {epoch}: loss {loss_val:.4f} ({times[-1]:.2f} s)", flush=True)
        if loss_val < 0.5:
            break
        with torch.no_grad():
            norm = torch.sqrt(sum(torch.sum(x.grad**2) for x in leaves))
            for x in leaves:
                x.grad.div_(norm**beta)
        optimizer.step()
    print(f"best loss {best_loss:.4f} after {len(losses)} iterations, {np.sum(times):.0f} s")

    best_params = prob.forward([{k: x} for k, x in zip(prob.keys, best)])
    cell.delete_recordings()
    cell.record(verbose=False)
    with torch.no_grad():
        prediction = jx.integrate(cell, params=best_params).cpu().numpy()
        cell.delete_recordings()
        cell.record(verbose=False)
        observation_all = jx.integrate(cell).cpu().numpy()
    np.savez(
        os.path.join(args.out, "fit.npz"),
        losses=np.asarray(losses),
        x_o=x_o.cpu().numpy(),
        prediction=prediction,
        observation_all=observation_all,
        best_flat=torch.cat(best).cpu().numpy(),
        distances=np.asarray([distances[b] for b in range(cell.total_nbranches)]),
        apical=np.asarray(apical),
        axonal=np.asarray(axonal),
    )
    with open(os.path.join(args.out, "fit.json"), "w") as f:
        json.dump({"legacy": bool(args.legacy), "best_loss": best_loss, "iterations": len(losses),
                   "time_s": float(np.sum(times)), "x_o_max_abs_diff_vs_paper": float(diff.max()),
                   "paper_gd_iterations": len(reference("gd_losses"))}, f, indent=1)


def hmc(args, device):
    cell, apical, axonal, _ = build(args, device)
    with torch.no_grad():
        x_o = jx.integrate(cell, compile=bool(args.compile))
    prob = Problem(cell, apical, axonal, x_o, bool(args.compile))
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def prior_logp(flat):
        return -prob.regularizer(prob.unflatten(flat)) + prob.log_det(flat)

    def posterior_logp(flat):
        opt = prob.unflatten(flat)
        v = prob.simulate(prob.forward(opt))
        return -prob.data_loss(v) / args.likelihood_std - prob.regularizer(opt) + prob.log_det(flat)

    def value_and_grad(logp, x):
        x = x.detach().requires_grad_(True)
        lp = logp(x)
        (g,) = torch.autograd.grad(lp.sum(), x)
        return lp.detach(), g

    def run(logp, x, num_steps, step_size, label):
        lp, g = value_and_grad(logp, x)
        accepted = []
        for i in range(num_steps):
            t0 = time.time()
            p0 = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype)
            xn, pn = x, p0 + 0.5 * step_size * g
            for leap in range(args.leapfrog):
                xn = xn + step_size * pn
                lpn, gn = value_and_grad(logp, xn)
                pn = pn + (step_size if leap < args.leapfrog - 1 else 0.5 * step_size) * gn
            log_ratio = (lpn - 0.5 * torch.sum(pn**2, -1)) - (lp - 0.5 * torch.sum(p0**2, -1))
            u = torch.rand(log_ratio.shape, generator=generator, device=device, dtype=x.dtype)
            accept = torch.log(u) < torch.nan_to_num(log_ratio, nan=-torch.inf)
            x = torch.where(accept[:, None], xn, x)
            lp = torch.where(accept, lpn, lp)
            g = torch.where(accept[:, None], gn, g)
            accepted.append(accept.double().mean().item())
            if i % 25 == 0 or i == num_steps - 1:
                print(f"{label} step {i}: mean logp {lp.mean().item():.2f}, acceptance {np.mean(accepted[-25:]):.2f} "
                      f"({time.time() - t0:.2f} s/step)", flush=True)
        return x, accepted

    x0 = 2.0 * torch.randn(args.chains, sum(prob.sizes), generator=generator, device=device)
    t0 = time.time()
    prior, _ = run(prior_logp, x0, args.prior_steps, 0.1, "prior")
    posterior, accepted = run(posterior_logp, prior, args.steps, args.step_size, "posterior")
    elapsed = time.time() - t0
    params = torch.cat([p[k] for p, k in zip(prob.forward(prob.unflatten(posterior)), prob.keys)], -1)
    np.savez(os.path.join(args.out, "hmc.npz"), posterior_unconstrained=posterior.cpu().numpy(),
             posterior=params.cpu().numpy(), acceptance=np.asarray(accepted))
    with open(os.path.join(args.out, "hmc.json"), "w") as f:
        json.dump({"legacy": bool(args.legacy), "chains": args.chains, "steps": args.steps,
                   "mean_acceptance": float(np.mean(accepted)), "time_s": elapsed}, f, indent=1)
    print(f"HMC done in {elapsed:.0f} s, mean acceptance {np.mean(accepted):.2f}")


def plot(args):
    fit_ = np.load(os.path.join(args.out, "fit.npz"))
    apical, axonal, distances = fit_["apical"], fit_["axonal"], fit_["distances"]
    device = torch.device("cpu")
    cell, _, _, _ = build(args, device)
    keys = [next(iter(p)) for p in cell.get_parameters()]
    sizes = [p[k].shape[-1] for p, k in zip(cell.get_parameters(), keys)]
    offsets = dict(zip(keys, np.cumsum([0] + sizes[:-1])))

    def morph(ax, values, title, cmap="viridis", vmin=None, vmax=None):
        norm = plt.Normalize(vmin if vmin is not None else np.nanmin(values), vmax if vmax is not None else np.nanmax(values))
        for b, xyzr in enumerate(cell.xyzr):
            c = plt.get_cmap(cmap)(norm(values[b])) if np.isfinite(values[b]) else "0.85"
            ax.plot(xyzr[:, 0], xyzr[:, 1], c=c, lw=0.8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=8)

    fig, ax = plt.subplots(1, 7, figsize=(17, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.4, 1.4]})
    gt = np.full(cell.total_nbranches, np.nan)
    gt[apical] = cell.nodes.groupby("global_branch_index")["apical_NaTs2T_gNaTs2T"].first().to_numpy()[apical]
    morph(ax[0], gt, "e  apical gNaT (gt)", cmap="magma")
    obs = fit_["x_o"]
    for i, t in enumerate([1.5, 2.5]):
        k = int(round(t / DT))
        morph(ax[1 + i], obs[:, k], f"f  data, {t} ms", vmin=-80, vmax=40)
    rec_first = cell.nodes.groupby("global_branch_index").head(1).index.to_numpy()
    for i, t in enumerate([1.5, 2.5]):
        k = int(round(t / DT))
        morph(ax[3 + i], fit_["prediction"][rec_first, k], f"g  fit, {t} ms", vmin=-80, vmax=40)
    hmc_path = os.path.join(args.out, "hmc.npz")
    a = ax[5]
    key = "apical_NaTs2T_gNaTs2T"
    order = np.argsort(distances[apical])
    d = distances[apical][order]
    a.plot(d, gt[apical][order], c="k", label="gt")
    if os.path.exists(hmc_path):
        post = np.load(hmc_path)["posterior"][:, offsets[key] : offsets[key] + len(apical)]
        lo, hi = np.quantile(post, [0.05, 0.95], axis=0)
        a.fill_between(d, lo[order], hi[order], color="#9ecae1", alpha=0.7, label="posterior 90%")
    best = fit_["best_flat"][offsets[key] : offsets[key] + len(apical)]
    a.plot(d, (SigmoidTransform(*BOUNDS[key]).forward(torch.tensor(best))).numpy()[order], c="#08519c", lw=0.8, label="GD fit")
    a.set_xlabel("distance to soma (um)")
    a.set_ylabel("apical gNaT (S/cm2)")
    a.legend(fontsize=6)
    a.set_title("h  recovered profile", fontsize=8)
    ax[6].semilogy(fit_["losses"], c="#08519c", label="GD (ours)")
    try:
        ax[6].semilogy([float(x) for x in reference("gd_losses")], c="0.6", ls="--", label="GD (paper)")
        ga = reference("ga_losses")
        ga = np.asarray([float(x) for x in ga]) if np.ndim(ga[0]) == 0 else np.min(np.asarray(ga, dtype=float), axis=0)
        ax[6].semilogy(ga, c="k", label="GA (paper)")
    except Exception as e:  # noqa: BLE001
        print("reference losses not plotted:", e)
    ax[6].set_xlabel("iteration")
    ax[6].set_ylabel("loss")
    ax[6].legend(fontsize=6)
    ax[6].set_title("i  training", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig2_efghi.png"), dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["fit", "hmc", "plot"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--swc", default=SWC)
    parser.add_argument("--legacy", type=int, default=1, help="behave like the paper's Jaxley")
    parser.add_argument("--compile", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--chains", type=int, default=200)
    parser.add_argument("--prior-steps", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--leapfrog", type=int, default=5)
    parser.add_argument("--likelihood-std", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)  # As in the paper (jax_enable_x64).
    use_legacy_axial_conductances(bool(args.legacy))
    if args.mode == "fit":
        fit(args, torch.device(args.device))
    elif args.mode == "hmc":
        hmc(args, torch.device(args.device))
    else:
        plot(args)


if __name__ == "__main__":
    main()
