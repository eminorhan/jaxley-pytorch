"""Paper Fig. 3: a hybrid (statistical/biophysical) model of the calcium responses of a
mouse retinal ganglion cell (RGC).

Port of `nex/fig4_rgc/{train.py, simulate.py, utils/*.py}` of
https://github.com/mackelab/jaxley_experiments. Data: the recordings of Ran et al. (2020),
preprocessed by the paper (Zenodo 14166049, `fig4_results.zip`, `results/data/`): the
outputs of 110 OFF bipolar cells (BCs, a linear-nonlinear model of 1,024 noise images
for each of 15 scan fields) and the calcium responses of 147 regions of interest (ROIs).
Morphology: RGC 20161028_1 (`tests/swc_files/morph_retina_20161028_1.swc`).

Model: the RGC with Fohlmeister channels receives, at 287 synapses, step currents
(200 ms) proportional to the BC outputs; the intracellular calcium at every ROI,
convolved with a calcium-indicator kernel, is the prediction. Trained (as in the paper):
the 287 synaptic weights, the axial resistivity and radius of every basal branch, and
six maximal conductances each in the basal dendrites and the soma; SGD with momentum
0.5, learning rate 0.01 (divided by 10 every 350 iterations) on gradients normalized
per parameter group (/ norm**0.99 * size) and clipped at 10, batches of 256 images, the
200 ms simulated in four pieces with truncated gradients. Splits as in the paper
(numpy seed 0): 9,216 train, 3,072 validation, 3,072 test images.

Modes:
  evaluate  The paper's trained parameters (`--epoch`, from the paper's run
            `2024_08_01__20_57_21`): mean correlation over ROIs per split (paper's log:
            train 0.252, val 0.255, test 0.247 at the best validation epoch).
  train     The paper's training run (`--iterations 1000`).

The paper's Jaxley version read SWC files with the old reader (`--legacy 1`, default);
all capacitances are 1 uF/cm^2, so the old capacitance handling plays no role.

Usage: python fig3_retina.py --out <dir> --mode evaluate|train
"""
import argparse
import json
import math
import os
import sys
import time
from itertools import chain

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jaxley_pytorch as jx
from jaxley_pytorch.mech.channels.fohlmeister97 import Ca, CaNernstReversal, CaPump, K, KA, KCa, Leak, Na
from jaxley_pytorch.optimize.transforms import SigmoidTransform
from legacy_jaxley import read_swc_legacy

HERE = os.path.dirname(os.path.abspath(__file__))
SWC = os.path.join(HERE, "..", "..", "tests", "swc_files", "morph_retina_20161028_1.swc")
REFERENCE = "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/data/zenodo_14166049/fig4_results.zip"
RUN = "results/train_runs/2024_08_01__20_57_21/0"
CELL = "20161028_1"
REC_IDS = list(range(1, 16))
N_IMAGES = 1024  # per scan field
DT, T_MAX, PIECES, WARMUP, I_AMP = 0.025, 200.0, 4, 5.0, 0.1
CFG = {"lr": 0.01, "momentum": 0.5, "reduce_lr_every": 350, "reduce_lr_by": 10.0, "beta": 0.99,
       "soma_radius": 5.0, "output_scale": 60.0, "batchsize": 256, "val_frac": 0.25, "test_num": 3072}
BOUNDS = {
    "w_bc_to_rgc": (0.0, 0.2), "axial_resistivity": (100.0, 10_000.0), "radius": (0.1, 1.0),
}
BASAL_BOUNDS = {"Na_gNa": (0.0, 0.5), "K_gK": (0.01, 0.1), "Leak_gLeak": (1e-5, 1e-3),
                "KA_gKA": (10e-3, 100e-3), "Ca_gCa": (2e-3, 3e-3), "KCa_gKCa": (0.02e-3, 0.2e-3)}
SOMATIC_BOUNDS = dict(BASAL_BOUNDS, Na_gNa=(0.05, 0.5))
INIT_CONDS = {"Na_gNa": 0.05, "K_gK": 0.05, "Leak_gLeak": 1e-4, "KA_gKA": 36e-3, "Ca_gCa": 2.2e-3, "KCa_gKCa": 0.05e-3}


def reference(member, pandas=True):
    sys.path.insert(0, "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools")
    from safe_pickle import load_from_zip

    return load_from_zip(REFERENCE, member, allow_pandas=pandas)


def load_data():
    """Port of `read_data` and `build_training_data` (paper code), without the noise
    images (only used for receptive fields)."""
    get = lambda name: reference(f"results/data/{name}")
    stimuli_meta = get(f"stimuli_meta_{CELL}.pkl")
    labels_df = get(f"labels_lowpass_{CELL}.pkl")
    bc_output = get(f"off_bc_output_{CELL}.pkl")
    stimuli_meta = stimuli_meta[stimuli_meta["cell_id"] == CELL]
    bc_output = bc_output[(bc_output["cell_id"] == CELL) & bc_output["rec_id"].isin(REC_IDS)].copy()
    bc_output["activity"] = np.stack(bc_output["activity"].to_numpy())[:, :N_IMAGES].tolist()
    concatenated = bc_output.groupby("bc_id", sort=False)["activity"].apply(lambda x: list(chain(*list(x))))
    bc_output = bc_output[bc_output["rec_id"] == REC_IDS[0]].copy()
    bc_output["activity"] = list(concatenated.to_numpy())
    stimuli = stimuli_meta.join(bc_output.set_index("bc_id"), on="bc_id", how="left", rsuffix="_bc")
    # The paper averaged the ROIs within a compartment (`build_avg_recordings`).
    avg = reference("results/03_results/avg_recordings.pkl")
    per_field = list(avg.groupby("rec_id").size())

    bc_activity = np.stack(stimuli["activity"].to_numpy()).T
    currents = I_AMP * bc_activity / stimuli["num_synapses_of_bc"].to_numpy()
    n_rec = len(avg)
    labels = np.zeros((n_rec, len(REC_IDS) * N_IMAGES))
    weights = np.zeros_like(labels)
    cum = np.cumsum([0] + per_field)
    for i, rec_id in enumerate(REC_IDS):
        weights[cum[i] : cum[i + 1], i * N_IMAGES : (i + 1) * N_IMAGES] = 1.0
        in_field = avg[avg["rec_id"] == rec_id]
        labels[cum[i] : cum[i + 1], i * N_IMAGES : (i + 1) * N_IMAGES] = np.stack(in_field["ca"].to_numpy())[:, :N_IMAGES]
    return stimuli, avg, currents, labels.T, weights.T


def splits(n):
    """Port of `build_dataloaders` (paper code); the paper seeds numpy with 0 first."""
    np.random.seed(0)
    test_frac = CFG["test_num"] / n
    val_frac = CFG["val_frac"] * (1 - test_frac)
    num_train = int(n * (1 - val_frac - test_frac))
    inds = np.random.permutation(np.arange(n))
    return {"test": inds[: CFG["test_num"]], "train": inds[CFG["test_num"] : CFG["test_num"] + num_train],
            "val": inds[CFG["test_num"] + num_train :]}


def build_kernel(n):
    rise, decay = 5.0, 100.0
    k_rise = 1 - np.exp(np.log(0.5) / rise * np.arange(0, 5 * rise, DT))
    k_decay = np.max(k_rise) * np.exp(np.log(0.5) / decay * np.arange(0, 3 * decay, DT))
    return np.concatenate([k_rise, k_decay])[:n]


class Model:
    def __init__(self, stimuli, avg, device, legacy, compile):
        if legacy:
            cell = read_swc_legacy(SWC, nseg=4, max_branch_len=300.0, min_radius=1.0, assign_groups=True)
        else:
            cell = jx.read_swc(SWC, ncomp=4, max_branch_len=300.0, min_radius=1.0, assign_groups=True)
        cell.set("axial_resistivity", 5_000.0)
        for channel in [Na, K, Leak, KA, CaNernstReversal, CaPump, Ca, KCa]:
            cell.insert(channel())
        cell.soma.set("Na_gNa", 0.15)
        cell.basal.set("Na_gNa", 0.05)
        cell.set("CaPump_taur", 100.0)
        cell.soma.set("radius", CFG["soma_radius"])
        cell.soma.set("length", 2 * CFG["soma_radius"] / 4)
        cell.basal.set("radius", 0.2)
        cell.set("v", -69.0)
        cell.init_states()
        assert stimuli["branch_ind"].max() < cell.total_nbranches and avg["branch_ind"].max() < cell.total_nbranches
        # Several ROIs fall into the same compartment; Jaxley v0.14 records every
        # compartment once (the paper's version recorded it once per ROI), so the ROIs
        # are mapped onto the recordings.
        roi_nodes = []
        for _, rec in avg.iterrows():
            view = cell.branch(int(rec["branch_ind"])).loc(float(rec["comp"]))
            view.record("Cai", verbose=False)
            roi_nodes.append(int(view.nodes.index[0]))
        rec_nodes = cell.recordings["rec_index"].tolist()
        self.roi_to_rec = torch.as_tensor([rec_nodes.index(n) for n in roi_nodes], device=device)
        # Compartment of every synapse (set once; the currents are passed as data).
        self.stim_inds = np.asarray([
            cell.branch(int(b)).loc(float(c)).nodes.index[0]
            for b, c in zip(stimuli["branch_ind"], stimuli["comp"])
        ])
        self.basal_inds = np.unique(cell.basal.nodes["global_branch_index"]).tolist()
        self.somatic_inds = np.unique(cell.soma.nodes["global_branch_index"]).tolist()
        with torch.no_grad():
            _, self.init_states = jx.integrate(cell, t_max=WARMUP, return_states=True)
        self.init_states = {k: v.to(device) for k, v in self.init_states.items()}
        cell.delete_trainables()
        cell.basal.branch("all").make_trainable("axial_resistivity", verbose=False)
        cell.basal.branch("all").make_trainable("radius", verbose=False)
        self.cell, self.device, self.compile = cell, device, compile
        steps = int(T_MAX / PIECES / DT) + 2
        self.ckpt = [int(math.ceil(math.sqrt(steps)))] * 2
        self.kernel = torch.as_tensor(build_kernel(int(T_MAX / DT) + 2), device=device)
        self.basal_view = cell.branch(self.basal_inds)
        self.somatic_view = cell.branch(self.somatic_inds)

    def predict(self, params, basal, somatic, currents, checkpointing=True):
        """params: {w_bc_to_rgc, axial_resistivity, radius}; basal, somatic: {key: value};
        currents: (batch, 287). Returns the calcium prediction (batch, 147)."""
        input_currents = params["w_bc_to_rgc"] * currents
        piece = T_MAX / PIECES
        step = jx.datapoint_to_step_currents(0.0, piece, input_currents, DT, piece)
        stimuli = (["i"], [step], [self.stim_inds])
        pstate = None
        for key, value in basal.items():
            pstate = self.basal_view.data_set(key, value, pstate)
        for key, value in somatic.items():
            pstate = self.somatic_view.data_set(key, value, pstate)
        cell_params = [{"axial_resistivity": params["axial_resistivity"]}, {"radius": params["radius"]}]
        # The prediction is the mean over time of the flipped kernel times the calcium
        # trace (of length `len(kernel)`); it is accumulated piece by piece so that the
        # traces need not be kept.
        kernel = torch.flip(self.kernel, [0])
        states, start, total = self.init_states, 0, 0.0
        for _ in range(PIECES):
            states = {k: v.detach() for k, v in states.items()}  # Truncated gradients.
            v, states = jx.integrate(self.cell, params=cell_params, param_state=pstate, data_stimuli=stimuli,
                                     checkpoint_lengths=self.ckpt if checkpointing else None,
                                     all_states=states, return_states=True, compile=self.compile)
            v = v[..., 1:]
            stop = min(start + v.shape[-1], len(kernel))
            total = total + torch.sum(kernel[start:stop] * v[..., self.roi_to_rec, : stop - start], -1)
            start += v.shape[-1]
        return total / len(kernel) * CFG["output_scale"] - 1.3


def correlations(pred, labels, weights):
    rhos = []
    for roi in range(labels.shape[1]):
        m = weights[:, roi].astype(bool)
        rhos.append(np.corrcoef(labels[m, roi], pred[m, roi])[0, 1])
    return float(np.nanmean(rhos)), rhos


class Transforms:
    def __init__(self, device):
        self.params = {k: SigmoidTransform(*b) for k, b in BOUNDS.items()}
        self.basal = {k: SigmoidTransform(*b) for k, b in BASAL_BOUNDS.items()}
        self.somatic = {k: SigmoidTransform(*b) for k, b in SOMATIC_BOUNDS.items()}

    def forward(self, opt):
        return [{k: t.forward(opt[i][k]) for k, t in group.items()} for i, group in enumerate([self.params, self.basal, self.somatic])]


def evaluate_all(model, transforms, opt, data, inds, batch=256):
    params, basal, somatic = transforms.forward(opt)
    currents, labels, weights = data
    out = {}
    with torch.no_grad():
        for split in ["val", "test", "train"]:
            preds = [model.predict(params, basal, somatic, torch.as_tensor(currents[c], device=model.device), checkpointing=False).cpu().numpy()
                     for c in np.array_split(inds[split], max(1, len(inds[split]) // batch))]
            out[split] = correlations(np.concatenate(preds), labels[inds[split]], weights[inds[split]])[0]
    return out


def paper_opt_params(epoch, device):
    opt, basal, somatic = reference(f"{RUN}/opt_params/params_{epoch}.pkl", pandas=False)
    t = lambda x: torch.as_tensor(np.asarray(x), device=device)
    return [{k: t(v) for d in opt for k, v in d.items()}, {k: t(v) for d in basal for k, v in d.items()},
            {k: t(v) for d in somatic for k, v in d.items()}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["evaluate", "train"], required=True)
    parser.add_argument("--epoch", type=int, default=22, help="evaluate: the paper's parameters at the start of this epoch")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--legacy", type=int, default=1)
    parser.add_argument("--compile", type=int, default=1)
    parser.add_argument("--graph", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)  # As in the paper (jax_enable_x64).
    device = torch.device("cuda")
    stimuli, avg, currents, labels, weights = load_data()
    inds = splits(len(currents))
    print(f"currents {currents.shape}, labels {labels.shape}, splits " + ", ".join(f"{k} {len(v)}" for k, v in inds.items()), flush=True)
    model = Model(stimuli, avg, device, bool(args.legacy), bool(args.compile))
    print(f"cell: {model.cell.total_nbranches} branches, {len(model.basal_inds)} basal, {len(model.cell.recordings)} recorded compartments for {len(model.roi_to_rec)} ROIs", flush=True)
    transforms = Transforms(device)
    data = (currents, labels, weights)

    if args.mode == "evaluate":
        t0 = time.time()
        rhos = evaluate_all(model, transforms, paper_opt_params(args.epoch, device), data, inds)
        rhos["time_s"] = time.time() - t0
        print(f"paper's parameters (start of epoch {args.epoch}): {rhos}", flush=True)
        with open(os.path.join(args.out, f"evaluate_paper_epoch{args.epoch}.json"), "w") as f:
            json.dump(rhos, f, indent=1)
        return

    # Initial parameters (paper: `np.random.seed(1)` for the weights).
    np.random.seed(1)
    w0 = 0.1 * np.random.rand(currents.shape[1])
    p0 = model.cell.get_parameters()
    init = [{"w_bc_to_rgc": torch.as_tensor(w0, device=device),
             "axial_resistivity": p0[0]["axial_resistivity"].to(device), "radius": p0[1]["radius"].to(device)},
            {k: torch.tensor(v, device=device) for k, v in INIT_CONDS.items()},
            {k: torch.tensor(v, device=device) for k, v in INIT_CONDS.items()}]
    opt = [{k: transforms_group[k].inverse(v).clone() for k, v in g.items()}
           for g, transforms_group in zip(init, [transforms.params, transforms.basal, transforms.somatic])]
    leaves = [v for g in opt for v in g.values()]
    momentum = [torch.zeros_like(v) for v in leaves]
    lr = torch.tensor(CFG["lr"], device=device)
    static = {"currents": torch.zeros(CFG["batchsize"], currents.shape[1], device=device),
              "labels": torch.zeros(CFG["batchsize"], labels.shape[1], device=device),
              "weights": torch.zeros(CFG["batchsize"], labels.shape[1], device=device),
              "loss": torch.zeros((), device=device)}

    def iteration():
        """Loss, gradients and the paper's update, in place."""
        leaves_rg = [v.detach().requires_grad_(True) for v in leaves]
        it = iter(leaves_rg)
        opt_rg = [{k: next(it) for k in g} for g in opt]
        params, basal, somatic = transforms.forward(opt_rg)
        pred = model.predict(params, basal, somatic, static["currents"])
        loss = torch.mean(torch.sum(static["weights"] * torch.abs(pred - static["labels"]), -1))
        grads = torch.autograd.grad(loss, leaves_rg)
        g_it = iter(grads)
        grouped = [[next(g_it) for _ in g] for g in opt]
        beta = CFG["beta"]
        normalized = [g / torch.sqrt(torch.sum(g**2)) ** beta * g.numel() for g in grouped[0]]
        for group in grouped[1:]:
            norm = torch.sqrt(sum(torch.sum(g**2) for g in group))
            normalized += [g / norm**beta * len(group) for g in group]
        n0, n1 = len(grouped[0]), len(grouped[0]) + len(grouped[1])
        with torch.no_grad():
            for i, (p, m, g) in enumerate(zip(leaves, momentum, normalized)):
                if i < n1:  # The somatic optimizer of the paper does not clip.
                    g = torch.clamp(g, -10.0, 10.0)
                m.mul_(CFG["momentum"]).add_(g)
                p.sub_(lr * m)
        static["loss"].copy_(loss.detach())

    rng = np.random.default_rng(0)
    graph, side = None, torch.cuda.Stream() if args.graph else None
    log, best = {"loss": [], "rho": []}, {"val": -1.0}
    n_batches = len(inds["train"]) // CFG["batchsize"]
    t_start, step = time.time(), 0
    epochs = math.ceil(args.iterations / n_batches)
    for epoch in range(epochs):
        order = rng.permutation(inds["train"])
        epoch_loss = 0.0
        for b in range(n_batches):
            if step >= args.iterations:
                break
            batch = order[b * CFG["batchsize"] : (b + 1) * CFG["batchsize"]]
            for k, x in zip(["currents", "labels", "weights"], data):
                static[k].copy_(torch.as_tensor(x[batch], device=device))
            lr.fill_(CFG["lr"] / CFG["reduce_lr_by"] ** (step // CFG["reduce_lr_every"]))
            if graph is not None:
                graph.replay()
            elif side is not None and step == 0:
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    iteration()
                torch.cuda.current_stream().wait_stream(side)
            elif side is not None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    iteration()
                graph.replay()
            else:
                iteration()
            loss = static["loss"].item()
            epoch_loss += loss
            log["loss"].append(loss)
            step += 1
            if step % 10 == 0:
                print(f"iteration {step}: loss {loss:.3f} ({(time.time() - t_start) / step:.1f} s/iteration)", flush=True)
        rhos = evaluate_all(model, transforms, opt, data, inds)
        log["rho"].append(rhos)
        if rhos["val"] > best["val"]:
            best = dict(rhos, epoch=epoch)
        print(f"epoch {epoch}: loss {epoch_loss:.2f}, rho {rhos}, best {best}", flush=True)
        with open(os.path.join(args.out, "train.json"), "w") as f:
            json.dump({"log": log, "best": best, "time_s": time.time() - t_start}, f, indent=1)
        torch.save([{k: v.cpu() for k, v in g.items()} for g in opt], os.path.join(args.out, "opt_params.pt"))
        if step >= args.iterations:
            break


if __name__ == "__main__":
    main()
