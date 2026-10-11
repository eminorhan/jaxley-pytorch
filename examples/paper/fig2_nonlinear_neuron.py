"""Paper Fig. 2j-m: a single neuron learns a nonlinear (XOR-like) pattern separation.

Port of `nex/fig3_l5pc/notebooks/04_nonlinear_single_neuron.ipynb` of
https://github.com/mackelab/jaxley_experiments. A cell with three branches (soma and
two dendrites) of four compartments each, with Hodgkin-Huxley channels, receives two
step currents (one per dendrite) whose amplitudes are the two coordinates of a data
point. The radius, length, axial resistivity and the three HH conductances of every
compartment are trained (72 parameters) such that the somatic voltage after 3 ms is
35 mV for one class and -70 mV for the other. The two classes are not linearly
separable.

As in the paper: Adam (lr 0.01), batch size 1, 200 epochs over 32 training points, 10
random seeds. The 10 seeds are trained simultaneously as a batch dimension (each seed
has its own parameters, data and data order; Adam is elementwise, so this is equivalent
to 10 separate runs).

Usage: python fig2_nonlinear_neuron.py --out <dir> [--device cuda] [--epochs 200]
           [--compile 1] [--graph 1]
"""
import argparse
import json
import os
import time
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import jaxley_pytorch as jx
from jaxley_pytorch.channels import HH
from jaxley_pytorch.optimize.transforms import SigmoidTransform

DT = 0.025
T_MAX = 5.95
I_DELAY, I_DUR, I_AMP = 1.0, 0.9, 0.05
N_TRAIN, N_TEST = 32, 16
LOWERS = {
    "radius": 0.1,
    "length": 1.0,
    "axial_resistivity": 500.0,
    "HH_gNa": 0.05,
    "HH_gK": 0.01,
    "HH_gLeak": 0.0001,
}
UPPERS = {
    "radius": 5.0,
    "length": 20.0,
    "axial_resistivity": 5500.0,
    "HH_gNa": 1.1,
    "HH_gK": 0.3,
    "HH_gLeak": 0.001,
}


def gen_data(n_data, switch_labels=0, std=0.1, center_x=2.0, center_y=2.0, spacing=1.0):
    """Three blobs; the outer two are one class, the middle one the other class."""
    num_blobs = 3
    x_locs = np.linspace(center_x - spacing, center_x + spacing, num_blobs)
    y_locs = np.linspace(center_y + spacing, center_y - spacing, num_blobs)
    blobs_pos, blobs_neg = [], []
    for blob in range(num_blobs):
        xs = np.random.randn(n_data) * std + x_locs[blob]
        ys = np.random.randn(n_data) * std + y_locs[blob]
        if not blob % 2:
            blobs_pos.append(np.stack([xs, ys]).T)
        else:
            blobs_neg.append(np.stack([xs, ys]).T)
    class0 = np.concatenate(blobs_pos)
    class1 = np.concatenate(blobs_neg)
    inds = np.random.permutation(np.arange(len(class0)))
    class0 = class0[inds][: n_data // 2]
    inds = np.random.permutation(np.arange(len(class1)))
    class1 = class1[inds][: n_data // 2]
    all_data = np.concatenate([class0, class1])
    if switch_labels:
        all_labels = np.concatenate([np.zeros(n_data // 2), np.ones(n_data // 2)])
    else:
        all_labels = np.concatenate([np.ones(n_data // 2), np.zeros(n_data // 2)])
    inds = np.random.permutation(np.arange(len(all_labels)))
    all_data, all_labels = all_data[inds], all_labels[inds]
    voltage_labels = np.where(all_labels == 1, 35.0, -70.0)
    return all_data, all_labels, voltage_labels


def build_cell():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(3)], parents=[-1, 0, 0])
    cell.insert(HH())
    cell.set("v", -70.0)
    cell.set("HH_m", 0.074901)
    cell.set("HH_h", 0.4889)
    cell.set("HH_n", 0.3644787)
    cell.branch(0).loc(0.0).record(verbose=False)
    return cell


def initial_params(cell, num_seeds):
    """Draw the initial values as the paper does (one value per key and seed)."""
    per_seed = []
    for seed in range(num_seeds):
        np.random.seed(1 + seed)
        per_seed.append(
            {k: np.random.rand() * (UPPERS[k] - LOWERS[k]) + LOWERS[k] for k in LOWERS}
        )
        # Advance the RNG exactly as the paper's notebook (data is drawn after the
        # cell is built, from the same RNG).
        per_seed[-1]["data"] = (
            gen_data(N_TRAIN, std=0.1, center_x=2.5, center_y=2.5, spacing=1.0),
            gen_data(N_TEST, std=0.1, center_x=2.5, center_y=2.5, spacing=1.0),
        )
    for key in LOWERS:
        cell.branch("all").comp("all").make_trainable(key, 1.0, verbose=False)
    params = []
    for p in cell.get_parameters():
        key = next(iter(p))
        vals = torch.as_tensor([s[key] for s in per_seed], dtype=torch.get_default_dtype())
        params.append({key: vals[:, None].expand(num_seeds, p[key].shape[-1]).clone()})
    return params, per_seed


def simulate(cell, params, images, compile):
    """images: (seeds, batch, 2). Returns the somatic voltage, (seeds, batch, time)."""
    currents = jx.datapoint_to_step_currents(I_DELAY, I_DUR, I_AMP * images, DT, T_MAX)
    data_stimuli = None
    for i, branch in enumerate([1, 2]):
        data_stimuli = cell.branch(branch).loc(1.0).data_stimulate(
            currents[..., i : i + 1, :], data_stimuli=data_stimuli
        )
    # Parameters have shape (seeds, n); add the batch dimension of the data points.
    params = [{k: v[:, None, :] for k, v in p.items()} for p in params]
    return jx.integrate(cell, params=params, data_stimuli=data_stimuli, compile=compile)[
        ..., 0, :
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--compile", type=int, default=1)
    parser.add_argument("--graph", type=int, default=1, help="CUDA graph of a training step")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)  # As in the paper (jax_enable_x64).
    args.graph = bool(args.graph) and device.type == "cuda"

    cell = build_cell()
    params, per_seed = initial_params(cell, args.seeds)
    keys = [next(iter(p)) for p in params]
    transform = jx.ParamTransform(
        [{k: SigmoidTransform(LOWERS[k], UPPERS[k])} for k in keys]
    )
    opt_params = [
        {k: v.to(device).requires_grad_(True) for k, v in p.items()}
        for p in transform.inverse(params)
    ]
    optimizer = torch.optim.Adam(
        [v for p in opt_params for v in p.values()], lr=0.01, capturable=args.graph
    )

    train_images = np.stack([s["data"][0][0] for s in per_seed])  # (seeds, 32, 2)
    train_voltages = np.stack([s["data"][0][2] for s in per_seed])
    test_images = np.stack([s["data"][1][0] for s in per_seed])
    test_voltages = np.stack([s["data"][1][2] for s in per_seed])
    to_t = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)

    rng = np.random.default_rng(0)
    seeds = torch.arange(args.seeds, device=device)
    train_images_t, train_voltages_t = to_t(train_images), to_t(train_voltages)
    # Static inputs/outputs of one training step (the CUDA graph reads and writes these).
    static_images = torch.zeros(args.seeds, 1, 2, device=device)
    static_labels = torch.zeros(args.seeds, 1, device=device)

    def train_step():
        optimizer.zero_grad(set_to_none=False)
        v = simulate(cell, transform.forward(opt_params), static_images, args.compile)
        loss_per_seed = torch.mean(torch.abs(v[..., 120] - static_labels), dim=-1)
        loss_per_seed.sum().backward()
        optimizer.step()
        return loss_per_seed.detach()

    # With --graph, the first steps run normally (warm-up, also on the side stream that
    # graph capture requires); then one step is captured (capture executes nothing) and
    # replayed for all remaining steps. Training is the same in both modes.
    num_warmup = 3
    graph = None
    side_stream = torch.cuda.Stream() if args.graph else None
    if side_stream is not None:
        side_stream.wait_stream(torch.cuda.current_stream())
    losses = torch.zeros(args.epochs, args.seeds, device=device)
    start = time.time()
    num_steps = 0
    for epoch in range(args.epochs):
        orders = np.stack([rng.permutation(N_TRAIN) for _ in range(args.seeds)])
        orders = torch.as_tensor(orders, device=device)
        for step in range(N_TRAIN):
            idx = orders[:, step]
            static_images.copy_(train_images_t[seeds, idx][:, None, :])
            static_labels.copy_(train_voltages_t[seeds, idx][:, None])
            if graph is not None:
                graph.replay()
            elif side_stream is not None and num_steps < num_warmup:
                with torch.cuda.stream(side_stream):
                    static_loss = train_step()
                torch.cuda.current_stream().wait_stream(side_stream)
            elif side_stream is not None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_loss = train_step()
                graph.replay()
            else:
                static_loss = train_step()
            losses[epoch] += static_loss
            num_steps += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch}: median epoch loss {losses[epoch].median().item():.2f}, "
                f"{(time.time() - start) / (epoch + 1):.2f} s/epoch",
                flush=True,
            )
    losses = losses.cpu().numpy()
    train_time = time.time() - start

    with torch.no_grad():
        final = transform.forward(opt_params)
        v_test = simulate(cell, final, to_t(test_images), args.compile)
        pred = v_test[..., 120].cpu().numpy()
        accuracy = np.mean((pred > -17.5) == (test_voltages > 0), axis=1)

        # Decision surface of the first seed.
        res = 80
        grid = np.stack(np.meshgrid(np.linspace(0, 5, res), np.linspace(0, 5, res)))
        xy = np.reshape(grid, (2, res**2)).T
        first = [{k: v[:1] for k, v in p.items()} for p in final]
        surface = simulate(cell, first, to_t(xy)[None], args.compile)[0, :, 120]
        surface = surface.cpu().numpy().reshape(res, res)

    results = {
        "train_time_s": train_time,
        "test_accuracy_per_seed": accuracy.tolist(),
        "final_epoch_loss_per_seed": losses[-1].tolist(),
        "num_parameters": int(sum(v.shape[-1] for p in params for v in p.values())),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez(
        os.path.join(args.out, "results.npz"),
        losses=losses,
        v_test=v_test.cpu().numpy(),
        test_images=test_images,
        test_voltages=test_voltages,
        train_images=train_images,
        train_voltages=train_voltages,
        surface=surface,
    )

    time_vec = np.arange(v_test.shape[-1]) * DT
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.6))
    for p, l in zip(train_images[0], train_voltages[0]):
        ax[0].scatter(p[0], p[1], c="k" if l > 0 else "#d7301f", s=4)
    ax[0].set_xlim([0, 5])
    ax[0].set_ylim([0, 5])
    ax[0].set_title("j  data (seed 0)")
    for trace, label in zip(v_test[0].cpu().numpy(), test_voltages[0]):
        ax[1].plot(time_vec, trace, c="k" if label > 0 else "#d7301f", lw=0.6)
    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("voltage (mV)")
    ax[1].set_title("k  learned traces")
    im = ax[2].imshow(surface, origin="lower", extent=[0, 5, 0, 5], vmin=-80, vmax=50)
    fig.colorbar(im, ax=ax[2], label="V (mV)")
    ax[2].set_title("l  decision surface")
    ax[3].semilogy(losses)
    ax[3].set_xlabel("epoch")
    ax[3].set_ylabel("epoch loss")
    ax[3].set_title("m  training (10 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig2_jklm.png"), dpi=150)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
