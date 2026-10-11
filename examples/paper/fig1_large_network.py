"""Paper Fig. 1f: simulate a network of 2,000 morphologically detailed neurons with
1 million synapses, and compute the gradient with respect to 3.2 million parameters.

Port of `nex/fig2_demonstration/notebooks/01_large_net.ipynb` of
https://github.com/mackelab/jaxley_experiments. 50 reconstructed morphologies (those of
the paper's MNIST network with fewer than 280 branches, 2 compartments per branch,
Zenodo 14166049 `fig6_morphologies.zip`) are repeated 40 times (plus one output neuron,
2,041 cells) and connected layer by layer (5 layers of 400 cells, fully connected plus
58% additional random synapses; ~1.01 million ionotropic synapses), with
Hodgkin-Huxley channels. The first layer gets step currents; 200 ms are simulated
(8,000 steps) with three levels of checkpointing. Paper (A100 GPU): simulation 21 s,
gradient with respect to all membrane and synaptic conductances 144 s.

Usage: python fig1_large_network.py --out <dir> [--size-factor 40] [--dtype float32]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jaxley_pytorch as jx
from jaxley_pytorch.channels import HH
from jaxley_pytorch.connect import fully_connect, sparse_connect
from jaxley_pytorch.synapses import IonotropicSynapse
from legacy_jaxley import read_swc_legacy

MORPHOLOGIES = "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/data/fig6_morphologies/morphologies"
I_DELAY, I_AMP, I_DUR, DT, T_MAX = 10.0, 0.1, 150.0, 0.025, 199.9


def build(size_factor):
    names = sorted(n for n in os.listdir(MORPHOLOGIES) if n.startswith("n"))
    cells = []
    for name in names:
        cell = read_swc_legacy(os.path.join(MORPHOLOGIES, name, f"{name}.CNG.swc"), nseg=2, max_branch_len=500.0)
        if cell.total_nbranches < 280:
            cells.append(cell)
    cells = cells[:50]
    cells = cells * size_factor + [cells[0]]
    net = jx.Network(cells)
    sf = size_factor
    np.random.seed(0)
    for layer in range(4):
        pre = net.cell(list(range(layer * 10 * sf, (layer + 1) * 10 * sf)))
        post = net.cell(list(range((layer + 1) * 10 * sf, (layer + 2) * 10 * sf)))
        fully_connect(pre, post, IonotropicSynapse(), random_post_comp=True)
        sparse_connect(pre, post, IonotropicSynapse(), 0.58, random_post_comp=True)
    fully_connect(net.cell(list(range(40 * sf, 50 * sf))), net.cell([50 * sf]), IonotropicSynapse(), random_post_comp=True)
    num_synapses = len(net.edges)
    net.set("IonotropicSynapse_gS", 0.00003 / sf * 20)
    net.IonotropicSynapse.edge(list(range(num_synapses - 10 * sf, num_synapses))).set("IonotropicSynapse_gS", 0.0001 / sf * 20)
    net.set("IonotropicSynapse_s", 0.0)
    for i in range(len(cells)):
        net.cell(i).branch(1).comp(0).record(verbose=False)
    net.insert(HH())
    net.set("v", -65.0)
    net.init_states()
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--size-factor", type=int, default=40)
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--compile", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(getattr(torch, args.dtype))
    device = torch.device("cuda")
    t0 = time.time()
    net = build(args.size_factor)
    build_s = time.time() - t0
    num_comps, num_syn = len(net.nodes), len(net.edges)
    num_cells = net.nodes["global_cell_index"].nunique()
    print(f"{num_cells} cells, {num_comps} compartments, {num_syn} synapses; built in {build_s:.0f} s", flush=True)
    sf = args.size_factor
    steps = (T_MAX + 0.05) / DT
    checkpoints = [int(np.ceil(steps ** (1 / 3)))] * 3

    def simulate(params, amplitudes):
        currents = jx.datapoint_to_step_currents(I_DELAY, I_DUR, amplitudes, DT, T_MAX)
        stimuli = net.cell(list(range(10 * sf))).branch(1).comp(0).data_stimulate(currents)
        return jx.integrate(net, params=params, data_stimuli=stimuli, checkpoint_lengths=checkpoints, compile=bool(args.compile))

    np.random.seed(0)
    amplitudes = torch.as_tensor(2 * I_AMP * np.random.rand(10 * sf), device=device)
    results = {"num_cells": int(num_cells), "num_compartments": num_comps, "num_synapses": num_syn, "build_s": build_s,
               "gpu": torch.cuda.get_device_name(), "dtype": args.dtype}
    for call in ["first", "second"]:
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            v = simulate([], amplitudes)
        torch.cuda.synchronize()
        results[f"simulation_{call}_s"] = time.time() - t0
        print(f"simulation ({call} call): {results[f'simulation_{call}_s']:.1f} s", flush=True)
    np.save(os.path.join(args.out, "voltages_every_40th_cell.npy"), v[::40].cpu().numpy())

    net.delete_trainables()
    for key in ["HH_gNa", "HH_gK", "HH_gLeak"]:
        net.cell("all").branch("all").comp("all").make_trainable(key, verbose=False)
    net.IonotropicSynapse.edge("all").make_trainable("IonotropicSynapse_gS", verbose=False)
    params = [{k: v.to(device).requires_grad_(True) for k, v in p.items()} for p in net.get_parameters()]
    results["num_parameters"] = int(sum(p[k].numel() for p in params for k in p))
    print(f"{results['num_parameters']} trainable parameters", flush=True)
    for call in ["first", "second"]:
        amps = amplitudes * (1.01 if call == "second" else 1.0)
        torch.cuda.synchronize()
        t0 = time.time()
        loss = simulate(params, amps)[-1, -1]
        grads = torch.autograd.grad(loss, [p[k] for p in params for k in p])
        torch.cuda.synchronize()
        results[f"gradient_{call}_s"] = time.time() - t0
        results["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"gradient ({call} call): {results[f'gradient_{call}_s']:.1f} s, peak memory {results['peak_memory_gb']:.1f} GB", flush=True)
    results["grad_finite"] = bool(all(torch.isfinite(g).all() for g in grads))
    tag = torch.cuda.get_device_name().replace(" ", "_")
    with open(os.path.join(args.out, f"timing_{tag}_{args.dtype}.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
