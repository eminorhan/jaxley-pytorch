"""Paper Fig. 5: a biophysical network with 100k parameters classifies MNIST digits.

Port of `nex/fig6_mnist/notebooks/{02_train_network, 01_forward_passes}.ipynb` of
https://github.com/mackelab/jaxley_experiments. 28 x 28 input neurons (ball and stick,
Hodgkin-Huxley), 64 hidden neurons with reconstructed morphologies (55 SWC files of
Zenodo 14166049, `fig6_morphologies.zip`) and 10 passive output neurons, densely
connected by conductance-based synapses (75% excitatory, 25% inhibitory presynaptic
neurons, separately parameterized). Every pixel drives a step current into an input
neuron; the class is read out from the somatic voltages of the outputs after 10 ms.

Modes:
  evaluate  The paper's trained network (`parameters_1770` in `fig6_results.zip`):
            test accuracy (paper: 94.2% on the test set, 95.2% on 1,024 images), and the
            accuracy when groups of parameters are reset to their initial values
            (Fig. 5e).
  attack    Adversarial attacks (Fig. 5f-g): 128 images (8 seeds x 16, as in the paper),
            a random target label, 30 steps of normalized gradient descent (learning
            rate 5) on the logit of the image.
  time      Time per gradient step of training (batch 16; paper: 25 s on a V100).

The network is built as the paper's Jaxley version did: the SWC reader of
`legacy_jaxley.py` and `fully_connect(..., random_post_comp=True)` with the paper's
seeds, which places every synapse on the paper's compartment. MNIST is downloaded with
torchvision; its image order differs from TensorFlow Datasets' (used by the paper).

Usage: python fig5_mnist.py --out <dir> --mode evaluate|attack|time
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
from jaxley_pytorch.channels import HH, Leak
from jaxley_pytorch.connect import fully_connect
from jaxley_pytorch.optimize.transforms import SigmoidTransform
from jaxley_pytorch.synapses import IonotropicSynapse
from legacy_jaxley import read_swc_legacy

DATA = "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/data"
REFERENCE = f"{DATA}/zenodo_14166049/fig6_results.zip"
# The paper's order of morphologies (`02_train_network.ipynb`, cell 12).
MORPHOLOGIES = (
    "n171 n174 n130 n400 n404 n140 n127 n122 n412 n125 n403 n420 n407 n417 n144 n141 n145 "
    "n410 n129 n416 n414 n408 n142 n406 n179 n177 n411 n181 n184 n402 n176 n409 n415 n422 "
    "n123 n121 n401 n128 n421 n418 n126 n172 n413 n170 n175 n178 n173 n405 n183 n143 n180 "
    "n419 n120 n423 n182"
).split()
DT, T_MAX = 0.025, 10.01
I_DELAY, I_DUR = 0.1, 1.0
NSEG = 4
COMP_LENGTH = 40.0
I_AMP_FACTOR = COMP_LENGTH / 10
READOUT_SCALE = 8
SAFETY = 5.0
BOUNDS = {
    "Conns1_exc_gS": (-SAFETY / 28**2 / 25.0, SAFETY / 28**2 / 25.0),
    "Conns1_inh_gS": (-SAFETY / 28**2 / 25.0, SAFETY / 28**2 / 25.0),
    "Conns2_exc_gS": (-SAFETY / 64.0 / 25.0 * I_AMP_FACTOR / READOUT_SCALE, SAFETY / 64.0 / 25.0 * I_AMP_FACTOR / READOUT_SCALE),
    "Conns2_inh_gS": (-SAFETY / 64.0 / 25.0 * I_AMP_FACTOR / READOUT_SCALE, SAFETY / 64.0 / 25.0 * I_AMP_FACTOR / READOUT_SCALE),
    "HH_gNa": (0.05, 0.5),
    "HH_gK": (0.01, 0.1),
    "HH_gLeak": (0.0001, 0.001),
    "axial_resistivity": (30.0, 1000.0),
}
N_IN, N_HIDDEN, N_OUT = 28**2, 64, 10
RESETS = {
    "no_first_layer": [0, 1],
    "no_second_layer": [2, 3],
    "no_sodium": [4],
    "no_potassium": [5],
    "no_leak": [6],
    "no_channels": [4, 5, 6],
    "no_axial_resistivities": [7],
}


def reference(name):
    sys.path.insert(0, "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools")
    from safe_pickle import load_from_zip

    return load_from_zip(REFERENCE, f"results/{name}")


def ball_and_stick():
    comp = jx.Compartment()
    branch = jx.Branch(comp, NSEG)
    cell = jx.Cell(branch, [-1])
    cell.branch(0).comp(list(range(1, NSEG))).set("length", COMP_LENGTH)
    cell.branch(0).comp(list(range(1, NSEG))).set("radius", 1.0)
    cell.branch(0).comp(0).set("length", 20.0)
    cell.branch(0).comp(0).set("radius", 10.0)
    return cell


def build_network(morph_dir):
    morph_cells = [read_swc_legacy(os.path.join(morph_dir, n, f"{n}.CNG.swc"), nseg=NSEG) for n in MORPHOLOGIES]
    hidden = [morph_cells[i % len(morph_cells)] for i in range(N_HIDDEN)]
    net = jx.Network([ball_and_stick() for _ in range(N_IN)] + hidden + [ball_and_stick() for _ in range(N_OUT)])

    np.random.seed(4)
    layer_inds = [0, N_IN, N_IN + N_HIDDEN]
    for layer in range(2):
        a, b = layer_inds[layer], layer_inds[layer + 1]
        exc = np.random.choice(np.arange(a, b), int((b - a) * 0.75), replace=False)
        inh = np.setdiff1d(np.arange(a, b), exc)
        net.cell(exc.tolist()).add_to_group(f"layer{layer}_exc")
        net.cell(inh.tolist()).add_to_group(f"layer{layer}_inh")
    net.cell(list(range(N_IN))).add_to_group("layer0")
    net.cell(list(range(N_IN, N_IN + N_HIDDEN))).add_to_group("layer1")
    net.cell(list(range(N_IN + N_HIDDEN, N_IN + N_HIDDEN + N_OUT))).add_to_group("layer2")
    net.layer0.insert(HH())
    net.layer1.insert(HH())
    net.layer2.insert(Leak())
    net.set("axial_resistivity", 100.0)

    np.random.seed(5)
    for pre, post, name in [
        ("layer0_exc", "layer1", "Conns1_exc"),
        ("layer0_inh", "layer1", "Conns1_inh"),
        ("layer1_exc", "layer2", "Conns2_exc"),
        ("layer1_inh", "layer2", "Conns2_inh"),
    ]:
        fully_connect(getattr(net, pre).cell("all"), getattr(net, post).cell("all"),
                      IonotropicSynapse().change_name(name), random_post_comp=True)

    np.random.seed(0)
    sizes = {name: len(getattr(net, name).edges) for name in ["Conns1_exc", "Conns1_inh", "Conns2_exc", "Conns2_inh"]}
    w = {name: np.random.randn(sizes[name]) for name in sizes}
    for name in ["Conns1_exc", "Conns1_inh"]:
        w[name] = w[name] / 28**2 / 25.0
    for name in ["Conns2_exc", "Conns2_inh"]:
        w[name] = w[name] / 64 / 25.0 * I_AMP_FACTOR / READOUT_SCALE
    for name in sizes:
        getattr(net, name).edge("all").set(f"{name}_gS", w[name])
        getattr(net, name).edge("all").set(f"{name}_k_minus", 0.25)

    net.set("v", -65.0)
    net.init_states()
    for name in sizes:
        getattr(net, name).edge("all").make_trainable(f"{name}_gS", verbose=False)
    for key in ["HH_gNa", "HH_gK", "HH_gLeak", "axial_resistivity"]:
        net.cell("all").branch("all").make_trainable(key, verbose=False)
    net.cell(list(range(N_IN + N_HIDDEN, N_IN + N_HIDDEN + N_OUT))).branch(0).comp(0).record(verbose=False)
    return net


def load_mnist(train):
    import torchvision

    ds = torchvision.datasets.MNIST(f"{DATA}/mnist", train=train, download=True)
    images = ds.data.reshape(-1, 28**2).double() / 255.0
    return images, ds.targets.long()


class Model:
    def __init__(self, net, device, compile):
        self.net, self.device, self.compile = net, device, compile
        self.keys = [next(iter(p)) for p in net.get_parameters()]
        self.i_amp = I_AMP_FACTOR * 50.0 / 500.0

    def simulate(self, params, images, checkpoint_lengths=None):
        """images: (batch, 784) in [0, 1]. Returns the output voltages (batch, 10, time)."""
        currents = jx.datapoint_to_step_currents(I_DELAY, I_DUR, self.i_amp * images, DT, T_MAX)
        stimuli = self.net.cell(list(range(N_IN))).branch(0).comp(NSEG - 1).data_stimulate(currents)
        return jx.integrate(self.net, params=params, data_stimuli=stimuli,
                            checkpoint_lengths=checkpoint_lengths, compile=self.compile)

    def logits(self, params, images):
        return (self.simulate(params, images)[..., -1] + 65.0) / 3.0


def as_params(model, groups, device):
    return [{k: torch.as_tensor(np.asarray(g[k]), device=device)} for g, k in zip(groups, model.keys)]


def accuracy(model, params, images, labels, batch):
    correct = []
    with torch.no_grad():
        for x, y in zip(torch.split(images, batch), torch.split(labels, batch)):
            pred = model.logits(params, x.to(model.device)).argmax(-1).cpu()
            correct.append(pred == y)
    return float(torch.cat(correct).double().mean())


def check_parameter_counts(model, groups):
    counts = {k: int(p[k].shape[-1]) for p, k in zip(model.net.get_parameters(), model.keys)}
    paper = {k: int(np.asarray(g[k]).shape[-1]) for g, k in zip(groups, model.keys)}
    assert counts == paper, (counts, paper)
    return counts


def evaluate(args, device):
    net = build_network(args.morphologies)
    model = Model(net, device, bool(args.compile))
    trained, initial = reference("parameters_1770"), reference("initial_parameters")
    counts = check_parameter_counts(model, trained)
    print("trainable parameters per group:", counts, "total", sum(counts.values()), flush=True)
    images, labels = load_mnist(train=False)
    params = as_params(model, trained, device)
    t0 = time.time()
    results = {"parameter_counts": counts, "accuracy_test_10000": accuracy(model, params, images, labels, args.batch)}
    results["eval_time_s"] = time.time() - t0
    results["accuracy_test_first_1024"] = accuracy(model, params, images[:1024], labels[:1024], args.batch)
    print(results, flush=True)
    paper_resets = {k: float(v) for k, v in reference("accuracies/dropout_accuracies.pkl").items()}
    results["resets"] = {}
    for name, inds in RESETS.items():
        reset = [initial[i] if i in inds else trained[i] for i in range(len(trained))]
        acc = accuracy(model, as_params(model, reset, device), images[:1024], labels[:1024], args.batch)
        results["resets"][name] = acc
        print(f"{name}: {acc:.4f} (paper {paper_resets[name]:.4f})", flush=True)
    results["paper_resets"] = paper_resets
    with open(os.path.join(args.out, "evaluate.json"), "w") as f:
        json.dump(results, f, indent=1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    names = list(RESETS)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [results["resets"][n] for n in names], width=0.4, color="k", label="ours")
    ax.bar(x + 0.2, [paper_resets[n] for n in names], width=0.4, color="0.6", label="paper")
    ax.axhline(results["accuracy_test_first_1024"], color="#08519c", label="ours, all trained")
    ax.axhline(paper_resets["all"], color="#08519c", ls="--", label="paper, all trained")
    ax.set_xticks(x, ["1st layer syn.", "2nd layer syn.", "sodium", "potassium", "leak", "all channels", "r_a"], rotation=60)
    ax.set_ylabel("accuracy w/o")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6)
    ax.set_title("e  reset parameters to initial values")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig5_e.png"), dpi=150)


def attack(args, device):
    net = build_network(args.morphologies)
    model = Model(net, device, bool(args.compile))
    trained = reference("parameters_1770")
    params = as_params(model, trained, device)
    train_images, train_labels = load_mnist(train=True)
    all_accuracies, all_travelled = [], []
    for seed in range(1, 9):
        np.random.seed(seed)
        n, n_candidates = 16, 24
        indices = np.arange((seed - 1) * n_candidates, seed * n_candidates)
        initial_labels = train_labels[indices].numpy()
        target_labels = np.random.randint(0, 10, n_candidates)
        keep = target_labels != initial_labels
        indices, initial_labels, targets = indices[keep][:n], initial_labels[keep][:n], target_labels[keep][:n]
        x = train_images[indices].clone()
        x[x == 0.0] += 0.01
        x[x == 1.0] -= 0.01
        z = torch.log(x / (1 - x)).to(device)  # Logit: the attack is unconstrained.
        targets_t = torch.as_tensor(targets, device=device)
        travelled = np.zeros(n)
        accs = []
        for step in range(args.attack_steps):
            z = z.detach().requires_grad_(True)
            logits = model.logits(params, torch.sigmoid(z))
            accs.append(float((logits.argmax(-1).cpu().numpy() == initial_labels).mean()))
            loss = -torch.log_softmax(logits, -1)[torch.arange(n), targets_t]
            (g,) = torch.autograd.grad(loss.sum(), z)
            with torch.no_grad():
                step_ = args.attack_lr * g / torch.sqrt(torch.sum(g**2, dim=-1, keepdim=True))
                previous = z.clone()
                z = z - step_
                travelled += torch.sqrt(torch.sum((z - previous) ** 2, -1)).cpu().numpy()
        all_accuracies.append(accs)
        all_travelled.append(travelled)
        print(f"seed {seed}: accuracies {np.round(accs, 3).tolist()}", flush=True)
    mean_acc = np.mean(np.asarray(all_accuracies), axis=0)
    paper = np.asarray(reference("adv_attacks/biophys_accuracies.pkl"), dtype=float)
    ann = np.asarray(reference("adv_attacks/ann_accuracies.pkl"), dtype=float)
    with open(os.path.join(args.out, "attack.json"), "w") as f:
        json.dump({"accuracy_per_step": mean_acc.tolist(), "paper_biophysical": paper.tolist(), "paper_ann": ann.tolist()}, f, indent=1)
    eps = np.arange(len(mean_acc)) * args.attack_lr
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(eps, mean_acc, c="#08519c", label="biophysical (ours)")
    ax.plot(np.arange(len(paper)) * 5.0, paper, c="#08519c", ls="--", label="biophysical (paper)")
    ax.plot(np.arange(len(ann)) * 5.0, ann, c="k", label="ANN (paper)")
    ax.set_xlabel("norm of perturbation (logit space)")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=7)
    ax.set_title("g  adversarial attacks (128 images)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig5_g.png"), dpi=150)


def time_gradient(args, device):
    net = build_network(args.morphologies)
    model = Model(net, device, bool(args.compile))
    images, labels = load_mnist(train=True)
    transforms = [SigmoidTransform(*BOUNDS[k]) for k in model.keys]
    initial = reference("initial_parameters")
    opt = [{k: t.inverse(torch.as_tensor(np.asarray(g[k]), device=device)).requires_grad_(True)}
           for g, k, t in zip(initial, model.keys, transforms)]
    n_steps = int(T_MAX // DT) + 2
    ckpt = [int(np.ceil(np.sqrt(n_steps)))] * 2
    times = []
    for step in range(args.time_steps):
        x, y = images[step * 16 : (step + 1) * 16].to(device), labels[step * 16 : (step + 1) * 16].to(device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        params = [{k: t.forward(p[k])} for p, k, t in zip(opt, model.keys, transforms)]
        logits = (model.simulate(params, x, ckpt)[..., -1] + 65.0) / 3.0
        loss = torch.nn.functional.cross_entropy(logits, y)
        grads = torch.autograd.grad(loss, [p[k] for p, k in zip(opt, model.keys)])
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append(time.time() - t0)
        print(f"step {step}: loss {loss.item():.4f}, {times[-1]:.2f} s", flush=True)
    with open(os.path.join(args.out, "time.json"), "w") as f:
        json.dump({"seconds_per_gradient_step": times, "median_after_first": float(np.median(times[1:]))}, f, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["evaluate", "attack", "time"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--morphologies", default=f"{DATA}/fig6_morphologies/morphologies")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--attack-steps", type=int, default=30)
    parser.add_argument("--attack-lr", type=float, default=5.0)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--compile", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)  # As in the paper (jax_enable_x64).
    device = torch.device(args.device)
    {"evaluate": evaluate, "attack": attack, "time": time_gradient}[args.mode](args, device)


if __name__ == "__main__":
    main()
