"""The biophysical RNN of paper Fig. 4, shared by the Fig. 4 scripts.

Port of `nex/fig5_rnn/scripts/{network.py, compute_EI.py, EI_stimulus.py}` of
https://github.com/mackelab/jaxley_experiments. The random draws (numpy's legacy RNG)
are made in the same order as there, so a given `RNN_seed` builds the paper's network.

Every recurrent unit has three single-compartment branches (apical, soma, basal) with
the L5PC channels; the readout units are passive. Synapses run from the soma of the
presynaptic unit to the apical dendrite of the postsynaptic one (and to the readouts).
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, lfilter

import jaxley_pytorch as jx
from jaxley_pytorch.connect import connect
from jaxley_pytorch.mech.channels.fohlmeister97 import Leak
from jaxley_pytorch.mech.channels.l5pc import (
    H,
    M,
    SKE2,
    CaHVA,
    CaLVA,
    CaNernstReversal,
    CaPump,
    NaTs2T,
    SKv3_1,
)
from jaxley_pytorch.synapses import IonotropicSynapse

# The paper states gains in units where the evidence-integration network has gain 5.0.
G_SCALING = 100_000 / 2 / np.pi / 10.0 / 1.0


def build_unit() -> jx.Cell:
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    cell = jx.Cell([branch, branch, branch], parents=[-1, 0, 0])
    cell.branch(0).add_to_group("apical")
    cell.branch(1).add_to_group("soma")
    cell.branch(2).add_to_group("basal")
    cell.compute_xyz()

    cell.apical.insert(NaTs2T())
    cell.apical.set("NaTs2T_gNaTs2T", 0.026145)
    cell.apical.insert(SKv3_1())
    cell.apical.set("SKv3_1_gSKv3_1", 0.004226)
    cell.apical.insert(M())
    cell.apical.set("M_gM", 0.000143)
    cell.apical.insert(H())
    # The paper evaluates the L5PC profile (-0.8696 + 2.087 exp(0.0031 d)) 8e-5 at the
    # distance between the branch and itself (d = 0).
    cell.apical.set("H_gH", (-0.8696 + 2.087) * 8e-5)
    cell.apical.set("capacitance", 2.0)

    ca_dynamics = CaNernstReversal()
    ca_dynamics.channel_constants["T"] = 307.15
    cell.soma.insert(ca_dynamics)
    cell.soma.set("CaCon_i", 5e-05)
    cell.soma.set("CaCon_e", 2.0)
    cell.soma.insert(CaPump())
    cell.soma.set("CaPump_gamma", 0.000609)
    cell.soma.set("CaPump_decay", 210.485291)
    cell.soma.insert(CaHVA())
    cell.soma.set("CaHVA_gCaHVA", 0.000994)
    cell.soma.insert(CaLVA())
    cell.soma.set("CaLVA_gCaLVA", 0.000333)
    cell.soma.insert(SKv3_1())
    cell.soma.set("SKv3_1_gSKv3_1", 0.303472)
    cell.soma.insert(SKE2())
    cell.soma.set("SKE2_gSKE2", 0.008407)
    cell.soma.insert(NaTs2T())
    cell.soma.set("NaTs2T_gNaTs2T", 0.983955)

    cell.basal.insert(H())
    cell.basal.set("H_gH", 8e-5)

    cell.insert(Leak())
    cell.set("Leak_gLeak", 3e-05)
    cell.set("Leak_eLeak", -75.0)
    cell.set("axial_resistivity", 100.0)
    cell.set("eNa", 50.0)
    cell.set("eK", -85.0)
    return cell


def build_readout() -> jx.Cell:
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    readout = jx.Cell([branch], parents=[-1])
    readout.insert(Leak())
    readout.set("Leak_gLeak", 3e-2)
    return readout


def edge_cells(net: jx.Network) -> Tuple[np.ndarray, np.ndarray]:
    """Pre- and postsynaptic cell of every edge (`pre_index` and `post_index` are labels
    of `net.nodes`)."""
    cell_of_comp = net.nodes["global_cell_index"]
    pre = cell_of_comp.loc[net.edges["pre_index"]].to_numpy()
    post = cell_of_comp.loc[net.edges["post_index"]].to_numpy()
    return pre, post


def init_inh_ex_gs(
    net: jx.Network,
    inh_indices: List[int],
    gain: float,
    out_indices: List[int] = (),
    e_syn_inh: float = -75.0,
    e_syn_ex: float = 0.0,
    out_scale: float = 1.0,
    dist: str = "normal",
) -> np.ndarray:
    """Random E/I weights whose recurrent eigenspectrum has a bulk of radius `gain`.

    Port of `init_inh_ex_gS` (paper code). Sets `IonotropicSynapse_gS` and `_e_syn` of
    every edge and returns the signed connectivity matrix.
    """
    pre_cells, post_cells = edge_cells(net)
    n_units = int(max(pre_cells.max(), post_cells.max())) + 1
    out_indices = list(out_indices)
    n_units_rec = n_units - len(out_indices)
    rec_inds = [i for i in range(n_units) if i not in out_indices]
    n_conn_rec = int(np.sum(~np.isin(post_cells, out_indices)))
    average_in_rec = n_conn_rec / n_units_rec
    p_inh = len(inh_indices) / n_units_rec
    p_rec = average_in_rec / n_units_rec
    ei_ratio = (1 - p_inh) / p_inh
    normaliser = np.sqrt((1 / (1 - (2 * p_rec) / np.pi)) / ei_ratio)

    conn_matrix = np.zeros((n_units, n_units))
    weights = np.zeros(len(pre_cells))
    e_syn = np.zeros(len(pre_cells))
    for i, (pre, post) in enumerate(zip(pre_cells, post_cells)):
        if dist == "normal":
            samp = abs(np.random.normal(0, 1, 1)[0])
        else:
            samp = abs(np.random.uniform(0, 1))
        if post in out_indices:
            samp *= out_scale
        if pre in inh_indices:
            weights[i] = samp * normaliser * gain * ei_ratio / np.sqrt(average_in_rec)
            e_syn[i] = e_syn_inh
            conn_matrix[pre, post] = -weights[i]
        else:
            weights[i] = samp * normaliser * gain / np.sqrt(average_in_rec)
            e_syn[i] = e_syn_ex
            conn_matrix[pre, post] = weights[i]
    net.edges["IonotropicSynapse_gS"] = weights
    net.edges["IonotropicSynapse_e_syn"] = e_syn
    radius = np.max(np.abs(np.linalg.eigvals(conn_matrix[rec_inds][:, rec_inds])))
    print(f"spectral radius of the recurrent weights: {radius:.3g} (gain {gain:.3g})")
    return conn_matrix


def initialize_rnn(params: Dict) -> Tuple[jx.Network, np.ndarray, Optional[np.ndarray]]:
    """Port of `initialize_RNN` (paper code). Returns the network, the signed
    connectivity matrix and the initial input weights."""
    n_cells, n_inh, n_out = params["n_rec"], params["n_inh"], params["n_out"]
    np.random.seed(params["RNN_seed"])

    unit, readout = build_unit(), build_readout()
    net = jx.Network([unit for _ in range(n_cells)] + [readout] * n_out)
    for i in range(n_cells):
        for j in range(n_cells):
            if np.random.binomial(1, params["rec_conn_prob"]):
                connect(
                    net.cell(i).branch(1).comp(0),
                    net.cell(j).branch(0).comp(0),
                    IonotropicSynapse(),
                )
        if n_out:
            for k in range(n_out):
                if params["out_conn_prob"] >= 1.0 or np.random.binomial(
                    1, params["out_conn_prob"]
                ):
                    connect(
                        net.cell(i).branch(1).comp(0),
                        net.cell(n_cells + k).branch(0).comp(0),
                        IonotropicSynapse(),
                    )

    conn_matrix = init_inh_ex_gs(
        net,
        list(range(n_inh)),
        params["init_gain"],
        out_indices=[n_cells + k for k in range(n_out)],
        out_scale=params["out_scale"],
    )

    input_weights = None
    if n_out:
        if params["in_conn_prob"] < 1.0:
            input_weights = np.zeros((n_cells, n_out))
            for i in range(n_cells):
                if np.random.rand() < params["in_conn_prob"]:
                    input_weights[i, 0] = abs(np.random.uniform(0, 1))
                if np.random.rand() < params["in_conn_prob"]:
                    input_weights[i, 1] = abs(np.random.uniform(0, 1))
            input_weights = input_weights * params["inp_scale"]
        else:
            input_weights = np.random.randn(n_cells) * params["inp_scale"]

    _, post_cells = edge_cells(net)
    net.edges["IonotropicSynapse_k_minus"] = np.where(
        post_cells < n_cells, params["k_minus"], params["out_k_minus"]
    ).astype(float)
    net.set("v", -65.0)
    net.init_states()
    return net, conn_matrix, input_weights


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype="low", analog=False)
    return lfilter(b, a, data)


def gen_stim(task_params, stim_mean_dist="normal", return_details=False):
    """One evidence-integration trial: low-pass filtered Gaussian noise whose mean is
    drawn around -0.005 or +0.005. The label is the sign of the summed stimulus outside
    the response window (port of `gen_stim`, paper code)."""
    if stim_mean_dist == "normal":
        mean_offset = np.random.binomial(1, 0.5)
        stim_mean = np.random.normal(
            task_params["stim_mean"][mean_offset], task_params["stim_mean_std"]
        )
    else:
        stim_mean = np.random.uniform(*task_params["stim_mean"])
    n = int(task_params["total_time"] / task_params["dt"])
    stim = np.random.normal(stim_mean, task_params["stim_std"], size=n)
    stim = butter_lowpass_filter(stim, 5, task_params["stim_freq_max"], 6)

    mask = np.zeros_like(stim)
    start = int(task_params["response_time"] / task_params["dt"])
    end = int((task_params["response_time"] + task_params["response_length"]) / task_params["dt"])
    mask[start:end] = 1
    evidence = np.sum(stim * (mask == 0))
    target = np.array([1.0, 0.0]) if evidence > 0 else np.array([0.0, 1.0])
    if return_details:
        return stim, target, mask, evidence, stim_mean
    return stim, target, mask


def gen_stim_batch(task_params, batch_size, stim_mean_dist="normal"):
    trials = [gen_stim(task_params, stim_mean_dist) for _ in range(batch_size)]
    return tuple(np.stack(x) for x in zip(*trials))
