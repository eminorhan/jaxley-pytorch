# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""The synapse step for synapses ordered by postsynaptic compartment (port-specific).

`Network._step_regular_synapses` must give the results of Jaxley's synapse step
(`_step_synapse_state`, then `_synapse_currents`). The tests compare the two on the
same network, with the regular layout switched off for the reference.
"""
import pytest
import torch

import jaxley_pytorch as jx
import jaxley_pytorch.modules.network as network_module
from jaxley_pytorch.channels import HH
from jaxley_pytorch.connect import connect
from jaxley_pytorch.kernels.tree_solve import HAS_TRITON
from jaxley_pytorch.synapses import (
    IonotropicSynapse,
    TanhConductanceSynapse,
    TanhRateSynapse,
    TestSynapse,
)
from jaxley_pytorch.utils.syn_utils import regular_layout
from tests.helpers import np

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason="Needs a GPU and Triton."
)


@pytest.fixture(autouse=True)
def float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def test_regular_layout():
    k, targets = regular_layout([2, 2, 5, 5, 7, 7])
    assert k == 2 and list(targets) == [2, 5, 7]
    k, targets = regular_layout([0, 1, 2])
    assert k == 1 and list(targets) == [0, 1, 2]
    for post in [[], [0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1, 0, 0]]:
        assert regular_layout(post) is None, post


def _cell(kind):
    if kind == "point":
        cell = jx.Cell()
    else:
        branch = jx.Branch([jx.Compartment()] * 2)
        cell = jx.Cell([branch] * 3, parents=[-1, 0, 0])
    cell.insert(HH())
    return cell


def _network(kind, synapses, n=6, k=3, seed=0, device="cpu"):
    """`n` cells; for every synapse type, `k` synapses onto every cell (in order of the
    postsynaptic cell, i.e. regular, or of the presynaptic cell, i.e. not)."""
    rng = np.random.default_rng(seed)
    net = jx.Network([_cell(kind) for _ in range(n)])
    target_branch = 0 if kind == "point" else 1
    for synapse_cls, regular in synapses:
        pairs = []
        for post in range(n):
            pres = rng.choice([c for c in range(n) if c != post], size=k, replace=False)
            pairs += [(int(pre), post) for pre in pres]
        if not regular:
            pairs.sort()
        for pre, post in pairs:
            connect(
                net.cell(pre).branch(0).loc(0.0),
                net.cell(post).branch(target_branch).loc(0.0),
                synapse_cls(),
            )
    for synapse_cls, _ in synapses:
        synapse = synapse_cls()
        name = synapse.name
        # The conductance (`gS`, `gC` for `TestSynapse`), 10 times its default (the
        # default for `TanhConductanceSynapse`, whose conductance can be negative: with
        # more, the gradients grow to 1e25 and amplify round-off beyond the tolerance).
        g, default = next(
            (p, v) for p, v in synapse.synapse_params.items() if p[-3:] in ("_gS", "_gC")
        )
        scale = 1.0 if synapse_cls is TanhConductanceSynapse else 10.0
        edges = net.edges
        is_type = (edges["type"] == name).to_numpy()
        edges.loc[is_type, g] = rng.uniform(0.5, 1.5, is_type.sum()) * scale * default
        getattr(net, name).edge("all").make_trainable(g, verbose=False)
        for state in synapse_cls().synapse_states:
            getattr(net, name).edge("all").record(state, verbose=False)
        getattr(net, name).edge("all").record(f"i_{name}", verbose=False)
    net.cell("all").branch(target_branch).make_trainable("HH_gNa", verbose=False)
    stimulus = torch.as_tensor(jx.step_current(0.5, 3.0, 0.2, 0.025, 8.0), device=device)
    net.cell(0).branch(0).loc(0.0).stimulate(stimulus, verbose=False)
    net.cell("all").branch(target_branch).loc(0.0).record(verbose=False)
    return net


def _simulate(net, batch=False):
    params = [
        {k: v.clone().requires_grad_(True) for k, v in p.items()} for p in net.get_parameters()
    ]
    inputs = [x for p in params for x in p.values()]
    if batch:  # Two sets of parameters, the second with all values scaled.
        params = [
            {k: torch.stack([v, 1.1 * v]) for k, v in p.items()} for p in params
        ]
    recordings = jx.integrate(net, params=params, delta_t=0.025)
    grads = torch.autograd.grad(torch.sum(recordings**2), inputs)
    assert torch.isfinite(recordings).all() and all(torch.isfinite(g).all() for g in grads)
    return recordings.detach(), [g.detach() for g in grads]


def _layout_used(net):
    return ["k" in sp for sp in net._build_synapse_plan(torch.device("cpu"))]


SYNAPSES = [IonotropicSynapse, TanhRateSynapse, TanhConductanceSynapse, TestSynapse]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("kind", ["point", "branched"])
@pytest.mark.parametrize("synapse_cls", SYNAPSES, ids=[s.__name__ for s in SYNAPSES])
def test_regular_step_matches_jaxley_step(synapse_cls, kind, batch, monkeypatch):
    net = _network(kind, [(synapse_cls, True)])
    assert _layout_used(net) == [True]
    if kind == "point":
        assert net._build_synapse_plan(torch.device("cpu"))[0]["targets_are_nodes"]
    rec, grads = _simulate(net, batch)
    assert rec.abs().max() > 0 and all(g.abs().max() > 0 for g in grads)

    monkeypatch.setattr(network_module, "regular_layout", lambda post: None)
    reference = _network(kind, [(synapse_cls, True)])
    assert _layout_used(reference) == [False]
    rec_ref, grads_ref = _simulate(reference, batch)
    np.testing.assert_allclose(rec, rec_ref, rtol=1e-10, atol=1e-12)
    for g, g_ref in zip(grads, grads_ref):
        np.testing.assert_allclose(g, g_ref, rtol=1e-9, atol=1e-12)


def test_unrecorded_synapse_currents_are_not_stored(monkeypatch):
    """Without a recording of `i_<synapse>`, `integrate` does not compute it for regular
    layouts (and the results do not change)."""
    results = []
    for regular in [True, False]:
        if not regular:
            monkeypatch.setattr(network_module, "regular_layout", lambda post: None)
        net = _network("point", [(IonotropicSynapse, True)])
        net.delete_recordings()
        net.cell("all").branch(0).loc(0.0).record(verbose=False)
        net.IonotropicSynapse.edge("all").record("IonotropicSynapse_s", verbose=False)
        params = [
            {k: v.clone().requires_grad_(True) for k, v in p.items()}
            for p in net.get_parameters()
        ]
        rec, states = jx.integrate(net, params=params, delta_t=0.025, return_states=True)
        assert ("i_IonotropicSynapse" in states) == (not regular)
        grads = torch.autograd.grad(rec.square().sum(), [x for p in params for x in p.values()])
        results.append((rec.detach(), [g.detach() for g in grads]))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-10, atol=1e-12)
    for g, g_ref in zip(results[0][1], results[1][1]):
        np.testing.assert_allclose(g, g_ref, rtol=1e-9, atol=1e-12)


def test_regular_and_other_synapse_types(monkeypatch):
    synapses = [(IonotropicSynapse, True), (TestSynapse, False)]
    net = _network("branched", synapses)
    assert _layout_used(net) == [True, False]
    rec, grads = _simulate(net)

    monkeypatch.setattr(network_module, "regular_layout", lambda post: None)
    reference = _network("branched", synapses)
    rec_ref, grads_ref = _simulate(reference)
    np.testing.assert_allclose(rec, rec_ref, rtol=1e-10, atol=1e-12)
    for g, g_ref in zip(grads, grads_ref):
        np.testing.assert_allclose(g, g_ref, rtol=1e-9, atol=1e-12)


@requires_gpu
def test_regular_step_compiled(monkeypatch):
    """Under `torch.compile`, on the GPU, against the reference step (eager)."""
    results = []
    for regular in [True, False]:
        if not regular:
            monkeypatch.setattr(network_module, "regular_layout", lambda post: None)
        net = _network("point", [(IonotropicSynapse, True)], device="cuda")
        assert _layout_used(net) == [regular]
        params = [
            {k: v.cuda().requires_grad_(True) for k, v in p.items()}
            for p in net.get_parameters()
        ]
        v = jx.integrate(net, params=params, delta_t=0.025, compile=regular)
        grads = torch.autograd.grad(torch.sum(v**2), [x for p in params for x in p.values()])
        results.append((v.detach().cpu(), [g.cpu() for g in grads]))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-10, atol=1e-12)
    for g, g_ref in zip(results[0][1], results[1][1]):
        np.testing.assert_allclose(g, g_ref, rtol=1e-8, atol=1e-12)


@requires_gpu
def test_regular_step_compiled_for_several_sizes():
    """Networks with different numbers of synapses per compartment, compiled one after
    another in one process (where `torch.compile` turns sizes that changed dynamic: the
    reshape to `(..., num_targets, k)` must still compile)."""
    for n, k in [(3, 2), (6, 3), (8, 4)]:
        net = _network("point", [(IonotropicSynapse, True)], n=n, k=k, device="cuda")
        assert _layout_used(net) == [True]
        params = [
            {key: v.cuda().requires_grad_(True) for key, v in p.items()}
            for p in net.get_parameters()
        ]
        v = jx.integrate(net, params=params, delta_t=0.025, compile=True)
        grads = torch.autograd.grad(v.square().sum(), [x for p in params for x in p.values()])
        assert torch.isfinite(v).all() and all(torch.isfinite(g).all() for g in grads)
