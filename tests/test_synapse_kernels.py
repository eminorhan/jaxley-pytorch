# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Tests of the fused synapse kernels (not part of Jaxley).

`jaxley_pytorch.kernels.synapse_step` must give the results of
`Network._step_regular_synapses`, which the tests compare with on the same networks
with the kernels switched off (`kernel_plan` returning no entries).
"""
import pytest
import torch

import jaxley_pytorch as jx
import jaxley_pytorch.kernels.synapse_step as synapse_step
from jaxley_pytorch.channels import HH
from jaxley_pytorch.kernels.tree_solve import HAS_TRITON
from jaxley_pytorch.synapses import KineticSynapse, SigmoidRateSynapse
from tests.helpers import np

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason="Needs a GPU and Triton."
)

# Per synapse model: ranges of the parameters (`gS` relative to its default).
PARAMS = {
    KineticSynapse: {"gS": (5.0, 15.0), "e_syn": (-80.0, 0.0), "alpha": (0.5, 1.5),
                     "beta": (0.1, 0.3), "v_th": (-50.0, -30.0), "delta": (1.5, 3.0)},
    SigmoidRateSynapse: {"gS": (5.0, 15.0), "x_offset": (-70.0, -60.0), "slope": (0.2, 0.3)},
}


@pytest.fixture(autouse=True)
def float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _network(synapse_cls, kind="point", n=6, k=3, trainable=None, record_current=True, seed=0):
    """`n` HH cells, `k` synapses onto every cell (ordered by postsynaptic cell), every
    synapse parameter random per synapse and trainable (or those in `trainable`)."""
    rng = np.random.default_rng(seed)
    if kind == "point":
        cell = jx.Cell()
    else:
        cell = jx.Cell([jx.Branch([jx.Compartment()] * 2)] * 3, parents=[-1, 0, 0])
    cell.insert(HH())
    net = jx.Network([cell] * n)
    target_branch = 0 if kind == "point" else 1
    pre = np.concatenate([rng.choice(np.delete(np.arange(n), post), k, replace=False) for post in range(n)])
    post = np.repeat(np.arange(n), k)
    nodes = net.nodes

    def node_of(cells, branch):
        """The node (index of `.nodes`) of compartment 0 of `branch` of every cell."""
        sel = nodes[(nodes["local_branch_index"] == branch) & (nodes["local_comp_index"] == 0)]
        return sel.sort_values("global_cell_index").index.to_numpy()[cells]

    net._append_multiple_synapses(
        net.select(nodes=node_of(pre, 0)).nodes,
        net.select(nodes=node_of(post, target_branch)).nodes,
        synapse_cls(),
    )
    name = synapse_cls().name
    defaults = synapse_cls().synapse_params
    for param, (low, high) in PARAMS[synapse_cls].items():
        values = rng.uniform(low, high, len(net.edges))
        if param == "gS":
            values = values * defaults[f"{name}_gS"]
        net.edges[f"{name}_{param}"] = values
    for param in trainable if trainable is not None else PARAMS[synapse_cls]:
        getattr(net, name).edge("all").make_trainable(f"{name}_{param}", verbose=False)
    net.cell("all").branch(target_branch).make_trainable("HH_gNa", verbose=False)
    net.cell("all").branch(target_branch).loc(0.0).record(verbose=False)
    for state in synapse_cls().synapse_states:
        getattr(net, name).edge("all").record(state, verbose=False)
    if record_current:
        getattr(net, name).edge("all").record(f"i_{name}", verbose=False)
    return net, net.cell(0).branch(0).loc(0.0)


def _simulate(net, stim_view, batch=False, **kwargs):
    params = [
        {key: v.cuda().clone().requires_grad_(True) for key, v in p.items()}
        for p in net.get_parameters()
    ]
    # Two simulations (amplitudes of shape `(2, 1)`: a batch of one input each), or one.
    amplitude = torch.tensor([[0.2], [0.3]] if batch else [0.2], device="cuda")
    current = jx.datapoint_to_step_currents(0.5, 3.0, amplitude, 0.025, 8.0)
    v = jx.integrate(
        net, params=params, data_stimuli=stim_view.data_stimulate(current), delta_t=0.025, **kwargs
    )
    grads = torch.autograd.grad(torch.sum(v**2), [x for p in params for x in p.values()])
    assert torch.isfinite(v).all() and all(torch.isfinite(g).all() for g in grads)
    return v.detach().cpu(), [g.detach().cpu() for g in grads]


def _compare(build, monkeypatch, rtol=1e-9, **kwargs):
    """Simulate with the kernels and without, on two networks from `build()`."""
    net, view = build()
    plan = net._build_synapse_plan(torch.device("cuda"))
    assert all("kernel_model" in sp for sp in plan)
    calls = []
    fused_step = synapse_step.fused_step

    def counted(*args):
        result = fused_step(*args)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(synapse_step, "fused_step", counted)
    v, grads = _simulate(net, view, **kwargs)
    assert calls and all(calls)
    monkeypatch.setattr(synapse_step, "kernel_plan", lambda *args: {})
    reference, reference_view = build()
    assert not any("kernel_model" in sp for sp in reference._build_synapse_plan(torch.device("cuda")))
    v_ref, grads_ref = _simulate(reference, reference_view, **kwargs)
    np.testing.assert_allclose(v, v_ref, rtol=rtol, atol=1e-10)
    for g, g_ref in zip(grads, grads_ref):
        np.testing.assert_allclose(g, g_ref, rtol=rtol, atol=1e-10 * max(1.0, g_ref.abs().max().item()))


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("kind", ["point", "branched"])
@pytest.mark.parametrize("synapse_cls", [KineticSynapse, SigmoidRateSynapse])
def test_kernels_match_pytorch(synapse_cls, kind, batch, monkeypatch):
    _compare(lambda: _network(synapse_cls, kind), monkeypatch, batch=batch)


@pytest.mark.parametrize("synapse_cls", [KineticSynapse, SigmoidRateSynapse])
def test_kernels_without_recorded_current_and_only_gS_trainable(synapse_cls, monkeypatch):
    build = lambda: _network(synapse_cls, trainable=["gS"], record_current=False)
    _compare(build, monkeypatch)


def test_kernels_with_more_synapses_than_a_block(monkeypatch):
    """`k` larger than the synapses of one loop iteration of the kernels."""
    assert synapse_step._MAX_BLOCK_K < 150
    _compare(lambda: _network(KineticSynapse, n=160, k=150, trainable=["gS"]), monkeypatch)


def test_kernels_with_atomic_adds(monkeypatch):
    """Without the transposed index (as for very uneven numbers of outgoing synapses)."""
    monkeypatch.setattr(synapse_step, "_MAX_TRANSPOSED_FACTOR", -1)
    net, _ = _network(KineticSynapse)
    assert net._build_synapse_plan(torch.device("cuda"))[0]["kernel_inv"].numel() == 0
    _compare(lambda: _network(KineticSynapse), monkeypatch)


def test_kernels_compiled(monkeypatch):
    _compare(lambda: _network(KineticSynapse, trainable=["gS"]), monkeypatch, compile=True)


def _reference_step(v, s, pre, k, model, params, area, dt, diff):
    """The custom op in PyTorch (targets are nodes): the library's formulas."""
    B, E = v.shape[0], pre.shape[0]
    T = E // k
    vpre = v[:, pre.long()].reshape(B, T, k)
    vp = v[:, :T, None]
    p = [x.reshape(T, k) for x in params]
    area = area.reshape(T, k)
    if model == synapse_step.KINETIC:
        gs, e_syn, alpha, beta, v_th, delta = p
        a = alpha * torch.sigmoid((vpre - v_th) / delta)
        rate = a + beta
        s_inf = a / rate
        s_new = s_inf + (s.reshape(B, T, k) - s_inf) * torch.exp(-dt * rate)
        i0 = gs * s_new * (vp - e_syn)
        i1 = gs * s_new * ((vp + diff) - e_syn)
    else:
        gs, x_offset, slope = p[:3]
        i0 = -gs * torch.sigmoid((vpre - x_offset) * slope)
        i1 = -gs * torch.sigmoid(((vpre + diff) - x_offset) * slope)
    d0 = i0 / area * 100000.0
    d1 = i1 / area * 100000.0
    vt = (d1 - d0) / diff
    ct = d0 - vt * vp
    out = [i0.reshape(B, E), vt.sum(-1), ct.sum(-1)]
    return ([s_new.reshape(B, E)] if model == synapse_step.KINETIC else []) + out


@pytest.mark.parametrize("synapse_cls", [KineticSynapse, SigmoidRateSynapse])
def test_synapse_step_op_against_autograd(synapse_cls):
    """Values and gradients (of the voltages, states and all parameters, for random
    gradients of all outputs) of the custom op against autograd through the same
    formulas in PyTorch. (`gradcheck` is not precise enough here: the slope is itself a
    finite difference.)"""
    rng = np.random.default_rng(0)
    B, n, k = 3, 6, 4
    E = n * k
    model = synapse_step.KINETIC if synapse_cls is KineticSynapse else synapse_step.SIGMOID_RATE
    pre_np = np.concatenate([rng.choice(np.delete(np.arange(n), j), k, replace=False) for j in range(n)])
    nodes, inv = (torch.as_tensor(x, device="cuda") for x in synapse_step._transposed_index(pre_np))
    pre = torch.as_tensor(pre_np.astype(np.int32), device="cuda")
    targets = torch.arange(n, dtype=torch.int32, device="cuda")
    scalars = synapse_step.kernel_scalars(0.025, 1e-3, torch.float64, "cuda")
    area = torch.full((E,), 600.0, device="cuda")

    def uniform(low, high, size):
        return torch.as_tensor(rng.uniform(low, high, size), device="cuda").requires_grad_(True)

    ranges = list(PARAMS[synapse_cls].values())
    params = [uniform(ranges[0][0] * 1e-4, ranges[0][1] * 1e-4, E)] + [uniform(*r, E) for r in ranges[1:]]
    v = uniform(-70.0, -10.0, (B, n))
    s = uniform(0.0, 1.0, (B, E)) if model == synapse_step.KINETIC else v.new_empty(0)
    padded = params + [params[0]] * (6 - len(params))
    out = synapse_step.synapse_step_op(v, s, pre, targets, *padded, area, scalars, inv, nodes, k, model, True, True)
    out = [x for x in out if x.numel() > 0]
    ref = _reference_step(v, s, pre, k, model, params, area, 0.025, 1e-3)
    inputs = [v] + ([s] if model == synapse_step.KINETIC else []) + params
    upstream = [torch.as_tensor(rng.standard_normal(tuple(x.shape)), device="cuda") for x in ref]
    grads = torch.autograd.grad(sum((g * x).sum() for g, x in zip(upstream, out)), inputs)
    grads_ref = torch.autograd.grad(sum((g * x).sum() for g, x in zip(upstream, ref)), inputs)
    for x, x_ref in zip(out, ref):
        np.testing.assert_allclose(x.detach().cpu(), x_ref.detach().cpu(), rtol=1e-9, atol=1e-12)
    for g, g_ref in zip(grads, grads_ref):
        # The slope (a difference over 1e-3 mV) amplifies the round-off of the voltages'
        # gradients to about 1e-10 of their largest entry.
        np.testing.assert_allclose(g.cpu(), g_ref.cpu(), rtol=1e-9, atol=1e-8 * g_ref.abs().max().item())


@pytest.mark.parametrize("compile", [False, True])
def test_training_iteration_with_kernels_in_cuda_graph(compile):
    """A training iteration with the fused synapse kernels captured in a CUDA graph and
    replayed gives the results of eager execution."""
    net, view = _network(KineticSynapse, trainable=["gS", "e_syn"], record_current=False)

    def make_params():
        return [
            {key: v.cuda().clone().requires_grad_(True) for key, v in p.items()}
            for p in net.get_parameters()
        ]

    def loss_fn(params, amplitude):
        current = jx.datapoint_to_step_currents(0.5, 2.0, amplitude, 0.025, 5.9)
        v = jx.integrate(net, params=params, data_stimuli=view.data_stimulate(current), compile=compile)
        return torch.mean((v - 20.0) ** 2)

    amplitudes = [torch.tensor([a], device="cuda") for a in [0.2, 0.3, 0.4]]
    params = make_params()
    opt = torch.optim.Adam([x for p in params for x in p.values()], lr=1e-3)
    for a in amplitudes:
        opt.zero_grad(set_to_none=False)
        loss_fn(params, a).backward()
        opt.step()
    eager = [x.detach().clone() for p in params for x in p.values()]

    params = make_params()
    leaves = [x for p in params for x in p.values()]
    opt = torch.optim.Adam(leaves, lr=1e-3, capturable=True)
    static_amp = amplitudes[0].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        opt.zero_grad(set_to_none=False)
        loss_fn(params, static_amp).backward()
        opt.step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opt.zero_grad(set_to_none=False)
        loss_fn(params, static_amp).backward()
        opt.step()
    for a in amplitudes[1:]:
        static_amp.copy_(a)
        graph.replay()
    torch.cuda.synchronize()
    for e, g in zip(eager, leaves):
        np.testing.assert_allclose(e.cpu(), g.detach().cpu(), rtol=1e-10, atol=1e-12)
