# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Tests of the fused GPU tree solve (not part of Jaxley)."""

import os

import pytest
import torch

import jaxley_pytorch as jx
from jaxley_pytorch.channels import HH
from jaxley_pytorch.connect import fully_connect
from jaxley_pytorch.kernels.tree_solve import HAS_TRITON
from jaxley_pytorch.solver_voltage import DHSSolver, dhs_solve
from jaxley_pytorch.synapses import IonotropicSynapse
from tests.helpers import np

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason="Needs a GPU and Triton."
)
SWC = os.path.join(os.path.dirname(__file__), "swc_files")


def _modules():
    ca1 = jx.read_swc(os.path.join(SWC, "morph_ca1_n120.swc"), ncomp=4)
    comp = jx.Compartment()
    branch = jx.Branch(comp, 3)
    small = jx.Cell([branch] * 5, parents=[-1, 0, 0, 1, 1])
    # A node with many children (8 branches emerging from one branchpoint).
    star = jx.Cell([branch] * 9, parents=[-1] + [0] * 8)
    net = jx.Network([small, ca1, jx.Cell(), star, small])
    return {"ca1": ca1, "network": net, "star": star, "point": jx.Cell()}


def _random_system(n, batch, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    lowers = -torch.rand(*batch, n, generator=g, dtype=dtype)
    uppers = -torch.rand(*batch, n, generator=g, dtype=dtype)
    diags = 3.0 + torch.rand(*batch, n, generator=g, dtype=dtype)
    solves = torch.randn(*batch, n, generator=g, dtype=dtype)
    return diags, lowers, uppers, solves


def _system_size(module):
    return len(module._dhs_solve_indexer["parent_lookup"])


@pytest.fixture(params=["registers", "levels"])
def kernel_mode(request, monkeypatch):
    """Run a test with each of the two fused kernels."""
    monkeypatch.setenv("JAXLEY_PYTORCH_TREE_KERNEL", request.param)
    return request.param


def _check_mode(solver, mode):
    from jaxley_pytorch.kernels.tree_solve import LevelTreeSolveKernel, TreeSolveKernel

    expected = TreeSolveKernel if mode == "registers" else LevelTreeSolveKernel
    assert isinstance(solver.kernel, expected)


@requires_gpu
@pytest.mark.parametrize("name", ["ca1", "network", "star", "point"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_solve_matches_torch_solve(name, dtype, kernel_mode):
    module = _modules()[name]
    if not isinstance(module, jx.Network):
        module._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=32)
    indexer = module._dhs_solve_indexer
    fused = DHSSolver(indexer, optimize_for_gpu=True, device="cuda")
    reference = DHSSolver(indexer, optimize_for_gpu=True, device="cuda", use_fused_kernel=False)
    if _system_size(module) > 1:
        _check_mode(fused, kernel_mode)

    inputs = [x.cuda() for x in _random_system(_system_size(module), (3,), dtype)]
    x_fused = fused.raw_solve(*inputs)
    x_ref = reference.raw_solve(*inputs)
    tol = 1e-12 if dtype == torch.float64 else 1e-4
    np.testing.assert_allclose(x_fused.cpu(), x_ref.cpu(), rtol=tol, atol=tol)


@requires_gpu
def test_fused_solve_gradients(kernel_mode):
    module = _modules()["network"]
    if not isinstance(module, jx.Network):
        module._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=32)
    indexer = module._dhs_solve_indexer
    fused = DHSSolver(indexer, optimize_for_gpu=True, device="cuda")
    reference = DHSSolver(indexer, optimize_for_gpu=True, device="cuda", use_fused_kernel=False)
    n = _system_size(module)
    inputs = _random_system(n, (2,), torch.float64)
    weights = torch.randn(2, n, dtype=torch.float64).cuda()

    grads = []
    for solver in [fused, reference]:
        leaves = [x.cuda().requires_grad_(True) for x in inputs]
        loss = torch.sum(weights * dhs_solve(*leaves, solver))
        grads.append(torch.autograd.grad(loss, leaves))
    for g_fused, g_ref in zip(*grads):
        np.testing.assert_allclose(g_fused.cpu(), g_ref.cpu(), rtol=1e-10, atol=1e-12)


@requires_gpu
def test_fused_solve_gradcheck(kernel_mode):
    module = _modules()["star"]
    if not isinstance(module, jx.Network):
        module._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=32)
    fused = DHSSolver(module._dhs_solve_indexer, optimize_for_gpu=True, device="cuda")
    n = _system_size(module)
    inputs = tuple(
        x.cuda().requires_grad_(True) for x in _random_system(n, (), torch.float64)
    )
    assert torch.autograd.gradcheck(lambda *a: dhs_solve(*a, fused), inputs)


@requires_gpu
def test_simulation_with_and_without_kernel(kernel_mode):
    """Full simulation on GPU: fused kernel vs. level-by-level torch solve."""
    torch.set_default_dtype(torch.float64)
    cell = jx.read_swc(os.path.join(SWC, "morph_ca1_n120.swc"), ncomp=4)
    cell.insert(HH())
    cell.branch(0).loc(0.0).stimulate(
        jx.step_current(1.0, 2.0, 0.5, 0.025, 10.0).cuda(), verbose=False
    )
    for b in [0, 10, 40]:
        cell.branch(b).loc(0.5).record(verbose=False)
    cell._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=32)
    v_fused = jx.integrate(cell, voltage_solver="jaxley.dhs.gpu")
    # Reference: same simulation on CPU (torch level-by-level solve).
    cell.delete_stimuli()
    cell.branch(0).loc(0.0).stimulate(
        jx.step_current(1.0, 2.0, 0.5, 0.025, 10.0), verbose=False
    )
    v_ref = jx.integrate(cell, voltage_solver="jaxley.dhs.gpu")
    np.testing.assert_allclose(v_fused.cpu(), v_ref, rtol=1e-9, atol=1e-9)


@requires_gpu
def test_compiled_step_matches_eager():
    """`jx.integrate(..., compile=True)` gives the same voltages and gradients."""
    torch.set_default_dtype(torch.float64)
    from jaxley_pytorch.synapses import IonotropicSynapse

    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell([branch] * 3, parents=[-1, 0, 0])
    net = jx.Network([cell] * 3)
    net.insert(HH())
    jx.fully_connect(net.cell([0, 1]), net.cell([2]), IonotropicSynapse())
    net.set("IonotropicSynapse_gS", 0.01)
    net.cell([0, 1]).branch(0).comp(0).stimulate(
        jx.step_current(1.0, 2.0, 0.3, 0.025, 10.0).cuda(), verbose=False
    )
    net.cell(2).branch(1).comp(3).record(verbose=False)
    net.cell(0).branch(0).comp(0).record("HH_m", verbose=False)
    net.make_trainable("IonotropicSynapse_gS", verbose=False)
    net.cell(2).make_trainable("radius", verbose=False)

    results = []
    for compile in [False, True]:
        params = [
            {k: v.cuda().requires_grad_(True) for k, v in p.items()}
            for p in net.get_parameters()
        ]
        v = jx.integrate(net, params=params, compile=compile, checkpoint_lengths=[20, 21])
        grads = torch.autograd.grad(torch.sum(v[0]), [x for p in params for x in p.values()])
        results.append((v.detach().cpu(), [g.cpu() for g in grads]))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-10, atol=1e-10)
    for g_eager, g_compiled in zip(results[0][1], results[1][1]):
        np.testing.assert_allclose(g_eager, g_compiled, rtol=1e-8, atol=1e-10)


def _graph_test_cell():
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell([branch] * 3, parents=[-1, 0, 0])
    cell.insert(HH())
    cell.branch(0).comp(0).record(verbose=False)
    cell.branch("all").make_trainable("HH_gNa", verbose=False)
    cell.make_trainable("radius", verbose=False)
    return cell, cell.branch(1).comp(3)


def _graph_test_network():
    """Two cells with calcium mechanisms whose `compute_current` returns a constant,
    connected by ionotropic synapses with trainable conductances."""
    from jaxley_pytorch.mech.channels import l5pc

    comp = jx.Compartment()
    branch = jx.Branch(comp, 2)
    cell = jx.Cell([branch] * 3, parents=[-1, 0, 0])
    net = jx.Network([cell, cell])
    net.insert(HH())
    net.insert(l5pc.CaHVA())
    net.insert(l5pc.CaPump())
    net.insert(l5pc.CaNernstReversal())
    fully_connect(net.cell([0]), net.cell([1]), IonotropicSynapse())
    fully_connect(net.cell([1]), net.cell([0]), IonotropicSynapse())
    net.cell(1).branch(0).comp(0).record(verbose=False)
    net.IonotropicSynapse.edge("all").make_trainable("IonotropicSynapse_gS", verbose=False)
    net.cell("all").make_trainable("HH_gK", verbose=False)
    return net, net.cell(0).branch(1).comp(1)


@requires_gpu
@pytest.mark.parametrize("model", ["cell", "network"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_training_iteration_in_cuda_graph(checkpointing, model):
    """A whole training iteration (simulation, loss, backward, optimizer step) can be
    captured in a CUDA graph and replayed, with the same result as eager execution."""
    torch.set_default_dtype(torch.float64)
    module, stim_view = _graph_test_cell() if model == "cell" else _graph_test_network()
    kwargs = dict(checkpoint_lengths=[16, 15]) if checkpointing else {}

    def make_params():
        return [
            {k: v.cuda().clone().requires_grad_(True) for k, v in p.items()}
            for p in module.get_parameters()
        ]

    def loss_fn(params, amplitude):
        current = jx.datapoint_to_step_currents(0.5, 2.0, amplitude, 0.025, 5.9)
        stimuli = stim_view.data_stimulate(current)
        v = jx.integrate(module, params=params, data_stimuli=stimuli, **kwargs)
        return torch.mean((v - 20.0) ** 2)

    amplitudes = [torch.tensor([a], device="cuda") for a in [0.2, 0.3, 0.4, 0.5]]

    # Eager reference.
    params = make_params()
    opt = torch.optim.Adam([x for p in params for x in p.values()], lr=1e-3)
    for a in amplitudes:
        opt.zero_grad(set_to_none=False)
        loss_fn(params, a).backward()
        opt.step()
    eager = [x.detach().clone() for p in params for x in p.values()]

    # CUDA graph: warm up on a side stream (first iteration), then capture the second.
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
        static_loss = loss_fn(params, static_amp)
        static_loss.backward()
        opt.step()
    for a in amplitudes[1:]:
        static_amp.copy_(a)
        graph.replay()
    torch.cuda.synchronize()
    graphed = [x.detach().clone() for x in leaves]
    for e, g in zip(eager, graphed):
        np.testing.assert_allclose(e.cpu(), g.cpu(), rtol=1e-12, atol=1e-12)


@requires_gpu
def test_jvp_through_simulation_on_gpu():
    """`torch.func.jvp` through `jx.integrate` on CUDA (the raw solves use the fused
    kernel) agrees with reverse mode and with the CPU result."""
    torch.set_default_dtype(torch.float64)
    comp = jx.Compartment()
    branch = jx.Branch(comp, 3)
    cell = jx.Cell([branch] * 3, parents=[-1, 0, 0])
    cell.insert(HH())
    cell.branch(0).comp(0).record(verbose=False)
    cell.branch(2).comp(1).record(verbose=False)

    def simulate(gna):
        current = jx.step_current(0.5, 1.0, torch.tensor(0.3, device=gna.device), 0.025, 4.0)
        stimuli = cell.branch(1).comp(2).data_stimulate(current)
        pstate = None
        for b in range(3):
            pstate = cell.branch(b).data_set("HH_gNa", gna[b], pstate)
        return jx.integrate(cell, param_state=pstate, data_stimuli=stimuli)

    theta = torch.tensor([0.12, 0.10, 0.14])
    q = torch.tensor([0.3, -1.1, 0.6])
    v_cpu, jq_cpu = torch.func.jvp(simulate, (theta,), (q,))
    v, jq = torch.func.jvp(simulate, (theta.cuda(),), (q.cuda(),))
    np.testing.assert_allclose(v.cpu(), v_cpu, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(jq.cpu(), jq_cpu, rtol=1e-7, atol=1e-7)

    w = torch.randn(v.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    leaf = theta.cuda().requires_grad_(True)
    (g,) = torch.autograd.grad(torch.sum(w * simulate(leaf)), leaf)
    np.testing.assert_allclose(float(torch.sum(w * jq)), float(torch.sum(g * q.cuda())), rtol=1e-9)



@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_level_kernel_on_deep_trees(dtype):
    """A deep tree (a long branched cable, too deep for the register kernel to be
    efficient) next to small trees: the level kernel matches the torch solve, and the
    automatic choice picks it."""
    comp = jx.Compartment()
    long = jx.Branch(comp, 500)
    short = jx.Branch(comp, 3)
    deep = jx.Cell([long, long, short, long, long], parents=[-1, 0, 0, 1, 3])
    small = jx.Cell([short] * 3, parents=[-1, 0, 0])
    net = jx.Network([small, deep, small])
    indexer = net._dhs_solve_indexer
    fused = DHSSolver(indexer, optimize_for_gpu=True, device="cuda")
    _check_mode(fused, "levels")
    reference = DHSSolver(indexer, optimize_for_gpu=True, device="cuda", use_fused_kernel=False)
    inputs = [x.cuda() for x in _random_system(_system_size(net), (4,), dtype)]
    tol = 1e-10 if dtype == torch.float64 else 1e-3
    np.testing.assert_allclose(fused.raw_solve(*inputs).cpu(), reference.raw_solve(*inputs).cpu(), rtol=tol, atol=tol)
