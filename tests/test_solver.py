# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>

from tests.helpers import np
import torch
import pytest

import jaxley_pytorch as jx
from jaxley_pytorch.channels import HH
from jaxley_pytorch.connect import connect
from jaxley_pytorch.solver_gate import exponential_euler
from jaxley_pytorch.solver_voltage import DHSSolver, dhs_solve
from jaxley_pytorch.synapses import IonotropicSynapse


@pytest.mark.parametrize("x_inf", [3.0, 30.0])
def test_exp_euler(x_inf):
    """Test whether `dx = -(x - x_inf) / x_tau` is solved correctly."""
    x = 20.0
    x_tau = 12.0
    dt = 0.1
    vectorfield = -(x - x_inf) / x_tau
    fwd_euler = x + dt * vectorfield
    exp_euler = exponential_euler(x, dt, x_inf, x_tau)
    assert np.abs(fwd_euler - exp_euler) / np.abs(fwd_euler) < 1e-4


def test_fwd_euler_and_crank_nicolson(SimpleNet):
    """Compare accuracies of different solvers."""
    net = SimpleNet(2, 3, 4, connect=True)

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    net.cell(0).branch(0).comp(0).stimulate(current)
    net.cell(1).branch(0).comp(3).record()

    net.IonotropicSynapse.set("IonotropicSynapse_gS", 0.01)
    net.insert(HH())

    # As expected, using significantly shorter compartments or lower r_a leads to NaN
    # in forward Euler.
    net.set("length", 50.0)
    net.set("axial_resistivity", 200.0)

    bwd_v = jx.integrate(net, solver="bwd_euler")
    fwd_v = jx.integrate(net, solver="fwd_euler")
    crank_v = jx.integrate(net, solver="crank_nicolson")
    exp_v = jx.integrate(net, solver="exp_euler")

    # These allowed voltage differences may appear large, but due to the steepness
    # of a spike these values actually correspond to sub-millisecond spike time
    # differences.
    assert np.max(np.abs(bwd_v - fwd_v)) < 12.0
    assert np.max(np.abs(bwd_v - crank_v)) < 6.0
    assert np.max(np.abs(bwd_v - exp_v)) < 4.0


def test_exp_euler_solver_customization(SimpleCell):
    cell = SimpleCell(2, 3)
    delta_t = 0.025

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=delta_t, t_max=5.0
    )
    cell.branch(0).comp(0).stimulate(current)
    cell.branch(0).comp(1).record()
    cell.insert(HH())

    cell.customize_solver_exp_euler(
        exp_euler_transition=cell.build_exp_euler_transition_matrix(delta_t)
    )
    v = jx.integrate(cell, solver="exp_euler")
    assert np.invert(np.any(np.isnan(v)))


_RAGGED_INDEXER = {
    "node_order_grouped": [
        np.asarray([[1, 0], [2, 0]], dtype=int),
        np.asarray([[3, 1]], dtype=int),
    ],
    "all_children": np.asarray([1, 2, 3], dtype=int),
    "all_parents": np.asarray([0, 0, 1], dtype=int),
    "parent_lookup": np.asarray([-1, 0, 0, 1, -1], dtype=int),
}


def _ragged_system():
    diags = torch.tensor([4.0, 5.0, 6.0, 7.0, 1.0])
    lowers = torch.tensor([0.0, -0.4, -0.3, -0.2, 0.0])
    uppers = torch.tensor([0.0, -0.5, -0.1, -0.6, 0.0])
    rhs = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0])
    return diags, lowers, uppers, rhs


def _dense_matrix(diags, lowers, uppers):
    """The dense matrix of the tree system (differentiable)."""
    matrix = torch.diag_embed(diags)
    children = torch.as_tensor(_RAGGED_INDEXER["all_children"])
    parents = torch.as_tensor(_RAGGED_INDEXER["all_parents"])
    matrix = matrix.index_put((parents, children), uppers[children], accumulate=True)
    matrix = matrix.index_put((children, parents), lowers[children], accumulate=True)
    return matrix


@pytest.mark.parametrize("optimize_for_gpu", [False, True])
def test_dhs_solve_handles_ragged_grouped_edges(optimize_for_gpu):
    solver = DHSSolver(_RAGGED_INDEXER, optimize_for_gpu=optimize_for_gpu, device="cpu")
    diags, lowers, uppers, rhs = _ragged_system()

    matrix = np.asarray(_dense_matrix(diags, lowers, uppers))
    expected = np.linalg.solve(matrix, np.asarray(rhs))
    actual = np.asarray(dhs_solve(diags, lowers, uppers, rhs, solver))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    rhs_leaf = rhs.clone().requires_grad_(True)
    torch.sum(dhs_solve(diags, lowers, uppers, rhs_leaf, solver)).backward()
    actual_grad = np.asarray(rhs_leaf.grad)
    expected_grad = np.linalg.solve(matrix.T, np.ones_like(expected))
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("optimize_for_gpu", [False, True])
def test_dhs_solve_gradients_match_dense_solve(optimize_for_gpu):
    """Gradients w.r.t. all inputs match autograd through a dense solve."""
    solver = DHSSolver(_RAGGED_INDEXER, optimize_for_gpu=optimize_for_gpu, device="cpu")
    inputs = [x.clone().requires_grad_(True) for x in _ragged_system()]
    weights = torch.tensor([0.3, -1.2, 0.7, 2.0, 0.0])

    loss = torch.sum(weights * dhs_solve(*inputs, solver))
    grads = torch.autograd.grad(loss, inputs)

    dense_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
    matrix = _dense_matrix(*dense_inputs[:3])
    dense_loss = torch.sum(weights * torch.linalg.solve(matrix, dense_inputs[3]))
    dense_grads = torch.autograd.grad(dense_loss, dense_inputs)

    names = ["diags", "lowers", "uppers", "rhs"]
    for name, g, g_dense in zip(names, grads, dense_grads):
        np.testing.assert_allclose(
            np.asarray(g), np.asarray(g_dense), rtol=1e-10, atol=1e-12, err_msg=name
        )


@pytest.mark.parametrize("optimize_for_gpu", [False, True])
def test_dhs_solve_gradcheck(optimize_for_gpu):
    """Finite-difference check of the backward pass and of its derivative."""
    solver = DHSSolver(_RAGGED_INDEXER, optimize_for_gpu=optimize_for_gpu, device="cpu")
    inputs = tuple(x.clone().requires_grad_(True) for x in _ragged_system())

    def fn(d, l, u, b):
        return dhs_solve(d, l, u, b, solver)

    assert torch.autograd.gradcheck(fn, inputs)
    assert torch.autograd.gradgradcheck(fn, inputs)


def test_dhs_solve_batched():
    """Leading batch dimensions are solved independently."""
    solver = DHSSolver(_RAGGED_INDEXER, optimize_for_gpu=True, device="cpu")
    diags, lowers, uppers, rhs = _ragged_system()
    scale = torch.tensor([1.0, 2.0, 0.5])[:, None]
    batched = dhs_solve(diags * scale, lowers * scale, uppers, rhs * scale, solver)
    for i in range(3):
        single = dhs_solve(
            diags * scale[i], lowers * scale[i], uppers, rhs * scale[i], solver
        )
        np.testing.assert_allclose(np.asarray(batched[i]), np.asarray(single))


@pytest.mark.parametrize("optimize_for_gpu", [False, True])
def test_dhs_solve_jvp_matches_dense_solve(optimize_for_gpu):
    """Forward-mode derivatives (`torch.func.jvp`) w.r.t. all inputs match a dense
    solve, and pass a finite-difference check."""
    solver = DHSSolver(_RAGGED_INDEXER, optimize_for_gpu=optimize_for_gpu, device="cpu")
    primals = _ragged_system()
    generator = torch.Generator().manual_seed(0)
    tangents = tuple(torch.randn(x.shape, generator=generator) for x in primals)

    def fn(d, l, u, b):
        return dhs_solve(d, l, u, b, solver)

    def dense(d, l, u, b):
        return torch.linalg.solve(_dense_matrix(d, l, u), b)

    out, tangent = torch.func.jvp(fn, primals, tangents)
    out_dense, tangent_dense = torch.func.jvp(dense, primals, tangents)
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_dense), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(tangent), np.asarray(tangent_dense), rtol=1e-10, atol=1e-12
    )

    inputs = tuple(x.clone().requires_grad_(True) for x in primals)
    assert torch.autograd.gradcheck(fn, inputs, check_forward_ad=True)


@pytest.mark.parametrize("api", ["func", "forward_ad"])
def test_jvp_through_simulation_matches_vjp(api):
    """Forward mode through `jx.integrate` agrees with reverse mode:
    w^T (J q) == (J^T w)^T q."""
    torch.set_default_dtype(torch.float64)
    comp = jx.Compartment()
    branch = jx.Branch(comp, 3)
    cell = jx.Cell([branch] * 3, parents=[-1, 0, 0])
    cell.insert(HH())
    cell.branch(1).comp(2).stimulate(jx.step_current(0.5, 1.0, 0.3, 0.025, 4.0))
    cell.branch(0).comp(0).record(verbose=False)
    cell.branch(2).comp(1).record(verbose=False)
    theta = torch.tensor([0.12, 0.10, 0.14])

    def simulate(gna):
        pstate = None
        for b in range(3):
            pstate = cell.branch(b).data_set("HH_gNa", gna[b], pstate)
        return jx.integrate(cell, param_state=pstate)

    generator = torch.Generator().manual_seed(1)
    q = torch.randn(3, generator=generator)
    if api == "func":
        v, jq = torch.func.jvp(simulate, (theta,), (q,))
    else:
        import torch.autograd.forward_ad as fwAD

        with fwAD.dual_level():
            dual = simulate(fwAD.make_dual(theta, q))
            v, jq = fwAD.unpack_dual(dual)
    w = torch.randn(v.shape, generator=generator)

    leaf = theta.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(torch.sum(w * simulate(leaf)), leaf)
    np.testing.assert_allclose(float(torch.sum(w * jq)), float(torch.sum(g * q)), rtol=1e-9)


def test_dhs_network_with_trailing_single_compartment_cells():
    """Single-compartment cells at the end of a network (e.g. passive readouts): the
    DHS solve matches the dense solve. (Jaxley's indexer points one past the last
    lower/upper for them and relies on JAX clamping out-of-bounds gathers.)"""
    torch.set_default_dtype(torch.float64)
    comp = jx.Compartment()
    branch = jx.Branch(comp, 1)
    cell = jx.Cell([branch] * 3, parents=[-1, 0, 0])
    point = jx.Cell([branch], parents=[-1])
    net = jx.Network([cell, cell, point, point])
    net.insert(HH())
    connect(net.cell(0).branch(1).comp(0), net.cell(2).branch(0).comp(0), IonotropicSynapse())
    connect(net.cell(1).branch(1).comp(0), net.cell(3).branch(0).comp(0), IonotropicSynapse())
    net.cell(0).branch(2).comp(0).stimulate(jx.step_current(0.5, 1.0, 0.3, 0.025, 5.0))
    for i in range(4):
        net.cell(i).branch(0).comp(0).record(verbose=False)
    v_dhs = jx.integrate(net, voltage_solver="jaxley.dhs")
    v_dense = jx.integrate(net, voltage_solver="jax.sparse")
    np.testing.assert_allclose(np.asarray(v_dhs), np.asarray(v_dense), rtol=1e-8, atol=1e-8)
