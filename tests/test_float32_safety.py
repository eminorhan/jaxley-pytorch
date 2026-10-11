# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Gating rates and `IonotropicSynapse` in float32 (port-specific).

Jaxley evaluates rates of the form `x / (exp(x) - 1)` directly, which is 0/0 at their
removable singularity, and the time constant of `IonotropicSynapse` as
`(1 - s_inf) / k_minus`, which is 0 once a steep release saturates. In float32 both
happen in ordinary simulations (and give NaN); `x_over_expm1` and the rewritten
synapse update avoid them without changing the float64 results.
"""
import math

import pytest
import torch

import jaxley_pytorch as jx
from jaxley_pytorch.channels import HH, CaL, K, Na
from jaxley_pytorch.connect import connect
from jaxley_pytorch.mech.channels import fohlmeister97 as fm
from jaxley_pytorch.solver_gate import save_exp, x_over_expm1
from jaxley_pytorch.synapses import IonotropicSynapse


def _reference(x: float, y: float) -> float:
    """`x / (exp(x / y) - 1)` with the exponent clipped at 20, in float64."""
    z = min(x / y, 20.0)
    return y if z == 0.0 else x / math.expm1(z)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("y", [1.0, 10.0])
def test_x_over_expm1(dtype, y):
    xs = [0.0, 1e-9, -1e-9, 1e-6, -1e-6, 3e-4 * y, -3e-4 * y, 1e-2, -0.5, 2.0, 25.0 * y, -40.0 * y]
    x = torch.tensor(xs, dtype=dtype, requires_grad=True)
    out = x_over_expm1(x, y)
    ref = torch.tensor([_reference(float(xi), y) for xi in x.detach().double()], dtype=torch.float64)
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    assert torch.allclose(out.double(), ref, rtol=tol, atol=tol * y)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert abs(x.grad[0].item() + 0.5) < 1e-6  # d/dx at x = 0 is -1/2.
    # Away from the switch between the two branches, the gradient is exact.
    x64 = torch.tensor([0.3, -2.0, 5.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: x_over_expm1(t, y), (x64,))


# (rate function, voltage at its removable singularity, rate there)
SINGULAR_RATES = [
    ("HH alpha_m", lambda v: HH.m_gate(v)[0], -40.0, 1.0),
    ("HH alpha_n", lambda v: HH.n_gate(v)[0], -55.0, 0.1),
    ("Pospischil Na alpha_m", lambda v: Na.m_gate(v, -60.0)[0], -47.0, 1.28),
    ("Pospischil Na beta_m", lambda v: Na.m_gate(v, -60.0)[1], -20.0, 1.4),
    ("Pospischil K alpha_n", lambda v: K.n_gate(v, -60.0)[0], -45.0, 0.16),
    ("Pospischil CaL alpha_q", lambda v: CaL.q_gate(v)[0], -27.0, 0.209),
    ("fohlmeister97 Na alpha_m", lambda v: fm.Na.m_gate(v)[0], -30.0, 6.0),
    ("fohlmeister97 K alpha_n", lambda v: fm.K.n_gate(v)[0], -40.0, 0.2),
    ("fohlmeister97 KA alpha_A", lambda v: fm.KA.A_gate(v)[0], -90.0, 0.006),
    ("fohlmeister97 Ca alpha_c", lambda v: fm.Ca.c_gate(v)[0], -13.0, 3.0),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name,rate,v0,expected", SINGULAR_RATES, ids=[r[0] for r in SINGULAR_RATES])
def test_rates_at_removable_singularities(name, rate, v0, expected, dtype):
    for dv in [0.0, 1e-4, -1e-4]:
        v = torch.tensor(v0 + dv, dtype=dtype, requires_grad=True)
        r = rate(v)
        assert torch.isfinite(r), (name, dv)
        assert abs(r.item() - expected) < 1e-3 * expected + 1e-6, (name, dv, r.item())
        r.backward()
        assert torch.isfinite(v.grad), (name, dv)


@pytest.fixture
def float32_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


def test_hh_cell_at_singular_voltage_float32(float32_default):
    """A cell that starts exactly at -40 mV (HH's m gate) stays finite in float32."""
    cell = jx.Cell()
    cell.insert(HH())
    cell.set("v", -40.0)
    cell.init_states()
    cell.record("v", verbose=False)
    cell.stimulate(torch.zeros(41), verbose=False)
    v = jx.integrate(cell, delta_t=0.025)
    assert v.dtype == torch.float32
    assert torch.isfinite(v).all()


def _original_ionotropic_update(s, v_pre, v_th, delta, k_minus, dt):
    """`IonotropicSynapse.update_states` as Jaxley computes it."""
    s_inf = 1.0 / (1.0 + save_exp((v_th - v_pre) / delta))
    tau_s = (1.0 - s_inf) / k_minus
    exp_term = save_exp(-1.0 / tau_s * dt)
    return s * exp_term + s_inf * (1.0 - exp_term)


def _update(dtype, v_pre, s, delta, k_minus=0.1, dt=0.025):
    syn = IonotropicSynapse()
    p = syn._name
    params = {
        f"{p}_v_th": torch.tensor(-20.0, dtype=dtype),
        f"{p}_delta": torch.tensor(delta, dtype=dtype),
        f"{p}_k_minus": torch.as_tensor(k_minus, dtype=dtype),
    }
    return syn.update_states({f"{p}_s": s}, dt, v_pre, None, params)[f"{p}_s"]


@pytest.mark.parametrize("delta", [10.0, 2.0])
def test_ionotropic_update_matches_jaxley_in_float64(delta):
    v_pre = torch.linspace(-250.0, 60.0 if delta == 10.0 else 25.0, 621, dtype=torch.float64)
    s = torch.linspace(0.0, 1.0, 621, dtype=torch.float64)
    new = _update(torch.float64, v_pre, s, delta)
    old = _original_ionotropic_update(s, v_pre, -20.0, delta, 0.1, 0.025)
    assert torch.allclose(new, old, rtol=1e-12, atol=1e-14)


def test_ionotropic_saturating_release_float32():
    v_pre = torch.linspace(-150.0, 120.0, 271, dtype=torch.float32, requires_grad=True)
    s = torch.full((271,), 0.3, dtype=torch.float32, requires_grad=True)
    k_minus = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
    # Jaxley's formula: s_inf rounds to 1 during a spike, the gradient is NaN.
    old = _original_ionotropic_update(s, v_pre, -20.0, 2.0, k_minus, 0.025)
    g_old = torch.autograd.grad(old.sum(), v_pre)[0]
    assert not torch.isfinite(g_old).all()
    new = _update(torch.float32, v_pre, s, 2.0, k_minus=k_minus)
    assert torch.isfinite(new).all()
    new.sum().backward()
    for t in (v_pre, s, k_minus):
        assert torch.isfinite(t.grad).all()


def test_float32_network_gradient_with_steep_ionotropic_synapse():
    """A presynaptic spike through a steep `IonotropicSynapse` gives finite gradients."""
    cell = jx.Cell()
    cell.insert(HH())
    net = jx.Network([cell, cell])
    connect(net.cell(0).branch(0).comp(0), net.cell(1).branch(0).comp(0), IonotropicSynapse())
    net.IonotropicSynapse.edge("all").set("IonotropicSynapse_delta", 2.0)
    net.IonotropicSynapse.edge("all").set("IonotropicSynapse_v_th", -20.0)
    net.IonotropicSynapse.edge("all").make_trainable("IonotropicSynapse_gS", verbose=False)
    net.cell(0).branch(0).comp(0).stimulate(
        torch.as_tensor(jx.step_current(1.0, 2.0, 0.5, 0.025, 10.0), dtype=torch.float32), verbose=False
    )
    net.cell("all").branch(0).comp(0).record("v", verbose=False)
    params = [{k: v.to(torch.float32).requires_grad_(True) for k, v in p.items()} for p in net.get_parameters()]
    v = jx.integrate(net, params=params, delta_t=0.025)
    assert v.dtype == torch.float32
    assert v[0].max() > 20.0  # The presynaptic cell spiked (saturating the release).
    v[1].mean().backward()
    assert torch.isfinite(params[0]["IonotropicSynapse_gS"].grad).all()
