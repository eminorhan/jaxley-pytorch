# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
import torch
from torch import Tensor


def save_exp(x, max_value: float = 20.0):
    """Clip the input to a maximum value and return its exponential."""
    x = torch.clamp(torch.as_tensor(x), max=max_value)
    return torch.exp(x)


def x_over_expm1(x, y=1.0):
    """`x / (exp(x / y) - 1)` with the exponent clipped at 20 (as `save_exp`).

    Many gating rates have this form (Jaxley evaluates it as `x / (save_exp(x / y) - 1)`).
    It is 0/0 at `x = 0`, where its limit is `y`, and `exp(z) - 1` cancels near it. In
    float32, voltages land exactly on such a point now and then, and the NaN then
    spreads through a network. Here, the value near `x = 0` comes from the series
    `y * (1 - z / 2 + z**2 / 12)` with `z = x / y` (exact to O(z**4)), elsewhere from
    `expm1`. The masked branch never sees `z = 0`, so the gradient is finite as well.
    Not part of Jaxley.
    """
    x = torch.as_tensor(x)
    z = torch.clamp(x / y, max=20.0)
    small = torch.abs(z) < 1e-4
    z_safe = torch.where(small, torch.ones_like(z), z)
    series = y * (1.0 - z / 2.0 + z * z / 12.0)
    return torch.where(small, series, x / torch.expm1(z_safe))


def solve_gate_implicit(
    gating_state: Tensor,
    dt: float,
    alpha: Tensor,
    beta: Tensor,
):
    a_m = gating_state + dt * alpha
    b_m = 1.0 + dt * alpha + dt * beta

    return a_m / b_m


def solve_gate_exponential(
    x: Tensor,
    dt: float,
    alpha: Tensor,
    beta: Tensor,
):
    tau = 1 / (alpha + beta)
    xinf = alpha * tau
    return exponential_euler(x, dt, xinf, tau)


def exponential_euler(
    x: Tensor,
    dt: float,
    x_inf: Tensor,
    x_tau: Tensor,
):
    """An exact solver for the linear dynamical system `dx = -(x - x_inf) / x_tau`."""
    exp_term = save_exp(-dt / x_tau)
    return x * exp_term + x_inf * (1.0 - exp_term)


def solve_inf_gate_exponential(
    x: Tensor,
    dt: float,
    s_inf: Tensor,
    tau_s: Tensor,
):
    """solves dx/dt = (s_inf - x) / tau_s
    via exponential Euler

    Args:
        x (Tensor): gate variable
        dt (float): time_delta
        s_inf (Tensor): _description_
        tau_s (Tensor): _description_

    Returns:
        _type_: updated gate
    """
    slope = -1.0 / tau_s
    exp_term = save_exp(slope * dt)
    return x * exp_term + s_inf * (1.0 - exp_term)
