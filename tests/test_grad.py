# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>



from tests.helpers import np
import torch
import pytest
from tests.helpers import value_and_grad

import jaxley_pytorch as jx
from jaxley_pytorch.channels import HH


@pytest.mark.parametrize("key", ["HH_m", "v"])
def test_grad_against_finite_diff_initial_state(key, SimpleComp):
    def simulate():
        return np.sum(jx.integrate(comp))

    def simulate_with_params(params):
        return torch.sum(jx.integrate(comp, params=params))

    comp = SimpleComp(copy=True)
    comp.insert(HH())
    comp.record()
    comp.stimulate(
        jx.step_current(i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0)
    )

    val = 0.2 if key == "HH_m" else -70.0
    step_size = 0.01

    # Finite differences.
    comp.set(key, val)
    r1 = simulate()
    comp.set(key, val + step_size)
    r2 = simulate()
    finitediff_grad = (r2 - r1) / step_size

    # Autodiff gradient.
    grad_fn = value_and_grad(simulate_with_params)
    comp.set(key, val)
    comp.make_trainable(key)
    params = comp.get_parameters()
    v, g = grad_fn(params)
    autodiff_grad = np.asarray(g[0][key])

    # Less than 5% error on the gradient difference.
    assert np.abs(autodiff_grad - finitediff_grad) / autodiff_grad < 0.05


@pytest.mark.parametrize("key", ["HH_m", "v"])
def test_branch_grad_against_finite_diff_initial_state(key, SimpleBranch):
    def simulate():
        return np.sum(jx.integrate(branch))

    def simulate_with_params(params):
        return torch.sum(jx.integrate(branch, params=params))

    branch = SimpleBranch(4)
    branch.loc(0.0).record()
    branch.loc(0.0).stimulate(
        jx.step_current(i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0)
    )
    branch.loc(0.0).insert(HH())

    val = 0.2 if key == "HH_m" else -70.0
    step_size = 0.01

    # Finite differences.
    branch.loc(0.0).set(key, val)
    r1 = simulate()
    branch.loc(0.0).set(key, val + step_size)
    r2 = simulate()
    finitediff_grad = (r2 - r1) / step_size

    # Autodiff gradient.
    grad_fn = value_and_grad(simulate_with_params)
    branch.loc(0.0).set(key, val)
    branch.loc(0.0).make_trainable(key)
    params = branch.get_parameters()
    v, g = grad_fn(params)
    autodiff_grad = np.asarray(g[0][key])

    # Less than 6% error on the gradient difference, error for HH_m is around 5.2%.
    assert np.abs(autodiff_grad - finitediff_grad) / autodiff_grad < 0.06
