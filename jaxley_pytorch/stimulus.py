# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>

import torch


def step_current(
    i_delay: float,
    i_dur: float,
    i_amp: float,
    delta_t: float,
    t_max: float,
    i_offset: float = 0.0,
):
    """
    Return step current in unit nA.

    Unlike the `datapoint_to_step()` method, this takes a single value for the amplitude
    and returns a single step current. The output of this function can be passed to
    `.stimulate()`, but not to `integrate(..., currents=)`.

    The returned tensor has the default dtype and lives on the default device. If
    `i_amp` is a tensor (e.g. one that requires grad), the result is differentiable
    with respect to it.
    """
    i_amp = torch.as_tensor(i_amp, dtype=torch.get_default_dtype())
    dt = delta_t
    window_start = int(i_delay / dt)
    window_end = int((i_delay + i_dur) / dt)
    time_steps = int(t_max // dt) + 2
    current = torch.zeros((time_steps,), dtype=i_amp.dtype, device=i_amp.device)
    current = current + i_offset
    mask = torch.zeros((time_steps,), dtype=torch.bool, device=i_amp.device)
    mask[window_start:window_end] = True
    return torch.where(mask, i_amp, current)


def datapoint_to_step_currents(
    i_delay: float,
    i_dur: float,
    i_amp,
    delta_t: float,
    t_max: float,
    i_offset: float = 0.0,
):
    """
    Return several step currents in unit nA.

    Unlike the `step_current()` method, this takes a vector of amplitude and returns
    a step current for each value in the vector. The output of this function can be
    passed to `integrate(..., currents=)`, but can not be passed to `.stimulate()`.

    Returns a tensor of shape `(len(i_amp), time_steps)`. If `i_amp` has leading batch
    dimensions, `(..., dim)`, the output has shape `(..., dim, time_steps)` (this
    replaces `vmap`-ing over `i_amp` in Jaxley).
    """
    if isinstance(i_amp, torch.Tensor) and i_amp.is_floating_point():
        dtype = i_amp.dtype
    else:
        dtype = torch.get_default_dtype()
    i_amp = torch.as_tensor(i_amp, dtype=dtype)
    dt = delta_t
    window_start = int(i_delay / dt)
    window_end = int((i_delay + i_dur) / dt)
    time_steps = int(t_max // dt) + 2
    current = torch.zeros((time_steps,), dtype=i_amp.dtype, device=i_amp.device)
    current = current + i_offset
    mask = torch.zeros((time_steps,), dtype=torch.bool, device=i_amp.device)
    mask[window_start:window_end] = True
    return torch.where(mask, i_amp[..., None], current)
