# This file is part of jaxley-pytorch, a PyTorch port of Jaxley and jaxley-mech.
# Ported from jaxley-mech (https://github.com/jaxleyverse/jaxley-mech, MIT license).
import torch

from jaxley_pytorch.solver_gate import save_exp, x_over_expm1


def efun(x, y):
    # Out of place (in jaxley-mech, `x += 1e-9` rebinds a JAX array). The shift of
    # jaxley-mech is kept (it does not prevent 0/0 in float32, where it is rounded away);
    # `x_over_expm1` handles `x = 0`.
    x = x + 1e-9
    return x_over_expm1(x, y)


def prettify(outputs_array, rec_states, dt):
    """For single compartment model output only."""
    outputs_length = outputs_array.shape[1]
    outputs_dict = {key: outputs_array[i] for (i, key) in enumerate(rec_states)}
    outputs_dict["time"] = torch.arange(0, outputs_length * dt, dt)[:outputs_length]
    return outputs_dict
