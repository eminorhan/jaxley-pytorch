# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from jaxley_pytorch.utils.torch_utils import as_index


def regular_layout(post_syn_comp_inds) -> Optional[Tuple[int, np.ndarray]]:
    """`(k, targets)` if the synapses are ordered by postsynaptic compartment with `k`
    synapses per compartment, else `None` (not part of Jaxley).

    `targets` are the postsynaptic compartments, in increasing order (the synapses of
    `targets[i]` are `i * k` to `(i + 1) * k - 1`).
    """
    post = np.asarray(post_syn_comp_inds)
    if post.ndim != 1 or len(post) == 0:
        return None
    starts = np.flatnonzero(np.r_[True, post[1:] != post[:-1]])
    counts = np.diff(np.r_[starts, len(post)])
    if np.any(counts != counts[0]) or np.any(np.diff(post[starts]) <= 0):
        return None
    return int(counts[0]), post[starts]


def gather_synapes(
    number_of_compartments: int,
    post_syn_comp_inds,
    current_each_synapse_voltage_term: Tensor,
    current_each_synapse_constant_term: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute current at the post synapse.

    All this does it that it sums the synaptic currents that come into a particular
    compartment. It returns an array of as many elements as there are compartments
    (with the leading batch dimensions of the synaptic currents).
    """
    voltage_term = current_each_synapse_voltage_term
    constant_term = current_each_synapse_constant_term
    inds = as_index(post_syn_comp_inds, voltage_term.device)

    batch_v = voltage_term.shape[:-1]
    incoming_currents_voltages = torch.zeros(
        (*batch_v, number_of_compartments),
        dtype=voltage_term.dtype,
        device=voltage_term.device,
    ).index_add(-1, inds, voltage_term)

    batch_c = constant_term.shape[:-1]
    incoming_currents_content = torch.zeros(
        (*batch_c, number_of_compartments),
        dtype=constant_term.dtype,
        device=constant_term.device,
    ).index_add(-1, inds, constant_term)
    return incoming_currents_voltages, incoming_currents_content
