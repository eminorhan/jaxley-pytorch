# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>

import torch

from jaxley_pytorch.utils.torch_utils import tree_leaves


def l2_norm(x: "PyTree") -> torch.Tensor:
    """Return the L2-norm of a pytree (nested lists, tuples and dicts of tensors)."""
    leaves = [leaf for leaf in tree_leaves(x) if leaf is not None]
    return torch.sqrt(sum([torch.sum(torch.as_tensor(leaf) ** 2) for leaf in leaves]))
