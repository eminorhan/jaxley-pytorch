# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>

from typing import Dict, Optional

import torch.nn.functional as F

from jaxley_pytorch.synapses.synapse import Synapse


class SoftplusRateSynapse(Synapse):
    """Current-based rate synapse,
    `current = -gS * softplus((v_pre - x_offset) * slope)` (not part of Jaxley; no
    state).

    `gS > 0` depolarizes. With `x_offset = E_L - S * b` and `slope = 1 / S`, the
    presynaptic rate is `softplus(x + b)` with `x = (v - E_L) / S`: the rate neuron of
    Beiran & Litwin-Kumar (2025), see `jaxley_pytorch.training.rate_network`. As in
    `TanhRateSynapse`, `gS` is a current (nA).
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 0.0,
            f"{prefix}_x_offset": -70.0,
            f"{prefix}_slope": 0.1,
        }
        self.synapse_states = {}

    def update_states(
        self,
        states: Dict,
        delta_t: float,
        pre_voltage: float,
        post_voltage: float,
        params: Dict,
    ) -> Dict:
        return {}

    def compute_current(
        self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict
    ) -> float:
        prefix = self._name
        rate = F.softplus(
            (pre_voltage - params[f"{prefix}_x_offset"]) * params[f"{prefix}_slope"]
        )
        return -params[f"{prefix}_gS"] * rate
