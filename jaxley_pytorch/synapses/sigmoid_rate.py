# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>

from typing import Dict, Optional

import torch

from jaxley_pytorch.synapses.synapse import Synapse


class SigmoidRateSynapse(Synapse):
    """Current-based rate synapse, `current = -gS * sigmoid((v_pre - x_offset) * slope)`
    (not part of Jaxley; no state).

    The presynaptic rate is non-negative, so the sign of `gS` alone sets the sign of the
    synapse: `gS > 0` depolarizes, `gS < 0` hyperpolarizes (unlike `TanhRateSynapse`,
    whose tanh turns an excitatory synapse inhibitory below the offset). As in
    `TanhRateSynapse`, `gS` is a current (nA). On CUDA, synapses of this type in a regular
    layout are simulated with fused kernels (`jaxley_pytorch.kernels.synapse_step`).
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,
            f"{prefix}_x_offset": -60.0,
            f"{prefix}_slope": 0.25,
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
        rate = torch.sigmoid(
            (pre_voltage - params[f"{prefix}_x_offset"]) * params[f"{prefix}_slope"]
        )
        return -params[f"{prefix}_gS"] * rate
