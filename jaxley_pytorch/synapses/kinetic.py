# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>

from typing import Dict, Optional

import torch

from jaxley_pytorch.synapses.synapse import Synapse


class KineticSynapse(Synapse):
    """First-order kinetic synapse (Destexhe, Mainen & Sejnowski, 1994; not part of
    Jaxley).

    `ds/dt = alpha * T(v_pre) * (1 - s) - beta * s` with the transmitter
    `T(v) = sigmoid((v - v_th) / delta)`, solved with exponential Euler, and the current
    `gS * s * (v_post - e_syn)` (nA for `gS` in uS). Unlike that of `IonotropicSynapse`,
    the time constant `1 / (alpha * T + beta)` stays bounded however steep the release.
    On CUDA, synapses of this type in a regular layout are simulated with fused kernels
    (`jaxley_pytorch.kernels.synapse_step`).
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
            f"{prefix}_e_syn": 0.0,  # mV
            f"{prefix}_alpha": 1.0,  # 1/ms
            f"{prefix}_beta": 0.2,  # 1/ms
            f"{prefix}_v_th": -20.0,  # mV
            f"{prefix}_delta": 2.0,  # mV
        }
        self.synapse_states = {f"{prefix}_s": 0.0}

    def update_states(
        self,
        states: Dict,
        delta_t: float,
        pre_voltage: float,
        post_voltage: float,
        params: Dict,
    ) -> Dict:
        prefix = self._name
        transmitter = torch.sigmoid(
            (pre_voltage - params[f"{prefix}_v_th"]) / params[f"{prefix}_delta"]
        )
        a = params[f"{prefix}_alpha"] * transmitter
        rate = a + params[f"{prefix}_beta"]
        s_inf = a / rate
        s = states[f"{prefix}_s"]
        return {f"{prefix}_s": s_inf + (s - s_inf) * torch.exp(-delta_t * rate)}

    def compute_current(
        self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict
    ) -> float:
        prefix = self._name
        return (
            params[f"{prefix}_gS"]
            * states[f"{prefix}_s"]
            * (post_voltage - params[f"{prefix}_e_syn"])
        )
