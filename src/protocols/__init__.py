"""Protocol registry."""

from src.protocols.base import Protocol
from src.protocols.kernel_evolution import KernelEvolutionProtocol
from src.protocols.quantization import QuantizationProtocol
from src.protocols.interp import InterpProtocol
from src.protocols.scaling_laws import ScalingLawsProtocol
from src.protocols.reward_hacking import RewardHackingProtocol

PROTOCOLS = {
    "kernel": KernelEvolutionProtocol,
    "quantization": QuantizationProtocol,
    "interp": InterpProtocol,
    "scaling": ScalingLawsProtocol,
    "reward_hacking": RewardHackingProtocol,
}


def get_protocol(name: str, **kwargs) -> Protocol:
    if name not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {name}. Available: {list(PROTOCOLS.keys())}")
    return PROTOCOLS[name](**kwargs)
