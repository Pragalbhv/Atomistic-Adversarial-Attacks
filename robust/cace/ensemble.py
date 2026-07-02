import torch
from typing import List, Dict, Optional


class CACEEnsemble:
    """Loads and wraps multiple CACE NeuralNetworkPotential checkpoints."""

    def __init__(
        self,
        model_paths: List[str],
        device: str = 'cuda',
        energy_key: str = 'CACE_energy',
        forces_key: str = 'CACE_forces',
        atomic_energies: Optional[Dict[int, float]] = None,
    ):
        self.models = [
            torch.load(p, map_location=device, weights_only=False)
            for p in model_paths
        ]
        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        self.device = device
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.atomic_energies = atomic_energies or {}
        # representation.cutoff is set in Cace.__init__; raises early if wrong attribute
        self.cutoff = self.models[0].representation.cutoff
        self.num_models = len(self.models)
