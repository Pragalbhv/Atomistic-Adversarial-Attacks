# CACE Model Integration for Adversarial Attacks

**Objective:** Enable adversarial attacks on CACE models trained via the MetalWall module, extending the adversarial-attacks package from NFF/SchNet to support CACE neural network potentials.

**Status:** Planning phase. Integration requires new adapter layer (`robust/cace/`); no changes to CACE or MetalWall internals needed.

---

## Table of Contents

1. [Exploration Summary](#exploration-summary)
2. [Architecture Overview](#architecture-overview)
3. [Integration Plan](#integration-plan)
4. [Implementation Details](#implementation-details)
5. [Workflow and Validation](#workflow-and-validation)
6. [References](#references)

---

## Exploration Summary

### 1. Adversarial Attacks Package Structure

**Repository:** `/pscratch/sd/p/pvashi/development/Atomistic-Adversarial-Attacks`

**Organization:**

- **Core library:** `robust/` — loss functions, attack loops, ensemble wrappers, data handling
- **Attack pipelines:** Two main strategies
  - **Toy (coordinates):** `AdversarialAttacker` optimizes flat `[batch, input_dim]` tensors with model returning `(x, energy, forces)`
  - **Atomistic (SchNet):** `robust/schnet/Attacker` uses NFF batch dicts (`nxyz`, `nbr_list`, `offsets`, `lattice`) and ensemble models with `{'energy', 'energy_grad', 'stress'}` outputs
- **Active learning:** `robust/actlearn/` — pipeline orchestration, deduplication scoring (RMSD, clustering, uncertainty percentile)
- **Entry points:** Jupyter notebooks (Ammonia, Zeolite, alanine attacks) and standalone scripts

**Public API exports** (`robust/__init__.py`):

- `PotentialDataset`, `VectorDataset`
- `NnRegressor`, `NnEnsemble`
- `MeanSquareLoss`, `AdvLoss`, `AdvLossEnergyUncertainty`
- `Trainer`, `batch_to`
- `ForwardPipeline`, `ActiveLearning`
- Submodules: `potentials`, `hooks`, `metrics`, `attacks`, `loss`, `schnet`

---

Update idea: new strategy must not change CACE, also CACE inherits nn.module, so perhaps it is closer to actlearn instead of schnet based strategies

---

#### Model Interface Contracts

All attacks require stacked ensemble outputs on the **last axis**:

```python
# Generic contract (core to AdvLoss)
energies: Tensor  # shape [..., num_models]
forces:   Tensor  # shape [N, 3, num_models] or [num_atoms, 3, num_models]
```

**Toy pipeline** (`AdversarialAttacker`):

- Input: flat coordinate tensor `x: [batch, input_dim]`
- Model: `nn.Module.forward(x) → (x, energy, forces)`
- Attack: `x_perturbed = x_init + delta` where `delta.requires_grad = True`

---

CACE has a forward function in cace/tasks/[loss.py](http://loss.py)

---

**SchNet pipeline** (`robust/schnet/Attacker`):

- Input: NFF batch dict with `nxyz: [N, 4]`, `nbr_list`, `offsets`, `lattice`
- Model: each ensemble member `m(batch) → {'energy', 'energy_grad': forces, 'stress': optional}`
- Attack: `nxyz[:,1:] += delta` (position-only perturbation); rebuild neighbor list every `nbr_list_update` epochs
- Perturbation bounds: `delta.data.clamp_(-epsilon, epsilon)` after each optimizer step

#### AdvLoss (Model-Agnostic)

```python
class AdvLoss:
    def __init__(self, train: PotentialDataset, temperature: float = 1):
        self.e = train.e  # training energies for Boltzmann weighting
        
    def probability_fn(self, energy):
        return exp(-energy / T) / partition_function
        
    def loss_fn(self, x, e, f=None, s=None):
        # Maximize ensemble disagreement weighted by Boltzmann probability
        if f is not None:  # force variance (Cartesian attacks)
            return -f.var(-1).mean(-1, keepdims=True) * probability_fn(e.mean(-1))
        if s is not None:  # stress variance (lattice attacks)
            return -s.var(-1).mean() * probability_fn(e.mean(-1))
```

This is **fully reusable** once energies and forces are stacked correctly.

#### Deduplication & Scoring

`robust/actlearn/score.py` provides attack filtering on `**PotentialDataset`** objects (flat `x`, `e`, `f` tensors), not raw ASE `Atoms` lists:

- `UncertaintyPercentile` — keep attacked points whose ensemble **force variance** exceeds a percentile of the training-set variance
- `RmsdScore` — keep attacked points whose **minimum RMSD to training coordinates** exceeds a threshold (filters out structures too similar to training)
- `ClusterScore` — hierarchical clustering on attacked coordinates; keeps one representative per cluster (within-set deduplication)

Orchestration is via `ActiveLearning.deduplicate(train_results, attack_results)` in `robust/actlearn/loop.py`, which AND-combines all configured score masks.

**CACE integration note:** before calling these scorers, convert attacked structures to `PotentialDataset` (e.g. flatten `positions` into `x`, store ensemble energies/forces from the final attack step). Alternatively, implement a small ASE-based clustering helper for within-set deduplication on Cartesian coordinates.

---

### 2. CACE Model Interface

**Repository:** `/pscratch/sd/p/pvashi/Models/cace`

#### Model Architecture

`NeuralNetworkPotential` (composable `nn.Module`):

- **Input modules:** preprocessing (stress displacement, etc.)
- **Representation:** `Cace` message-passing network → node features
- **Output modules:** stacked computation heads (energy, forces, charges, MetalWall, Ewald, etc.)

**Key modules:**

- `Atomwise` — per-atom or aggregated scalar/tensor outputs (energy, charges)
- `Forces` — autograd energy gradient w.r.t. positions (differentiable)
- `MetalWall` — electrode charge boundary conditions (Ni-only in NiNaCl)
- `EwaldPotential` — long-range electrostatics on predicted charges
- `FeatureAdd` — linear combination of energy terms
- `Preprocess` — displacement matrix for stress computation

#### Forward Pass

```python
def forward(self, 
            data: Dict[str, torch.Tensor], 
            training: bool = False, 
            compute_stress: bool = True, 
            compute_virials: bool = False) -> Dict[str, torch.Tensor]
```

**Input dict keys** (from `AtomicData.to_dict()`):


| Key                     | Shape                                                | Notes                  |
| ----------------------- | ---------------------------------------------------- | ---------------------- |
| `positions`             | `[n_nodes, 3]`                                       | Node coordinates       |
| `edge_index`            | `[2, n_edges]`                                       | Sender → receiver      |
| `shifts`, `unit_shifts` | `[n_edges, 3]`                                       | Periodic edge offsets  |
| `atomic_numbers`        | `[n_nodes]`                                          | Element IDs            |
| `cell`                  | `[3, 3]` (single graph) or `[batch, 3, 3]` (batched) | Lattice vectors        |
| `batch`                 | `[n_nodes]`                                          | Node→graph assignment  |
| `ptr`                   | `[batch+1]`                                          | Graph boundaries (PyG) |


**Output dict keys** (subset collected from output modules):


| Key                                               | Shape                                 | Notes                                                |
| ------------------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| `CACE_energy`                                     | `[n_graphs]` or `[n_graphs, n_heads]` | Total energy (sum of terms)                          |
| `CACE_forces`                                     | `[n_nodes, 3]`                        | Via `-dE/dR`, requires `training=True`               |
| `stress`                                          | `[n_graphs, 3, 3]`                    | If `Forces` module present and `compute_stress=True` |
| `q`, `q_mw`, `ewald_potential`, `SR_energy`, etc. | varies                                | Intermediate terms (optional)                        |


#### Data Preparation

```
ASE Atoms → AtomicData.from_atoms(cutoff) → PyG Batch → batch.to_dict() → model(dict)
```

`**AtomicData` factory:**

- Reads positions, cell, PBC, atomic numbers from ASE
- Calls matscipy `neighbour_list` with cutoff; extends cell in non-PBC directions
- Extracts labels (energy, forces) from ASE `info`/`arrays` via `data_key` mapping
- Subtracts `atomic_energies` E0 offsets from reference energy
- Converts stress (Voigt → 3×3)

**Batching:** `torch_geometric.dataloader.DataLoader` → `Batch.from_data_list()` → adds `batch` and `ptr`

#### Loading/Saving

**No `from_pretrained` API.** All models saved as full pickled `nn.Module`:

```python
model = torch.load(path, map_location=device, weights_only=False)
model.eval()
for p in model.parameters():
    p.requires_grad = False  # inference
```

Training saves:

- `best_model.pth` — full model object
- `checkpoint.pt` — training state dict
- `best_model_state.pth` — weights only (requires architecture pre-built)

**ASE Calculator:** `CACECalculator` wraps trained model for MD/relaxation inference. **Do not use it for attacks** — it freezes model parameters and rebuilds graphs without a differentiable `delta` link. Attacks need a custom batch builder (see `robust/cace/batch.py` below).

---

### 3. MetalWall Training Setup (NiNaCl)

**Repository:** `/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen1/`

#### What is MetalWall?

**Not a separate trainer.** `MetalWall` is a CACE neural module (`cace/modules/metalwall.py`) that:

- Takes per-atom charges `q` from upstream `Atomwise` head
- Zeros all Ni (Z=28) charges
- Solves metal-wall boundary conditions via S-matrix and Ewald sum
- Outputs corrected charges `q_mw` for long-range electrostatics

Training, data loading, and model assembly all come from **CACE library** (`TrainingTask`, `get_dataset_from_xyz`, `NeuralNetworkPotential`).

---

Metalwalls solves metal boundary condition using Siepmann-Sprik for fixed metal perfect crystal. Attacks should not change position of metal !

---

#### Training Pipeline for model (seed_3 example)

---

Models for ensbmle are kept in playground, alongside a single seed's train.xyz, test.xyz. and it's training cace_train_mw.py. You can use these models for building the ensemble.

Smoke tests can be made in /pscratch/sd/p/pvashi/development/Atomistic-Adversarial-Attacks/smoke_tests

---

**Directory structure: for the model us**

```
playground/
├── cace_train_mw_fixed.py         # seed-specific (SEED=3, TRAIN_PATH prepended)
├── train.extxyz                   # 3,865 training frames
├── test.extxyz                    # 430 held-out frames (not used in training)
└── seed*.pth                      # where * is seed
```

**Configuration (inline Python, no YAML):**

```python
SEED = 3
TRAIN_PATH = '.../seed_3/train.extxyz'
cutoff = 5.5
batch_size = 4
atomic_energies = {28: -0.7517, 11: -0.2286, 17: -0.3704}  # from reference_energies.json

# Representation: Cace([Ni, Na, Cl], n_atom_basis=5, n_radial_basis=12, max_l=3, max_nu=3, num_message_passing=0)
# Output stack: [SR_energy, q, MetalWall, EwaldPotential, FeatureAdd(CACE_energy), Forces(CACE_forces)]
# Optimizer: Adam(lr=5e-3), scheduler: StepLR(step_size=20, gamma=0.5), max_grad_norm=10
```

**Four-phase curriculum (500 epochs total):**


| Phase | Energy weight | Force weight | Epochs       | Checkpoint    |
| ----- | ------------- | ------------ | ------------ | ------------- |
| 1     | 0.1           | 1000         | 5 × 40 = 200 | `model.pth`   |
| 2     | 1             | 1000         | 100          | `model-2.pth` |
| 3     | 10            | 1000         | 100          | `model-3.pth` |
| 4     | 1000          | 1000         | 100          | `model-4.pth` |


**Final model for inference:** `best-model.pth` .

**Training status:** Completed (per `nohup.out`). **Note:** Checkpoints currently absent from filesystem; may need restoration.

#### Model Assembly (output modules)

```python
sr_energy = cace.modules.atomwise.Atomwise(n_layers=3, output_key='SR_energy', n_hidden=[48, 32])
q = cace.modules.Atomwise(n_layers=3, n_hidden=[48, 32], n_out=1, per_atom_output_key='q', output_key='tot_q', bias=False)
mw = cace.modules.MetalWall(metal_atomic_numbers=28, output_key='q_mw')
ep = cace.modules.EwaldPotential(dl=2, sigma=1.0, feature_key='q_mw', output_key='ewald_potential')
e_add = cace.modules.FeatureAdd(feature_keys=['SR_energy', 'ewald_potential'], output_key='CACE_energy')
forces = cace.modules.Forces(energy_key='CACE_energy', forces_key='CACE_forces')

model = cace.models.atomistic.NeuralNetworkPotential(
    representation=cace_representation,
    output_modules=[sr_energy, q, mw, ep, e_add, forces]
)
```

**Data dependency:** `reference_energies.json` (per-species E0 offsets subtracted during training; must be added back at inference via `CACECalculator` or manually).

**Note:** Each gen1 seed directory has its own `train.extxyz` copy (same frames, generated by `postprocessing_full.ipynb`). Any one seed's file is sufficient for `AdvLoss` weights.

---

## Architecture Overview

### Current State: Adversarial Attack on NFF/SchNet

```
robust/schnet/Attacker:
  ├── input: nff.io.AtomsBatch (ASE wrapper)
  │   ├── nxyz: [N, 4]
  │   ├── nbr_list, offsets, lattice
  │   └── neighbor_list refresh every ~2 epochs
  ├── ensemble: EnsembleNFF.models (list of NeuralFF instances)
  ├── model call: m(batch) → {'energy': scalar, 'energy_grad': forces, 'stress': optional}
  ├── loss: AdvLoss (reusable, model-agnostic)
  │   └── maximize force variance × Boltzmann weight
  └── output: attacked_structure.extxyz
```

### Target State: Adversarial Attack on CACE (with MetalWall)

```
robust/cace/Attacker (NEW):
  ├── input: ase.Atoms
  │   ├── AtomicData.from_atoms() → edge_index, shifts, unit_shifts, cell, batch
  │   ├── positions += delta (differentiable)
  │   ├── graph rebuild every ~2 epochs (via AtomicData.from_atoms())
  │   └── includes Ni electrode atoms, PBC required
  ├── ensemble: CACEEnsemble.models (list of NeuralNetworkPotential instances)
  │   └── loaded from gen1/seed_*/model-4.pth
  ├── model call: m(batch_dict, training=True) → {'CACE_energy', 'CACE_forces', 'q_mw', ...}
  │   └── training=True enables autograd through MetalWall → Ewald → Forces
  ├── loss: AdvLoss (reused from robust/loss.py, fully compatible)
  │   └── maximize force/stress variance × Boltzmann weight
  └── output: attacked_structure.extxyz
```

### Key Differences (Interface Gap)


| Component                | SchNet/NFF                                          | CACE                                       |
| ------------------------ | --------------------------------------------------- | ------------------------------------------ |
| **Batch builder**        | `nff.io.AtomsBatch.get_batch()`                     | `AtomicData.from_atoms()` + PyG `Batch`    |
| **Graph representation** | `nbr_list` (COO-like)                               | `edge_index` (COO) + edge shifts           |
| **Perturbation link**    | `nxyz[:,1:] += delta` (separate from `nxyz` tensor) | `positions += delta` (unified)             |
| **Graph refresh**        | `atoms.update_nbr_list()` (ASE neighbor list)       | `AtomicData.from_atoms()` (matscipy + PyG) |
| **Model forward**        | `m(batch)` (dict-like)                              | `m(batch.to_dict(), training=True)`        |
| **Energy output**        | `energy` (sum per structure)                        | `CACE_energy` (sum per structure)          |
| **Forces output**        | `energy_grad` (positions gradient)                  | `CACE_forces` (positions gradient)         |
| **Stress output**        | `stress` (3×3 per structure)                        | `stress` (3×3 per structure)               |


---

## Integration Plan



### Phase 1: DEV Core Adapter Layer (`robust/cace/`)

GOAL: Build a slim adapter that bridges CACE's interface to the existing `AdvLoss` + deduplication machinery.

Step 1: Build model ensemble 

Step 2: Build diff batch - utilize CACE modules/functions when possible

Step 3: Adv Attack

Step 4: Boltzmann weighting

#### Files to Create

**1. `robust/cace/ensemble.py` — Multi-seed ensemble loader**

```python
class CACEEnsemble:
    """Loads and wraps multiple CACE model checkpoints."""
    
    def __init__(self, 
                 model_paths: List[str],
                 device: str = 'cuda',
                 energy_key: str = 'CACE_energy',
                 forces_key: str = 'CACE_forces',
                 atomic_energies: Dict[int, float] = None):
        """
        Args:
            model_paths: List of paths to model-4.pth checkpoints
            device: torch device
            energy_key: output dict key for energy (default 'CACE_energy')
            forces_key: output dict key for forces (default 'CACE_forces')
            atomic_energies: Dict {Z: E0} for per-atom offset correction
        """
        # PyTorch 2.6+ may require safe_globals for pickled NeuralNetworkPotential
        from cace.models.atomistic import NeuralNetworkPotential
        torch.serialization.add_safe_globals([NeuralNetworkPotential])

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
        self.cutoff = self.models[0].representation.cutoff
        self.num_models = len(self.models)
```

**2. `robust/cace/batch.py` — Differentiable batch builder**

Core bridge: ASE → CACE dict with `positions.requires_grad = True`. Mirror the SchNet pattern: rebuild the neighbor graph from `atoms + delta.detach()` periodically, but keep a differentiable link `positions = positions_init + delta` for autograd. (not sure that this is the best approach to mirror)

```python
def prepare_attack_batch(atoms: ase.Atoms,
                         delta: torch.Tensor,
                         cutoff: float,
                         device: str,
                         positions_init: torch.Tensor = None) -> tuple:
    """
    Build CACE batch dict with differentiable positions.

    Returns:
        batch_dict: ready for model(batch_dict, training=True)
        positions_init: unperturbed positions on device (reuse between epochs)
    """
    from cace.data import AtomicData
    from cace.tools import torch_geometric

    # Rebuild graph from detached perturbation (like schnet update_nbr_list)
    atoms_perturbed = atoms.copy()
    atoms_perturbed.positions += delta.detach().cpu().numpy()

    data = AtomicData.from_atoms(atoms_perturbed, cutoff=cutoff)

    loader = torch_geometric.dataloader.DataLoader(
        [data], batch_size=1, shuffle=False, drop_last=False,
    )
    batch = next(iter(loader)).to(device)
    batch_dict = batch.to_dict()

    if positions_init is None:
        positions_init = batch_dict['positions'].detach().clone()

    # Differentiable link: forces backprop to delta, not through graph rebuild
    batch_dict['positions'] = positions_init + delta
    batch_dict['positions'].requires_grad_(True)

    return batch_dict, positions_init
```

**Do not pass `atomic_energies` here** — that argument only subtracts reference energies from **labels** in `AtomicData.from_atoms`, not from model predictions. Use it only when building `AdvLoss` training-energy weights.

**3. `robust/cace/attacker.py` — Main attack loop**

Mirror `robust/schnet/attacker.py` but for CACE. Core logic:

```python
import tqdm
import torch
from .batch import prepare_attack_batch

class Attacker:
    """Performs adversarial attack on CACE model ensemble."""
    
    def __init__(self,
                 initial: ase.Atoms,
                 ensemble: CACEEnsemble,
                 adv_loss,  # AdvLoss instance
                 delta_init: float = 0.01,
                 epsilon: float = 3.0,
                 optim_lr: float = 1e-2,
                 device: str = 'cuda',
                 nbr_list_update: int = 2):
        """
        Args:
            initial: Initial structure (ASE Atoms)
            ensemble: CACEEnsemble with loaded models
            adv_loss: AdvLoss instance (requires training energies)
            delta_init: Gaussian std for delta initialization
            epsilon: Clip perturbation magnitude
            optim_lr: Adam learning rate
            device: torch device
            nbr_list_update: Refresh graph every N epochs
        """
        self.initial = initial
        self.ensemble = ensemble
        self.loss_fn = adv_loss
        self.nbr_list_update = nbr_list_update
        self.delta_init = delta_init
        self.epsilon = epsilon
        self.optim_lr = optim_lr
        self.device = device
        self.num_atoms = len(initial)
        self.cutoff = ensemble.cutoff
        self.positions_init = None
        self.batch_dict = None
    
    def initialize_translation(self, lattice=False):
        if lattice:
            delta = self.delta_init * torch.randn((3, 3), device=self.device)
        else:
            delta = self.delta_init * torch.randn((self.num_atoms, 3), device=self.device)
        delta.requires_grad = True
        opt = torch.optim.Adam([delta], lr=self.optim_lr)
        return delta, opt
    
    def attack(self, lattice=False, epochs=60):
        """Run attack loop."""
        delta, opt = self.initialize_translation(lattice=lattice)
        results = []
        
        for epoch in tqdm.tqdm(range(epochs)):
            epoch_results = self.attack_epoch(opt, delta, epoch, lattice=lattice)
            results.append({'epoch': epoch, **epoch_results})
        
        return results
    
    def attack_epoch(self, opt, delta, epoch, lattice=False):
        """Single attack step."""
        opt.zero_grad()
        
        # Refresh graph periodically; otherwise reuse topology with new delta link
        if epoch % self.nbr_list_update == 0 or self.batch_dict is None:
            self.batch_dict, self.positions_init = prepare_attack_batch(
                self.initial, delta, self.cutoff, self.device, self.positions_init,
            )
        else:
            self.batch_dict['positions'] = self.positions_init + delta
            self.batch_dict['positions'].requires_grad_(True)

        batch_dict = self.batch_dict

        # Forward through all ensemble members
        results = [
            m(batch_dict, training=True, compute_stress=lattice)
            for m in self.ensemble.models
        ]
        
        # Stack outputs: [n_graphs, n_models] for energy, [n_nodes, 3, n_models] for forces
        energy = torch.stack(
            [r[self.ensemble.energy_key] for r in results],
            dim=-1
        )
        forces = torch.stack(
            [r[self.ensemble.forces_key] for r in results],
            dim=-1
        )
        
        energy_per_atom = energy / self.num_atoms
        
        if lattice:
            stress = -torch.stack([r['stress'] for r in results], dim=-1)
            loss = self.loss_fn.loss_fn(x=None, e=energy_per_atom, s=stress).sum()
        else:
            loss = self.loss_fn.loss_fn(x=None, e=energy_per_atom, f=forces).sum()
        
        loss_item = loss.item()
        loss.backward()
        opt.step()
        delta.data.clamp_(-self.epsilon, self.epsilon)
        
        return {
            'delta': delta.detach().cpu().numpy(),
            'energy': energy.detach().cpu().numpy(),
            'forces': forces.detach().cpu().numpy(),
            'loss': loss_item,
        }
```

**4. `robust/cace/data.py` — Training energies for AdvLoss**

Extract per-atom energies from CACE training data for Boltzmann weighting:

```python
def load_training_energies(train_path: str,
                           atomic_energies: Dict[int, float] = None):
    """
    Load per-atom energies from extxyz for AdvLoss Boltzmann weighting.

    Returns:
        Object with `.e` attribute (1D tensor of per-atom energies), compatible with AdvLoss.
    """
    from ase.io import read
    
    structures = read(train_path, index=':')
    if not isinstance(structures, list):
        structures = [structures]
    
    energies = []
    for atoms in structures:
        # extxyz stores total energy in atoms.info['energy']
        E_total = atoms.info.get('energy', atoms.get_potential_energy())
        n_atoms = len(atoms)
        if atomic_energies:
            E0 = sum(atomic_energies.get(Z, 0) for Z in atoms.get_atomic_numbers())
            E_total -= E0
        energies.append(E_total / n_atoms)

    class EnergyDataset:
        def __init__(self, energies):
            self.e = torch.tensor(energies, dtype=torch.float32)

    return EnergyDataset(energies)
```

**5. `robust/cace/__init__.py`**

```python
from .ensemble import CACEEnsemble
from .attacker import Attacker
from .batch import prepare_attack_batch
from .data import load_training_energies

__all__ = [
    'CACEEnsemble',
    'Attacker',
    'prepare_attack_batch',
    'load_training_energies',
]
```

#### Modify Existing Files

**6. `robust/__init__.py` — Add submodule export**

```python
# ... existing exports ...
from . import cace  # NEW
```

---

### Phase Prod: Example Notebook/Script

This is Production, should be relatively trivial 

**File:** `examples/ninacl_mw_attack.py` (or `ninacl_mw_attack.ipynb`)

Orchestration: load ensemble, run attacks, deduplicate, save.

```python
#!/usr/bin/env python
"""
Adversarial attacks on CACE+MetalWall ensemble for NiNaCl system.
Generates attacked structures for next-generation training.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '/pscratch/sd/p/pvashi/Models/cace')
sys.path.insert(0, '/pscratch/sd/p/pvashi/development/Atomistic-Adversarial-Attacks')

import torch
import numpy as np
from ase.io import read, write
import tqdm

import robust as rb
from robust.cace import CACEEnsemble, Attacker, load_training_energies

# ============================================================================
# Configuration
# ============================================================================

# Ensemble paths (gen1 training)
SEED_PATHS = [
    '/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen1/seed_1/model-4.pth',
    '/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen1/seed_2/model-4.pth',
    '/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen1/seed_3/model-4.pth',
    '/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen1/seed_4/model-4.pth',
    '/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen1/seed_5/model-4.pth',
]

# Training data (for AdvLoss Boltzmann weights)
TRAIN_PATH = '/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen1/seed_3/train.extxyz'

# Reference energies
with open('/pscratch/sd/p/pvashi/SciDAC/NiNaCl/data/reference_energies.json') as f:
    ref_energies = json.load(f)

ATOMIC_ENERGIES = {28: ref_energies['Ni'], 11: ref_energies['Na'], 17: ref_energies['Cl']}

# Attack parameters
DEVICE = 'cuda'
CUTOFF = 5.5
TEMPERATURE = 1.0  # AdvLoss Boltzmann temperature
DELTA_INIT = 0.01
EPSILON = 3.0  # max perturbation (Angstroms)
OPTIM_LR = 1e-2
NUM_ATTACK_EPOCHS = 60
NBR_LIST_UPDATE = 2
MAX_STRUCTURES = 20  # smoke test; increase for production campaign

# Output
OUTPUT_DIR = Path('/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen2/attacks')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading ensemble...")
    ensemble = CACEEnsemble(
        SEED_PATHS,
        device=DEVICE,
        atomic_energies=ATOMIC_ENERGIES,
    )
    print(f"Ensemble: {len(ensemble.models)} models, cutoff={ensemble.cutoff} Å")
    
    print(f"Loading training energies from {TRAIN_PATH}...")
    train_energies = load_training_energies(TRAIN_PATH, ATOMIC_ENERGIES)
    
    print(f"Creating AdvLoss (T={TEMPERATURE})...")
    adv_loss = rb.loss.AdvLoss(train_energies, temperature=TEMPERATURE)
    
    print(f"Loading seed structures from {TRAIN_PATH}...")
    structures = read(TRAIN_PATH, index=':')
    if not isinstance(structures, list):
        structures = [structures]
    structures = structures[:MAX_STRUCTURES]

    print(f"Selected {len(structures)} structures for attack")
    
    attacked_all = []
    
    for idx, atoms_init in enumerate(tqdm.tqdm(structures, desc="Attacking")):
        print(f"\n[{idx+1}/{len(structures)}] Attacking structure...")
        
        try:
            attacker = Attacker(
                atoms_init,
                ensemble,
                adv_loss,
                delta_init=DELTA_INIT,
                epsilon=EPSILON,
                optim_lr=OPTIM_LR,
                device=DEVICE,
                nbr_list_update=NBR_LIST_UPDATE,
            )
            
            attack_results = attacker.attack(lattice=False, epochs=NUM_ATTACK_EPOCHS)
            
            # Extract final perturbed structure (delta is already numpy from attacker)
            final_delta = attack_results[-1]['delta']
            final_forces = attack_results[-1]['forces']
            force_variance = np.var(final_forces, axis=-1).mean()

            atoms_attacked = atoms_init.copy()
            atoms_attacked.positions += final_delta
            
            attacked_all.append(atoms_attacked)
            
            print(f"  ✓ Attack success: force variance = {force_variance:.4e}")
        
        except Exception as e:
            print(f"  ✗ Attack failed: {e}")
            continue
    
    # Save attacked structures
    out_path = OUTPUT_DIR / f'attacked_all.extxyz'
    write(out_path, attacked_all)
    print(f"\nWrote {len(attacked_all)} attacked structures to {out_path}")
    
    # Within-set deduplication by hierarchical clustering on positions
    # (robust.actlearn.score.ClusterScore expects PotentialDataset, not ASE Atoms)
    print("Deduplicating by structural clustering...")
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster

    coords = np.array([a.get_positions().ravel() for a in attacked_all])
    if len(coords) > 1:
        Z = linkage(pdist(coords), method='average')
        clusters = fcluster(Z, t=0.5, criterion='distance')
        keep_idx = [np.where(clusters == c)[0][0] for c in np.unique(clusters)]
        attacked_dedup = [attacked_all[i] for i in keep_idx]
    else:
        attacked_dedup = attacked_all

    out_dedup = OUTPUT_DIR / 'attacked_dedup.extxyz'
    write(out_dedup, attacked_dedup)
    print(f"Wrote {len(attacked_dedup)} deduplicated structures to {out_dedup}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
```

---

## Implementation Details

### Data Flow: Attack Epoch

```
┌─────────────────────────────────────────────────────────────────┐
│ Epoch N                                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Initialize:                                                 │
│     delta ∈ ℝ^(N×3), delta.requires_grad = True               │
│     opt = Adam([delta], lr)                                    │
│                                                                 │
│  2. [Every nbr_list_update epochs]                             │
│     Rebuild graph: AtomicData.from_atoms(atoms + delta_np)    │
│     → edge_index, shifts, unit_shifts                          │
│                                                                 │
│  3. Differentiable link:                                        │
│     batch_dict['positions'] = pos_init + delta                │
│     positions.requires_grad = True                             │
│                                                                 │
│  4. Forward through ensemble:                                   │
│     for i in 1..num_models:                                    │
│       output_i = model_i(batch_dict, training=True)            │
│         → CACE_energy_i [1], CACE_forces_i [N,3]              │
│       │ (forces computed via autograd through MetalWall)       │
│       │                                                         │
│     E = stack([E_i], dim=-1)  → [1, M]                        │
│     F = stack([F_i], dim=-1)  → [N, 3, M]                     │
│                                                                 │
│  5. Loss (AdvLoss, maximizes disagreement):                      │
│     loss = -var(F, dim=-1).mean() * p(E_per_atom)             │
│     where p(E) = exp(-E/T) / partition_fn                     │
│     (loss is negative; should become more negative over time)  │
│                                                                 │
│  6. Backprop:                                                   │
│     loss.backward()  → ∂loss/∂delta                           │
│     opt.step()       → delta -= lr * ∂loss/∂delta             │
│     delta.clamp(-ε, ε)                                         │
│                                                                 │
│  7. Collect results:                                            │
│     {'delta': numpy, 'energy': [..., M], 'forces': [..., M], │
│      'loss': scalar}                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MetalWall-Specific Constraints

1. **Attacked tructures must not move Ni (Z=28) electrode atoms**
  - `MetalWall` zeros metal charges and solves BC; non-Ni-containing systems will fail or produce nonsense, especially if metal atoms are moved
2. **Periodic boundary conditions required**
  - `EwaldPotential` and `MetalWall` both depend on `cell` and PBC
  - Attacks must preserve cell vectors (no lattice-mode initially)
3. **Use total energy key `CACE_energy`, not `SR_energy`**
  - The attack probes disagreement in the full electrostatic pipeline
  - Short-range alone misses the charge/Ewald coupling
4. `**training=True` on forward**
  - Forces are computed via autograd through the entire stack (MetalWall → Ewald → Forces)
  - Critical for differentiable attacks
5. **Charge pathway is position-dependent**
  - When atoms move, predicted charges `q` change → `q_mw` (metal corrections) → `ewald_potential` → total energy
  - This multipath feedback is exactly what the attack exploits

---

## Workflow and Validation

### Stage 4→Attack: Sample Selection

**Input:** `gen1/seed_3/train.extxyz` (3,865 frames) or MD trajectory

**Selection strategy:**

- Start small: 10–20 frames for smoke test
- Pick diverse structures: min/max density, various T, edge cases
- Or: random stratified sample

### Attack Stage: Run Attacks

**Pseudocode:**

```
for each seed_structure in sample:
    attacker = Attacker(seed, ensemble, adv_loss, ...)
    results = attacker.attack(epochs=60)
    final_delta = results[-1]['delta']
    attacked_atoms = seed + final_delta
    save(attacked_atoms)
```

**Output:** `gen2/attacks/attacked_all.extxyz` (one attacked structure per seed)

### Filtering: Deduplication

**Techniques** (from `robust/actlearn/score.py`, via `ActiveLearning.deduplicate`):

1. `**UncertaintyPercentile`** — keep attacks with high ensemble force variance
2. `**RmsdScore`** — keep attacks **dissimilar** from training coordinates (min RMSD > threshold)
3. `**ClusterScore`** — within-set clustering; one structure per cluster

All require `PotentialDataset` inputs. For a first CACE script, use direct position clustering (as in the example above) or build a thin adapter from attacked ASE structures + ensemble outputs.

**Rough heuristic:** keep 10–20% of attacks for DFT (e.g. 400 attacks → 40–80 selected).

### Stage 5: DFT (VASP)

Label attacked structures with reference forces/energy.

### Gen2 Training

Same 4-phase curriculum; train on `gen1_train + labeled_attacks`.

---

## Validation Checklist


| Item                        | Check                                                                                         |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| **Graph integrity**         | No NaNs in `edge_index`, reasonable neighbor counts                                           |
| **Loss trend**              | AdvLoss is negative; magnitude should grow (more negative) as ensemble disagreement increases |
| **Structure preservation**  | Final attacked structures contain all original atoms (Ni + salt)                              |
| **Cell stability**          | Cell unchanged (if lattice=False)                                                             |
| **Force variance**          | Ensemble force disagreement increases then plateaus                                           |
| **Energy Boltzmann weight** | Weights should be ~1 for training-like energies, <1 for outliers                              |
| **Inference compatibility** | Attacked extxyz can be read by `CACECalculator` without NaN                                   |
| **Deduplication**           | Clustering removes near-duplicate attacked structures; retained set is diverse                |


---

## Project Structure (Gen2 Campaign)

```
SciDAC/NiNaCl/
├── data/
│   ├── reference_energies.json
│   └── notebooks/postprocessing_full.ipynb
├── models/
│   ├── fixed/
│   │   ├── gen1/
│   │   │   ├── seed_1..5/
│   │   │   │   ├── model-4.pth             ← Load for ensemble
│   │   │   │   ├── train.extxyz            ← Use for AdvLoss
│   │   │   │   └── ...
│   │   │   ├── cace_train_mw_fixed_template.py  ← gen1 master template
│   │   │   └── launch_nohup.sh
│   │   └── gen2/                          ← NEW
│   │       ├── attacks/
│   │       │   ├── attacked_all.extxyz     ← ALL attacked structures
│   │       │   ├── attacked_dedup.extxyz   ← Filtered for DFT
│   │       │   └── attack_manifest.yaml    ← Provenance
│   │       ├── dft/                        ← VASP outputs
│   │       │   ├── attacked_0000/OUTCAR
│   │       │   ├── attacked_0001/OUTCAR
│   │       │   └── ...
│   │       ├── train.extxyz                ← gen1_train + labeled_attacks
│   │       └── seed_1..5/
│   │           ├── cace_train_mw_fixed.py
│   │           ├── train.extxyz            ← Copy of gen2/train.extxyz
│   │           └── ...
│   └── cace_train_template.py              ← Pt-electrode template (no MetalWall)
└── ...

Atomistic-Adversarial-Attacks/
├── robust/
│   ├── cace/                            ← NEW
│   │   ├── __init__.py
│   │   ├── ensemble.py                  ← CACEEnsemble
│   │   ├── batch.py                     ← prepare_attack_batch
│   │   ├── attacker.py                  ← Attacker class
│   │   └── data.py                      ← load_training_energies
│   ├── schnet/
│   ├── actlearn/
│   ├── __init__.py                      ← Update: add cace export
│   └── ...
├── examples/
│   ├── ninacl_mw_attack.py              ← NEW: orchestration script
│   ├── Ammonia_attack.ipynb             ← (existing)
│   └── ...
└── ...
```

---

## References

### Adversarial Attacks Package

- **Repository:** `/pscratch/sd/p/pvashi/development/Atomistic-Adversarial-Attacks`
- **Core classes:**
  - `robust.attacks.AdversarialAttacker` (toy coordinate attacks)
  - `robust.schnet.Attacker` (NFF/SchNet attacks)
  - `robust.loss.AdvLoss` (model-agnostic adversarial objective)
  - `robust.actlearn.score` (deduplication: RMSD, clustering, uncertainty)
- **Example notebooks:**
  - `examples/Ammonia_attack.ipynb` — full SchNet workflow
  - `examples/1D_DoubleWell.ipynb`, `2D_DoubleWell.ipynb` — toy workflows

### CACE Repository

- **Repository:** `/pscratch/sd/p/pvashi/Models/cace`
- **Key files:**
  - `cace/models/atomistic.py` — `NeuralNetworkPotential`, `AtomisticModel`
  - `cace/data/atomic_data.py` — `AtomicData.from_atoms()`, graph building
  - `cace/calculators/cace_calculator.py` — ASE interface
  - `cace/modules/metalwall.py`, `metalwall_qeq.py` — electrode BC modules
  - `cace/modules/ewald.py` — long-range electrostatics
  - `cace/tasks/train.py` — `TrainingTask`, `save_model()`, `checkpoint()`
  - `cace/tasks/load_data.py` — `get_dataset_from_xyz()`, `load_data_loader()`

### MetalWall Training (NiNaCl)

- **Gen1 template:** `/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen1/cace_train_mw_fixed_template.py`
- **Pt-electrode template (no MetalWall):** `/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/cace_train_template.py`
- **Trained models:** `/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen1/seed_*/{model.pth,model-2.pth,model-3.pth,model-4.pth}`
- **Training data:** `/pscratch/sd/p/pvashi/SciDAC/NiNaCl/models/fixed/gen1/seed_3/train.extxyz` (3,865 frames)
- **Reference energies:** `/pscratch/sd/p/pvashi/SciDAC/NiNaCl/data/reference_energies.json`

### Related Concepts

- **CACE & MetalWall:** IRP/SciDAC documentation (internal)
- **PyTorch Geometric:** `torch_geometric.data.Data`, `Batch`, `DataLoader` (vendored in `cace/tools/torch_geometric/`)
- **ASE:** `ase.Atoms`, `Calculator`, neighbor lists, I/O

---

**Document Date:** June 17, 2026  
**Last Updated:** June 18, 2026 (redits made by MaterialsWizard)  
**Status:** Planning (Phase Dev -Step1 implementation ready to begin)