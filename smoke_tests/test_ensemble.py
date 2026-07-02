"""
Smoke test: Step 1 — CACEEnsemble loads all models and runs ensemble forward pass.

Run from the repo root:
    module load python && conda activate mace_env && python smoke_tests/test_ensemble.py
"""

import sys
import os

AAA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACE_ROOT = '/pscratch/sd/p/pvashi/Models/cace'
sys.path.insert(0, CACE_ROOT)
sys.path.insert(0, AAA_ROOT)

import torch
from ase.io import read
from cace.data import AtomicData
from cace.tools.torch_geometric import Batch
from robust.cace import CACEEnsemble

PLAYGROUND = os.path.join(AAA_ROOT, 'playground')
MODEL_PATHS = [os.path.join(PLAYGROUND, f'seed{i}.pth') for i in range(1, 6)]
TRAIN_PATH = os.path.join(PLAYGROUND, 'train.extxyz')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Loading {len(MODEL_PATHS)} models ...")

ensemble = CACEEnsemble(model_paths=MODEL_PATHS, device=DEVICE)
print(f"  cutoff = {ensemble.cutoff} Å,  num_models = {ensemble.num_models}")

print(f"Loading first structure from {TRAIN_PATH} ...")
atoms = read(TRAIN_PATH, index=0)
n_atoms = len(atoms)
print(f"  {n_atoms} atoms, species: {set(atoms.get_chemical_symbols())}")

data = AtomicData.from_atoms(atoms, cutoff=ensemble.cutoff)
batch = Batch.from_data_list([data]).to(DEVICE)
batch_dict = batch.to_dict()

# --- single-model checks ---
print("\n[1/3] Single-model forward (training=True) ...")
model = ensemble.models[0]
with torch.enable_grad():
    out0 = model(batch_dict, training=True)

assert 'CACE_energy' in out0, f"Missing CACE_energy; got: {list(out0.keys())}"
assert 'CACE_forces' in out0, f"Missing CACE_forces; got: {list(out0.keys())}"
assert out0['CACE_forces'].grad_fn is not None, "forces has no grad_fn — autograd broken"
print(f"  energy : {out0['CACE_energy'].item():.4f} eV")
print(f"  forces shape : {out0['CACE_forces'].shape}")
print("  OK")

# --- full ensemble forward + stacking ---
print(f"\n[2/3] Full ensemble forward ({ensemble.num_models} models) ...")
with torch.enable_grad():
    results = [m(batch_dict, training=True) for m in ensemble.models]

energies = torch.stack([r[ensemble.energy_key] for r in results], dim=-1)  # [1, M]
forces   = torch.stack([r[ensemble.forces_key] for r in results], dim=-1)  # [N, 3, M]

assert energies.shape == (1, ensemble.num_models), \
    f"energy shape {energies.shape} != (1, {ensemble.num_models})"
assert forces.shape == (n_atoms, 3, ensemble.num_models), \
    f"forces shape {forces.shape} != ({n_atoms}, 3, {ensemble.num_models})"
assert forces.grad_fn is not None, "stacked forces has no grad_fn"

print(f"  energies shape : {energies.shape}")
print(f"  forces shape   : {forces.shape}")
print(f"  energy per model (eV): {energies.squeeze().tolist()}")
print("  OK")

# --- ensemble disagreement ---
print("\n[3/3] Ensemble disagreement ...")
force_var = forces.var(dim=-1)          # [N, 3]
mean_force_var = force_var.mean().item()
energy_std = energies.std(dim=-1).item()

print(f"  mean force variance : {mean_force_var:.6e} (eV/Å)²")
print(f"  energy std          : {energy_std:.6f} eV")
assert mean_force_var >= 0, "negative variance — impossible"
print("  OK")


print(f"Loading 10 structure from {TRAIN_PATH} ...")
atoms = read(TRAIN_PATH, index=':10')

from tqdm import tqdm
for i, atom in tqdm(enumerate(atoms)):
    print(f"  Structure {i+1}: {len(atom)} atoms, species: {set(atom.get_chemical_symbols())}")
    n_atoms = len(atom)
    data = AtomicData.from_atoms(atom, cutoff=ensemble.cutoff)
    batch = Batch.from_data_list([data]).to(DEVICE)
    batch_dict = batch.to_dict()

    print(f"\n[{i+1}/{len(atoms)}] Full ensemble forward ({ensemble.num_models} models) ...")
    with torch.enable_grad():
        results = [m(batch_dict, training=True) for m in ensemble.models]

    energies = torch.stack([r[ensemble.energy_key] for r in results], dim=-1)  # [1, M]
    forces   = torch.stack([r[ensemble.forces_key] for r in results], dim=-1)  # [N, 3, M]

    assert energies.shape == (1, ensemble.num_models), \
        f"energy shape {energies.shape} != (1, {ensemble.num_models})"
    assert forces.shape == (n_atoms, 3, ensemble.num_models), \
        f"forces shape {forces.shape} != ({n_atoms}, 3, {ensemble.num_models})"
    assert forces.grad_fn is not None, "stacked forces has no grad_fn"

    print(f"  energies shape : {energies.shape}")
    print(f"  forces shape   : {forces.shape}")
    print(f"  energy per model (eV): {energies.squeeze().tolist()}")
    print("  OK")

    # --- ensemble disagreement ---
    print("\n[3/3] Ensemble disagreement ...")
    force_var = forces.var(dim=-1)          # [N, 3]
    mean_force_var = force_var.mean().item()
    energy_std = energies.std(dim=-1).item()

    print(f"  mean force variance : {mean_force_var:.6e} (eV/Å)²")
    print(f"  energy std          : {energy_std:.6f} eV")
    assert mean_force_var >= 0, "negative variance — impossible"
    print("  OK")




print("\nSMOKE TEST PASSED")
