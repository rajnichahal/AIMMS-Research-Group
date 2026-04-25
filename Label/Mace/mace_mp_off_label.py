from mace.calculators import mace_mp
# from mace.calculators import mace_off  # future use
from ase.io import read
import numpy as np

atoms = read('opt_f_rc.xyz')

# MACE-MP large model with D3 dispersion
calc = mace_mp(model="medium", dispersion=True, device="cuda")

atoms.set_calculator(calc)

potential_energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stresses = atoms.get_stress()

np.savetxt('force_mace-mp_disp.out', forces)

with open('pe_mace-mp_disp.out', 'w') as f:
    f.write(f"{potential_energy}\n")

with open('stress_mace-mp_disp.out', 'w') as f:
    f.write(f"{stresses}\n")