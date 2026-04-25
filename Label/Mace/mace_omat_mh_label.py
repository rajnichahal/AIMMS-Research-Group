from mace.calculators import mace_mp
from ase.io import read
import numpy as np
import os

folders = sorted([f for f in os.listdir('.') if f.startswith('ne_opt_f_rc') and os.path.isdir(f)])
print(f"Found {len(folders)} configurations")

models = {
    "mace_omat_0": mace_mp(model="medium-omat-0", device="cuda", default_dtype="float64"),
    "mace_mh_0":   mace_mp(model="mh-0", device="cuda", default_dtype="float64", head="omat_pbe"),
    "mace_mh_1":   mace_mp(model="mh-1", device="cuda", default_dtype="float64", head="omat_pbe"),
}

for folder in folders:
    xyz_path = os.path.join(folder, 'opt_f_rc.xyz')
    for model_name, calc in models.items():
        # Create model subfolder INSIDE each rc folder
        out_dir = os.path.join(folder, model_name)
        os.makedirs(out_dir, exist_ok=True)
        atoms = read(xyz_path, format='extxyz')
        atoms.calc = calc
        np.savetxt(f"{out_dir}/force_{model_name}.out", atoms.get_forces())
        with open(f"{out_dir}/pe_{model_name}.out", 'w') as f:
            f.write(f"{atoms.get_potential_energy()}\n")
        with open(f"{out_dir}/stress_{model_name}.out", 'w') as f:
            f.write(f"{atoms.get_stress()}\n")
        print(f"[{folder}] [{model_name}] done")

print("All done!")
