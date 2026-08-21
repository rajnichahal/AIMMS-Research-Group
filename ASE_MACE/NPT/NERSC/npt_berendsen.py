# -*- coding: utf-8 -*-
from mace.calculators import mace_off
from ase.io import read, write
from ase.md.nptberendsen import NPTBerendsen
from ase import units
import numpy as np
import os
import time

#  Settings Put the Trajectory file here Insert the temperature
xyz_path = "styrene-pbcoe-train6.10-2_s2-300k_rhoNNIP_nvtequi.xyz"
T = 300.0
P = 1.01325 * units.bar
dt = 1.0 * units.fs

npt_steps           = 10000    # 4_000_000 for full 4 ns run
output_interval     = 10     # density CSV every 5 ps (at full scale)
trajectory_interval = 50     # xyz frame every 25 ps
checkpoint_interval = 50     # checkpoint every 250 ps
progress_interval   = 50     # print % every 10%
chunk_size          = 10     # steps per dyn.run() call — reduces Python overhead
                             # set to 100 or 500 for the full production run

density_file      = "density.csv"
checkpoint_file   = "checkpoint.xyz"
restart_step_file = "restart_step.txt"
trajectory_file   = "npt_trajectory.xyz"
energies_file     = "energies.csv"
stress_file       = "stress.csv"
forces_file       = "forces.csv"

# Checkpoint / Restart 
start_step = 0

if os.path.exists(checkpoint_file) and os.path.exists(restart_step_file):
    with open(restart_step_file, "r") as f:
        start_step = int(f.read().strip())
    print(f"*** RESTARTING from checkpoint at step {start_step} "
          f"({start_step * dt / units.fs / 1000:.2f} ps) ***")
    atoms = read(checkpoint_file, index=-1)
    
    atoms.calc = mace_off(model="medium", dispersion=True, device="cuda",
                          default_dtype="float32")
else:
    print("*** FRESH START ***")
    atoms = read(xyz_path, index=-1)
    # float32: ~2-3x faster on GPU, sufficient accuracy for density/MD
    atoms.calc = mace_off(model="medium", dispersion=True, device="cuda",
                          default_dtype="float32")

    cell0 = atoms.get_cell().lengths()
    print(f"Initial cell: a={cell0[0]:.4f} b={cell0[1]:.4f} c={cell0[2]:.4f} A")
    print(f"Initial volume: {atoms.get_volume():.2f} A3")
    atoms.wrap()


dyn = NPTBerendsen(
    atoms,
    timestep=dt,
    temperature_K=T,
    pressure_au=1.01325 * units.bar,   
    taut=100 * units.fs,                #  thermostat time constant
    taup=1000 * units.fs,               # barostat time constant (replaces pfactor)
    compressibility_au=4.57e-5 / units.bar,  # polymer compressibility
)

#  CSV headers (fresh start only) 
if start_step == 0:
    initial_rho = atoms.get_masses().sum() / atoms.get_volume() * 1.66054
    with open(density_file, "w") as f:
        f.write("step,time_ps,volume_A3,density_gcm3,change_ppm,a_A,b_A,c_A\n")
    cell_i = atoms.get_cell().lengths()
    with open(density_file, "a") as f:
        f.write(f"0,0.000,{atoms.get_volume():.2f},{initial_rho:.6f},0.0,"
                f"{cell_i[0]:.4f},{cell_i[1]:.4f},{cell_i[2]:.4f}\n")
    with open(energies_file, "w") as f:
        f.write("frame,step,pe_eV\n")
    with open(stress_file, "w") as f:
        f.write("frame,step,sxx,syy,szz,syz,sxz,sxy\n")
    with open(forces_file, "w") as f:
        f.write("frame,atom,Fx,Fy,Fz\n")
    print(f"Initial density: {initial_rho:.6f} g/cm3")
else:
    with open(density_file, "r") as f:
        lines = f.readlines()
    initial_rho = float(lines[1].split(",")[3])
    print(f"Reference density (from CSV): {initial_rho:.6f} g/cm3")

print(f"\nRunning NPT (Berendsen): T={T} K  P=1 atm  dt={dt/units.fs} fs  "
      f"{npt_steps} steps = {npt_steps * dt / units.fs / 1e6:.1f} ns")
print(f"chunk_size={chunk_size}  |  float32 CUDA")
print("-" * 60)

frame_counter = start_step // output_interval
step = start_step
t_start = time.time()
steps_timed = 0

#  Main loop (chunked)
# dyn.run(chunk) instead of dyn.run(1) to reduce Python overhead
while step < npt_steps:
    this_chunk = min(chunk_size, npt_steps - step)

    # finds next output event within this chunk
    next_output = output_interval - (step % output_interval) if step % output_interval != 0 else output_interval
    next_traj   = trajectory_interval - (step % trajectory_interval) if step % trajectory_interval != 0 else trajectory_interval
    next_ckpt   = checkpoint_interval - (step % checkpoint_interval) if step % checkpoint_interval != 0 else checkpoint_interval
    next_event  = min(this_chunk, next_output, next_traj, next_ckpt)

    # prints first 20 steps one-by-one so we can see early behaviour
    if step < 20:
        next_event = 1

    dyn.run(next_event)
    step += next_event
    steps_timed += next_event

    # timing feedback
    elapsed = time.time() - t_start
    rate = steps_timed / elapsed  # steps/sec
    eta_hr = (npt_steps - step) / rate / 3600 if rate > 0 else 0

    # output
    do_output = (step % output_interval == 0) or (step <= 20)
    if do_output:
        atoms_out = atoms.copy()
        atoms_out.wrap()

        rho    = atoms.get_masses().sum() / atoms.get_volume() * 1.66054
        change = (rho - initial_rho) * 1_000_000
        time_ps = step * dt / units.fs / 1000
        vol    = atoms.get_volume()
        cell   = atoms.get_cell().lengths()
        pe     = atoms.get_potential_energy()
        stress = atoms.get_stress()
        forces = atoms.get_forces()
        atom_ids = np.arange(1, len(forces) + 1)

        print(f"Step {step:8d} | t={time_ps:7.3f} ps | rho={rho:.6f} g/cm3 "
              f"(d={change:+.1f} ppm) | V={vol:.2f} | "
              f"a={cell[0]:.4f} b={cell[1]:.4f} c={cell[2]:.4f} | "
              f"{rate:.2f} steps/s | ETA {eta_hr:.1f} hr")

        with open(density_file, "a") as f:
            f.write(f"{step},{time_ps:.3f},{vol:.2f},{rho:.6f},{change:.1f},"
                    f"{cell[0]:.4f},{cell[1]:.4f},{cell[2]:.4f}\n")
        with open(energies_file, "a") as f:
            f.write(f"{frame_counter},{step},{pe:.16e}\n")
        with open(stress_file, "a") as f:
            f.write(f"{frame_counter},{step},"
                    f"{stress[0]:.16e},{stress[1]:.16e},{stress[2]:.16e},"
                    f"{stress[3]:.16e},{stress[4]:.16e},{stress[5]:.16e}\n")
        with open(forces_file, "a") as f:
            for i, F in zip(atom_ids, forces):
                f.write(f"{frame_counter},{i},"
                        f"{F[0]:.16e},{F[1]:.16e},{F[2]:.16e}\n")
        frame_counter += 1

    if step % trajectory_interval == 0:
        atoms_out = atoms.copy()
        atoms_out.wrap()
        write(trajectory_file, atoms_out, append=True)

    if step % checkpoint_interval == 0:
        atoms_ckpt = atoms.copy()
        atoms_ckpt.wrap()
        write(checkpoint_file, atoms_ckpt)
        with open(restart_step_file, "w") as f:
            f.write(str(step))
        print(f"  [CHECKPOINT @ step {step} / {step * dt / units.fs / 1000:.1f} ps]")

    if step % progress_interval == 0:
        print(f"  >>> {step/npt_steps*100:.0f}% complete "
              f"({step * dt / units.fs / 1000:.2f} ps) | "
              f"{rate:.2f} steps/s | ETA {eta_hr:.1f} hr")

# Final summary
final_cell = atoms.get_cell().lengths()
final_rho  = atoms.get_masses().sum() / atoms.get_volume() * 1.66054
total_elapsed = time.time() - t_start
print("\n" + "=" * 60)
print("SIMULATION COMPLETE")
print(f"Final cell    : a={final_cell[0]:.4f} b={final_cell[1]:.4f} c={final_cell[2]:.4f} A")
print(f"Final density : {final_rho:.6f} g/cm3")
print(f"Total time    : {npt_steps * dt / units.fs / 1e6:.2f} ns")
print(f"Wall time     : {total_elapsed/3600:.2f} hr  |  "
      f"avg {npt_steps/total_elapsed:.2f} steps/s")
