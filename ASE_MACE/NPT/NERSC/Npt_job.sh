#!/bin/bash
#SBATCH --account='Project_Number'
#SBATCH -C gpu
#SBATCH -q regular          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32  
#SBATCH --gres=gpu:1        
#SBATCH -t 48:00:00         
#SBATCH -o maceoff_npt_%j.out
#SBATCH -e maceoff_npt_%j.err

module load python
python -m pip install --user ase
module load pytorch/2.8.0

# ---- Run NPT please run the same job script again after the first 48hr job run no need to change anything it will automatically detect the checkpoint and continue  ----
python npt_nosehover_maceoff.py