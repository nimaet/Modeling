#!/bin/bash
#SBATCH -JFE_piezoBeam                    # Job name
#SBATCH --account=gts-aerturk3                 # charge account
#SBATCH -N 1
#SBATCH --ntasks=1        # ← THIS IS THE KEY LINE
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH -t120
#SBATCH -qembers                   # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file


module load anaconda3                # Load module dependencies
srun python 2D_sweep.py                             # Example Process