#!/bin/bash
#SBATCH -FE_piezoBeam                    # Job name
#SBATCH --account=gts-aerturk3                 # charge account
#SBATCH -N1 --ntasks-per-node=24                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=8G                        # Memory per core
#SBATCH -t00:02:00:00                                  # Duration of the job (Ex: 15 mins)
#SBATCH -qembers                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=setemadi3@gatech.edu        # E-mail address for notifications

module load anaconda3/2022.05                   # Load module dependencies
srun python test.py                             # Example Process