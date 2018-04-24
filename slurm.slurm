SLURM CODE
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=largemem
#SBATCH --account=alzgen

module load anaconda3
module load tensorflow3/1.6.0

./HW4.py
