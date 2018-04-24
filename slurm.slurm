#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --account=alzgen

module load anaconda3
module load keras/2.0.8-tf2
module load tensorflow2/1.3.0

./HW4.py
