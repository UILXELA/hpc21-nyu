#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=24
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --verbose

module purge
module load amber/openmpi/intel/20.06

srun ./pingpong 0 23
