#!/bin/bash

#SBATCH -J test_ligandmpnn
#SBATCH -p kuhlab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 6-00:00:00
#SBATCH --mem=50g
#SBATCH --constraint=rhel8
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate ligandmpnn
module load gcc
module load cuda
python /proj/kuhl_lab/LigandMPNN/run_mpnn.py --config_file mpnn_basic.yaml
