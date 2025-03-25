#!/bin/bash

#SBATCH -J test_ligandmpnn
#SBATCH -p kuhlab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --mem=30g
#SBATCH --constraint=rhel8
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate evopro_pyrosetta
module load gcc
module load cuda
#python /proj/kuhl_lab/LigandMPNN/generate_json.py @json.flags
python /proj/kuhl_lab/LigandMPNN/run_mpnn.py --config_file mpnn_basic.yaml
