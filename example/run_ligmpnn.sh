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
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/" \
        --redesigned_residues "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10"
