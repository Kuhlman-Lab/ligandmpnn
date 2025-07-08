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

# Configure multi-PDB input JSONs
python /proj/kuhl_lab/LigandMPNN/get_pdb_multi_json.py --pdb_dir ./ --json_file multi_pdb.json
python /proj/kuhl_lab/LigandMPNN/get_sel_res_multi_json.py --pdb_dir ./ --json_file multi_res.json --sel_restypes "G" --flip

python /proj/kuhl_lab/LigandMPNN/run.py \
        --seed 111 \
        --pdb_path_multi "./multi_pdb.json" \
        --out_folder "./outputs/" \
        --fixed_residues_multi "./multi_res.json" \
        --checkpoint_ligand_mpnn /proj/kuhl_lab/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
        --pack_side_chains 1 \
        --repack_everything 0 \
        --number_of_packs_per_design 1 \
        --checkpoint_path_sc /proj/kuhl_lab/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt \
        --ligand_mpnn_use_side_chain_context 1
