import argparse
import os
import json

from data_utils import parse_PDB, restype_int_to_str


def main(args):

    d = args.pdb_dir
    if not os.path.exists(d):
        raise ValueError(f"JSON generation failed - input dir {d} does not exist!")
    
    pdbs = sorted([p for p in os.listdir(d) if p.endswith(".pdb")])

    sel_restypes = args.sel_restypes.split(",")

    data = {}
    for p in pdbs:
        path = os.path.realpath(os.path.join(d, p))

        # Parse the PDB with the LigandMPNN parser
        parsed = parse_PDB(path)
        pdb = parsed[0]
        icodes = parsed[3]
        res_idx = list(pdb["R_idx"].numpy())
        chain_idx = list(pdb["chain_letters"])
        seq_idx = list(pdb["S"].numpy())
        pdb_idx_list = []

        # Iterate over each parsed residue
        for ii, ri, ci, si in zip(icodes, res_idx, chain_idx, seq_idx):
            pdb_idx = str(ci) + str(ri) + str(ii)
            seq_str = restype_int_to_str[si]
            # Check if restype matches query
            sel_flag = seq_str in sel_restypes
            if args.flip:
                sel_flag = not sel_flag
            if sel_flag:
                pdb_idx_list.append(pdb_idx)
 
        data[path] = " ".join(pdb_idx_list)

    with open(args.json_file, "w", encoding="utf-8") as fopen:
        json.dump(data, fopen, ensure_ascii=False, indent=4)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
        Parses a directory of PDBs and generates a 'multi-PDB JSON' compatiable with the --fixed_residues_multi or --redesigned_residues_multi options.
    """)
    parser.add_argument("--pdb_dir", type=str, default="./", help="PDB directory to parse.")
    parser.add_argument("--json_file", type=str, default="sel_res_multi.json", help="Name of JSON file to produce.")
    parser.add_argument("--sel_restypes", type=str, default="G,X", help="Which restypes to select. Default is 'G,X'.")
    parser.add_argument("--flip", action="store_true", help="Whether to flip selection logic. If added, script will select all restypes EXCEPT --sel_restypes.")
    main(parser.parse_args())
