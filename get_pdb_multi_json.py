import argparse
import os
import json

def main(args):
    d = args.pdb_dir
    if not os.path.exists(d):
        raise ValueError(f"JSON generation failed - input dir {d} does not exist!")
    
    pdbs = sorted([p for p in os.listdir(d) if p.endswith(".pdb")])
    data = {}
    for p in pdbs:
        path = os.path.realpath(os.path.join(d, p))
        data[path] = ""

    with open(args.json_file, "w", encoding="utf-8") as fopen:
        json.dump(data, fopen, ensure_ascii=False, indent=4)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
        Parses a directory of PDB files and produces a 'multi-PDB JSON' that can be used with the --pdb_path_multi option.
    """)
    parser.add_argument("--pdb_dir", type=str, default="./", help="PDB directory to parse.")
    parser.add_argument("--json_file", type=str, default="pdb_multi.json", help="Name of JSON file to produce.")
    main(parser.parse_args())
