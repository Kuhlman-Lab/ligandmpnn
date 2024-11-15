import json
import re
import sys, os
import argparse
import math
import numpy as np
from typing import Optional, Sequence, Tuple, Dict
from Bio.PDB import PDBParser


class FileArgumentParser(argparse.ArgumentParser):
    """Overwrites default ArgumentParser to better handle flag files."""

    def convert_arg_line_to_args(self, arg_line: str) -> Optional[Sequence[str]]:
        """ Read from files where each line contains a flag and its value, e.g.
        '--flag value'. Also safely ignores comments denoted with '#' and 
        empty lines.
        """
        
        # Remove any comments from the line
        arg_line = arg_line.split('#')[0]
        
        # Escape if the line is empty
        if not arg_line:
            return None

        # Separate flag and values
        split_line = arg_line.strip().split(' ')
        
        # If there is actually a value, return the flag value pair
        if len(split_line) > 1:
            return [split_line[0], ' '.join(split_line[1:])]
        # Return just flag if there is no value
        else:
            return split_line


class ProteinDesignInputFormatter(object):
    def __init__(self, pdb_dir: str, designable_res: str = '', default_design_setting: str = 'all', 
                 symmetric_res: str = '', cluster_center: str = '', cluster_radius: float = 10.0) -> None:
        self.CHAIN_IDS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        self.AA3 = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 
                    'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'XXX']
        self.AA1 = list('ACDEFGHIKLMNPQRSTVWYX')
        self.AA3_to_AA1 = {aa3: aa1 for aa3, aa1 in zip(self.AA3, self.AA1)}
        
        if not os.path.isdir(pdb_dir):
            raise ValueError(f'The pdb_dir {pdb_dir} is does not exist.')
        else:
            pdb_list = [file for file in os.listdir(pdb_dir) if file[-3:]=='pdb']
            if len(pdb_list) < 1:
                raise ValueError(f'The pdb_dir {pdb_dir} does not contain any .pdb files.')
            elif len(pdb_list) > 1:
                raise ValueError(f'The pdb_dir {pdb_dir} contains more than 1 .pdb file. Currently this is not supported.')
            else:
                self.pdb_dir = pdb_dir
                self.pdb_list = pdb_list
                self.parser = PDBParser(QUIET=True)

        self.design_default = default_design_setting
        
        if designable_res:
            self.design_res = self.parse_designable_res(designable_res)
        else:
            self.design_res = []
            
        if cluster_center:
            cluster_mut = self.parse_cluster_center(cluster_center, cluster_radius)
            self.design_res += cluster_mut

        if symmetric_res:
            self.symmetric_res = self.parse_symmetric_res(symmetric_res)
        else:
            self.symmetric_res = []

    def _check_res_validity(self, res_item: str) -> Tuple[str, int]:
        split_item = [item for item in re.split('(\d+)', res_item) if item]
        
        if len(split_item) != 2:
            raise ValueError(f'Unable to parse residue: {res_item}.')
        if split_item[0] not in self.CHAIN_IDS:
            raise ValueError(f'Unknown chain id in residue: {res_item}')
        return (split_item[0], int(split_item[1]))

    def _check_range_validity(self, range_item: str) -> Sequence[Tuple[str, int]]:
        split_range = range_item.split('-')
        if len(split_range) != 2:
            raise ValueError(f'Unable to parse residue range: {range_item}')

        start_item, finish_item = split_range[0], split_range[1]

        s_chain, s_idx = self._check_res_validity(start_item)
        f_chain, f_idx = self._check_res_validity(finish_item)
        if s_chain != f_chain:
            raise ValueError(f'Residue ranges cannot span multiple chains: {range_item}')
        if s_idx >= f_idx:
            raise ValueError(f'Residue range starting index must be smaller than the ending index: '
                             f'{range_item}')

        res_range = []
        for i in range(s_idx, f_idx + 1):
            res_range.append( (s_chain, i) )

        return res_range
        
    def _check_symmetry_validity(self, symmetric_item: str) -> Dict[str, Sequence[Tuple[str, int]]]:        
        split_item = symmetric_item.split(':')

        symmetry_dict = {}
        for subitem in split_item:
            if '-' in subitem:
                res_range = self._check_range_validity(subitem)
                symmetry_dict[subitem] = res_range
            else:
                res_ch, res_idx = self._check_res_validity(subitem)
                symmetry_dict[subitem] = [(res_ch, res_idx)]

        item_lens = [len(symmetry_dict[key]) for key in symmetry_dict]
        if math.floor(sum([l == item_lens[0] for l in item_lens])/len(item_lens)) != 1:
            raise ValueError(f'Tied residues and residue ranges must be of the same '
                             f'size for forcing symmetry: {symmetric_item}')

        return symmetry_dict

    def _get_cluster_neighbors(self, center: str, cluster_radius: float) -> Sequence[str]:
        
        # Chain and residue number for center
        center_ch, center_res = center
        
        for pdb in self.pdb_list:
            # Get pdb structure
            pdb_id = pdb[:-4]
            pdb_file = os.path.join(self.pdb_dir, pdb)
            pdb_struct = self.parser.get_structure(id=pdb_id, file=pdb_file)
            
            # Get list of chains
            chains = list(pdb_struct.get_chains())
            # Determine cluster location
            for chain in chains:
                # Find correct chain
                if chain.id == center_ch:
                    for residue in chain.get_residues():
                        # Find correct residue
                        if residue.id[1] == center_res:
                            for atom in residue.get_atoms():
                                # Find the CA atom and get its coords
                                if atom.get_name() == 'CA':
                                    center_pos = atom.get_coord()
                                    break
                            break
                    break

            # Determine neighbors using cluster radius        
            neighbors = []
            for chain in chains:
                for residue in chain.get_residues():
                    # Residue chain and number
                    res_ch, res_num = chain.id, residue.id[1]
                    
                    for atom in residue.get_atoms():
                        # Find the CA atom and compute distance from center
                        if atom.get_name() == 'CA':
                            dist2center = np.sqrt(np.sum((atom.get_coord() - center_pos) ** 2) + 1e-12)
                            # If distance is less than radius, then add to neighbor list
                            if dist2center <= cluster_radius:
                                neighbors.append( (res_ch, res_num) )
                                
        return neighbors
    
    def parse_cluster_center(self, cluster_center: str, cluster_radius: float) -> Sequence[str]:
        
        cluster_center = [s for s in cluster_center.strip().split(',') if s]
        
        centers = []
        for item in cluster_center:
            if "-" not in item:
                item_ch, item_idx = self._check_res_validity(item)
                centers.append( (item_ch, item_idx) )
            else:
                range_res = self._check_range_validity(item)
                centers += range_res
        
        design_res = []    
        for center in centers:
            design_res += self._get_cluster_neighbors(center, cluster_radius)
                
        return design_res

    def parse_designable_res(self, design_str: str) -> Sequence[str]:
    
        design_str = [s for s in design_str.strip().split(",") if s]

        design_res = []
        for item in design_str:
            if "-" not in item:
                item_ch, item_idx = self._check_res_validity(item)
                design_res.append( (item_ch, item_idx) )
            else:
                range_res = self._check_range_validity(item)
                design_res += range_res
        
        return design_res

    def parse_symmetric_res(self, symmetric_str: str) -> Sequence[str]:

        symmetric_str = [s for s in symmetric_str.strip().split(",") if s]
        
        symmetric_res = []
        for item in symmetric_str:
            if ":" not in item:
                raise ValueError(f'No colon detected in symmetric res: {item}.')

            symmetry_dict = self._check_symmetry_validity(item)
            symmetric_res.append(symmetry_dict)
        
        return symmetric_res

    def generate_jsondata(self):

        pdb_dicts = []
        for pdb in self.pdb_list:
            pdb_id = pdb[:-4]
            pdb_file = os.path.join(self.pdb_dir, pdb)
            pdb_struct = self.parser.get_structure(id=pdb_id, file=pdb_file)

            chains = list(pdb_struct.get_chains())

            pdbids = {}
            chain_seqs = {}

            for chain in chains:
                chain_seqs[chain.id] = []
                res_index_chain = 1
                residues = list(chain.get_residues())

                for residue in residues:
                    # Get residue PDB number
                    num_id = residue.id[1]

                    # Record pdbid. E.g. A10
                    pdbid = chain.id + str(num_id)

                    # Add gapped residues to dictionary
                    if residue != residues[0]:

                        # Determine number of gapped residues
                        n_gaps = 0
                        while True:
                            prev_res = chain.id + str(num_id - n_gaps - 1)
                            if prev_res not in pdbids:
                                n_gaps += 1
                            else:
                                break

                        for i in range(n_gaps):
                            # Determine pdb id of gap residue
                            prev_res = chain.id + str(num_id - n_gaps + i)

                            # Update residue id dict with (X, residue chain, chain_index)
                            pdbids[prev_res] = (self.AA3_to_AA1['XXX'], chain.id, res_index_chain)

                            # Update chain sequence with X
                            chain_seqs[chain.id].append(self.AA3_to_AA1['XXX'])

                            # Increment
                            res_index_chain += 1

                    # Update dict with (residue name, residue chain, chain index)
                    pdbids[pdbid] = (self.AA3_to_AA1.get(residue.get_resname(), 'XXX'), chain.id, res_index_chain)

                    # Add to the chain_sequence
                    chain_seqs[chain.id].append(self.AA3_to_AA1.get(residue.get_resname(), 'XXX'))

                    # Update the chain index
                    res_index_chain += 1

                # Make the list of AA1s into a single string for the chain sequence
                chain_seqs[chain.id] = "".join([x for x in chain_seqs[chain.id] if x is not None])

            mutable = []
            for resind in self.design_res:
                res_id = resind[0] + str(resind[1])
                if pdbids[res_id][0] != 'X':
                    mutable.append({"chain": pdbids[res_id][1], "resid": pdbids[res_id][2], 
                                    "WTAA": pdbids[res_id][0], "MutTo": self.design_default})

            symmetric = []
            for symmetry in self.symmetric_res:
                values = list(symmetry.values())

                for tied_pos in zip(*values):
                    skip_tie = False
                    sym_res = []
                    for pos in tied_pos:
                        res_id = pos[0] + str(pos[1])
                        if pdbids[res_id][0] == 'X':
                            skip_tie = True
                            break
                        else:
                            sym_res.append( pdbids[res_id][1] + str(pdbids[res_id][2]) )

                    if not skip_tie:
                        symmetric.append(sym_res)

            dictionary = {"sequence" : chain_seqs, "designable": mutable, "symmetric": symmetric}
            pdb_dicts.append(dictionary)

        #with open(out_path, 'w') as f:
        #    for pdb_dict in pdb_dicts:
        #        f.write(json.dumps(pdb_dict, indent=2) + '\n')
        return pdb_dicts


def get_arguments() -> argparse.Namespace:
    """ Uses FileArgumentParser to parse commandline options to generate a json
    file for input into ProteinMPNN."""

    # Construct the parser and its arguments.
    parser = FileArgumentParser(description='Script that takes a PDB and design'
                                'specifications to create a json file for input'
                                'to ProteinMPNN', 
                                fromfile_prefix_chars='@')

    parser.add_argument('--pdb_dir',
                        type=str,
                        help='Path to the directory containing PDB files.')
    parser.add_argument('--designable_res',
                        default='',
                        type=str,
                        help='PDB chain and residue numbers to mutate, separated by '
                        'commas and/or hyphens. E.g. A10,A12-A15')
    parser.add_argument('--default_design_setting',
                        default='all',
                        type=str,
                        help="Default setting amino acid types that residues are "
                        "allowed to mutate to. Use 'all' to allow any amino acid "
                        "to be design. Default is 'all'.")
    parser.add_argument('--symmetric_res',
                        default='',
                        type=str,
                        help="PDB chain and residue numbers to force symmetric "
                        "design, separated by colons and commas. E.g. to force "
                        "symmetry between residue A1 and A15 use 'A1:A15' and for "
                        "symmetry between residues 1-5 on chains A and B use "
                        "'A1-A5:B1-B15'. (Note that the number of residues on"
                        " each side of the colon must be the same).")
    parser.add_argument('--cluster_center',
                        default='',
                        type=str,
                        help="PDB chain and residue numbers which will serve as "
                        "mutation centers. Every residue with a CA within "
                        "--cluster_radius Angstroms will be mutatable")
    parser.add_argument('--cluster_radius',
                        default=10.0,
                        type=float,
                        help="Radius from cluster mutation centers in which to "
                        "include residues for mutation. Default is 10.0 A.")
    parser.add_argument('--out_path',
                        default='proteinmpnn_res_specs.json',
                        type=str,
                        help="Path for output json file. Default is "
                        "proteinmpnn_res_specs.json.")

    # Parse the provided arguments
    args = parser.parse_args(sys.argv[1:])

    return args
    

if __name__=="__main__":
    args = get_arguments()

    pdif = ProteinDesignInputFormatter(args.pdb_dir, args.designable_res, args.default_design_setting, 
                                       args.symmetric_res, args.cluster_center, args.cluster_radius)
    j = pdif.generate_jsondata()[0]
    residues = [str(x['chain']) + str(x['resid']) for x in j['designable']]
    print(" ".join(residues))
