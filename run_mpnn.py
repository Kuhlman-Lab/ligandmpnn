import copy
import json
import os.path
import random
import sys
import argparse
import omegaconf

import numpy as np
import torch
from data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    parse_PDB,
    restype_1to3,
    restype_int_to_str,
    restype_str_to_int,
    write_full_PDB,
)
from model_utils import ProteinMPNN
from prody import writePDB
from sc_utils import Packer, pack_side_chains

#@hydra.main(version_base=None, config_path='configs/', config_name='mpnn_basic')
def main(conf, design_run = False, json_data=None, pdb_paths=None) -> None:
    """
    Inference function
    """
    
    if conf.inference.seed:
        seed = conf.inference.seed
    else:
        seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    folder_for_outputs = conf.inference.out_folder
    base_folder = folder_for_outputs
    if base_folder[-1] != "/":
        base_folder = base_folder + "/"
    if not design_run:
        # Disable outputs if design_run
        if not os.path.exists(base_folder):
            os.makedirs(base_folder, exist_ok=True)
        if not os.path.exists(base_folder + "seqs"):
            os.makedirs(base_folder + "seqs", exist_ok=True)
        if not os.path.exists(base_folder + "backbones"):
            os.makedirs(base_folder + "backbones", exist_ok=True)
        if not os.path.exists(base_folder + "packed"):
            os.makedirs(base_folder + "packed", exist_ok=True)
        if conf.inference.save_stats:
            if not os.path.exists(base_folder + "stats"):
                os.makedirs(base_folder + "stats", exist_ok=True)
    if conf.inference.model_type == "protein_mpnn":
        checkpoint_path = conf.checkpoint_protein_mpnn
    elif conf.inference.model_type == "ligand_mpnn":
        checkpoint_path = conf.checkpoint_ligand_mpnn
    elif conf.inference.model_type == "per_residue_label_membrane_mpnn":
        checkpoint_path = conf.checkpoint_per_residue_label_membrane_mpnn
    elif conf.inference.model_type == "global_label_membrane_mpnn":
        checkpoint_path = conf.checkpoint_global_label_membrane_mpnn
    elif conf.inference.model_type == "soluble_mpnn":
        checkpoint_path = conf.checkpoint_soluble_mpnn
    else:
        print("Choose one of the available models")
        sys.exit()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if conf.inference.model_type == "ligand_mpnn":
        atom_context_num = checkpoint["atom_context_num"]
        ligand_mpnn_use_side_chain_context = conf.inference.ligand_mpnn_use_side_chain_context
        k_neighbors = checkpoint["num_edges"]
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0
        k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=conf.inference.model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    ) #TODO: augment_eps to enable backbone noise

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if conf.inference.pack_side_chains:
        model_sc = Packer(
            node_features=128,
            edge_features=128,
            num_positional_embeddings=16,
            num_chain_embeddings=16,
            num_rbf=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            atom_context_num=16,
            lower_bound=0.0,
            upper_bound=20.0,
            top_k=32,
            dropout=0.0,
            augment_eps=0.0,
            atom37_order=False,
            device=device,
            num_mix=3,
        )

        checkpoint_sc = torch.load(conf.checkpoint_path_sc, map_location=device)
        model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
        model_sc.to(device)
        model_sc.eval()
    
    if not pdb_paths:
        if conf.inference.pdb_path_multi:
            with open(conf.inference.pdb_path_multi, "r") as fh:
                pdb_paths = list(json.load(fh))
        elif conf.inference.pdb_path:
            pdb_paths = [conf.inference.pdb_path]
    
    if not json_data:
        if conf.inference.json_file:
            with open(conf.inference.json_file, "r") as fh:
                json_data = json.load(fh)
                
    fixed_residues_multi = {}
    redesigned_residues_multi = {}
    omit_AA_per_residue_multi = {}

    fixed_residues = None
    redesigned_residues = None
    
    if conf.inference.omit_AA:
        omit_AA_list = conf.inference.omit_AA
    else:
        omit_AA_list = []
    omit_AA = torch.tensor(np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32), device=device)

    if json_data:
        #get redesigned residues from json
        redesigned_residues = []
        symmetry_residues_list_of_lists = []
        symmetry_weights = []
        
        omit_AA_per_residue = {}
        for d in json_data["designable"]:
            redesigned_residues.append(d["chain"] + str(d["resid"]))
            mut_to = list(d["MutTo"])
            omit = [x for x in alphabet if x not in mut_to]
            omit_AA_per_residue[d["chain"] + str(d["resid"])] = omit
            
        for pdb in pdb_paths:
            redesigned_residues_multi[pdb] = redesigned_residues
            omit_AA_per_residue_multi[pdb] = omit_AA_per_residue
            
        for sym in json_data["symmetric"]:
            symmetry_residues_list_of_lists.append(sym)
            symmetry_weights.append([1.0 for x in sym])
        if len(symmetry_residues_list_of_lists) == 0:
            symmetry_residues_list_of_lists = [[]]
            symmetry_weights = [[]]
            
    else:

        if conf.inference.fixed_residues_multi:
            with open(conf.inference.fixed_residues_multi, "r") as fh:
                fixed_residues_multi = json.load(fh)
        elif conf.inference.fixed_residues:
            fixed_residues = [item for item in conf.inference.fixed_residues.split()]
            for pdb in pdb_paths:
                fixed_residues_multi[pdb] = fixed_residues

        if conf.inference.redesigned_residues_multi:
            with open(conf.inference.redesigned_residues_multi, "r") as fh:
                redesigned_residues_multi = json.load(fh)
        elif conf.inference.redesigned_residues:
            redesigned_residues = [item for item in conf.inference.redesigned_residues.split()]
            for pdb in pdb_paths:
                redesigned_residues_multi[pdb] = redesigned_residues

        if conf.inference.omit_AA_per_residue_multi:
            with open(conf.inference.omit_AA_per_residue_multi, "r") as fh:
                omit_AA_per_residue_multi = json.load(
                    fh
                )  # {"pdb_path" : {"A12": "PQR", "A13": "QS"}}
        else:
            if conf.inference.omit_AA_per_residue:
                with open(conf.inference.omit_AA_per_residue, "r") as fh:
                    omit_AA_per_residue = json.load(fh)  # {"A12": "PG"}
            else:
                omit_AA_per_residue = {}
            for pdb in pdb_paths:
                omit_AA_per_residue_multi[pdb] = omit_AA_per_residue

    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    if conf.inference.bias_AA:
        tmp = [item.split(":") for item in conf.inference.bias_AA.split(",")]
        a1 = [b[0] for b in tmp]
        a2 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            bias_AA[restype_str_to_int[AA]] = a2[i]

    if conf.inference.bias_AA_per_residue_multi:
        with open(conf.inference.bias_AA_per_residue_multi, "r") as fh:
            bias_AA_per_residue_multi = json.load(
                fh
            )  # {"pdb_path" : {"A12": {"G": 1.1}}}
    else:
        if conf.inference.bias_AA_per_residue:
            with open(conf.inference.bias_AA_per_residue, "r") as fh:
                bias_AA_per_residue = json.load(fh)  # {"A12": {"G": 1.1}}
            bias_AA_per_residue_multi = {}
            for pdb in pdb_paths:
                bias_AA_per_residue_multi[pdb] = bias_AA_per_residue
                    
    # loop over PDB paths
    full_output_dict = {}
    for pdb in pdb_paths:
        if conf.inference.verbose:
            print("Designing protein from this path:", pdb)
        if pdb in fixed_residues_multi:
            fixed_residues = fixed_residues_multi[pdb]
        else:
            fixed_residues = []
        if pdb in redesigned_residues_multi:
            redesigned_residues = redesigned_residues_multi[pdb]
        else:
            redesigned_residues = []
        parse_all_atoms_flag = conf.inference.ligand_mpnn_use_side_chain_context or (
            conf.inference.pack_side_chains and not conf.inference.repack_everything
        )
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb,
            device=device,
            chains=conf.inference.parse_these_chains_only,
            parse_all_atoms=parse_all_atoms_flag,
            parse_atoms_with_zero_occupancy=conf.inference.parse_atoms_with_zero_occupancy,
        )
        # make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(
            zip(list(range(len(encoded_residues))), encoded_residues)
        )

        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if conf.inference.bias_AA_per_residue_multi or conf.inference.bias_AA_per_residue:
            bias_dict = bias_AA_per_residue_multi[pdb]
            for residue_name, v1 in bias_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            bias_AA_per_residue[i1, j1] = v2


        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        
        omit_dict = omit_AA_per_residue_multi[pdb]
        for residue_name, v1 in omit_dict.items():
            if residue_name in encoded_residues:
                i1 = encoded_residue_dict[residue_name]
                for amino_acid in v1:
                    if amino_acid in alphabet:
                        j1 = restype_str_to_int[amino_acid]
                        omit_AA_per_residue[i1, j1] = 1.0

        if not json_data:

            # specify which residues are linked
            if conf.inference.symmetry_residues:
                symmetry_residues_list_of_lists = [
                    x.split(",") for x in conf.inference.symmetry_residues.split("|")
                ]
            else:
                remapped_symmetry_residues = [[]]

            # specify linking weights
            if conf.inference.symmetry_weights:
                symmetry_weights = [
                    [float(item) for item in x.split(",")]
                    for x in conf.inference.symmetry_weights.split("|")
                ]
            else:
                symmetry_weights = [[]]

            if conf.inference.homo_oligomer:
                if conf.inference.verbose:
                    print("Designing HOMO-OLIGOMER")
                chain_letters_set = list(set(chain_letters_list))
                reference_chain = chain_letters_set[0]
                lc = len(reference_chain)
                residue_indices = [
                    item[lc:] for item in encoded_residues if item[:lc] == reference_chain
                ]
                remapped_symmetry_residues = []
                symmetry_weights = []
                for res in residue_indices:
                    tmp_list = []
                    tmp_w_list = []
                    for chain in chain_letters_set:
                        name = chain + res
                        tmp_list.append(encoded_residue_dict[name])
                        tmp_w_list.append(1 / len(chain_letters_set))
                    remapped_symmetry_residues.append(tmp_list)
                    symmetry_weights.append(tmp_w_list)
        else:
            pass
        
        remapped_symmetry_residues = []
        for t_list in symmetry_residues_list_of_lists:
            tmp_list = []
            for t in t_list:
                tmp_list.append(encoded_residue_dict[t])
            remapped_symmetry_residues.append(tmp_list)
    
        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=device,
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=device,
        )

        # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
        if conf.inference.transmembrane_buried:
            buried_residues = [item for item in conf.inference.transmembrane_buried.split()]
            buried_positions = torch.tensor(
                [int(item in buried_residues) for item in encoded_residues],
                device=device,
            )
        else:
            buried_positions = torch.zeros_like(fixed_positions)

        if conf.inference.transmembrane_interface:
            interface_residues = [item for item in conf.inference.transmembrane_interface.split()]
            interface_positions = torch.tensor(
                [int(item in interface_residues) for item in encoded_residues],
                device=device,
            )
        else:
            interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if conf.inference.model_type == "global_label_membrane_mpnn":
            protein_dict["membrane_per_residue_labels"] = (
                conf.inference.global_transmembrane_label + 0 * fixed_positions
            )
        if type(conf.inference.chains_to_design) == str:
            chains_to_design_list = conf.inference.chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=device,
        )

        # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        if conf.inference.verbose:
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)


        # set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors * 0.0)

        # adjust input PDB name by dropping .pdb if it does exist
        name = pdb[pdb.rfind("/") + 1 :]
        if name[-4:] == ".pdb":
            name = name[:-4]
            
        out_seqs = []

        with torch.no_grad():
            # run featurize to remap R_idx and add batch dimension
            if conf.inference.verbose:
                if "Y" in list(protein_dict):
                    atom_coords = protein_dict["Y"].cpu().numpy()
                    atom_types = list(protein_dict["Y_t"].cpu().numpy())
                    atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                    number_of_atoms_parsed = np.sum(atom_mask)
                else:
                    print("No ligand atoms parsed")
                    number_of_atoms_parsed = 0
                    atom_types = ""
                    atom_coords = []
                if number_of_atoms_parsed == 0:
                    print("No ligand atoms parsed")
                elif conf.inference.model_type == "ligand_mpnn":
                    print(
                        f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                    )
                    for i, atom_type in enumerate(atom_types):
                        print(
                            f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                        )
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=conf.inference.ligand_mpnn_cutoff_for_score,
                use_atom_context=conf.inference.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=atom_context_num,
                model_type=conf.inference.model_type,
            )
            feature_dict["batch_size"] = conf.inference.batch_size
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            # add additional keys to the feature dictionary
            feature_dict["temperature"] = conf.inference.temperature
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights

            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []
            for _ in range(conf.inference.number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=device,
                )
                # print("feature_dict")
                # for feat in feature_dict:
                #     if type(feature_dict[feat]) == torch.Tensor:
                #         print(feat, feature_dict[feat].shape)
                #     else:
                #         print(feat, feature_dict[feat])
                output_dict = model.sample(feature_dict)

                # compute confidence scores
                loss, loss_per_residue = get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"],
                )
                if conf.inference.model_type == "ligand_mpnn":
                    combined_mask = (
                        feature_dict["mask"]
                        * feature_dict["mask_XY"]
                        * feature_dict["chain_mask"]
                    )
                else:
                    combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
                loss_XY, _ = get_score(
                    output_dict["S"], output_dict["log_probs"], combined_mask
                )
                # -----
                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)
            S_stack = torch.cat(S_list, 0)
            log_probs_stack = torch.cat(log_probs_list, 0)
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
            rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

            native_seq = "".join(
                [restype_int_to_str[AA] for AA in feature_dict["S"][0].cpu().numpy()]
            )
            seq_np = np.array(list(native_seq))
            seq_out_str = []
            for mask in protein_dict["mask_c"]:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += [conf.inference.fasta_seq_separation]
            seq_out_str = "".join(seq_out_str)[:-1]

            if not conf.inference.file_ending:
                conf.inference.file_ending = ""
            output_fasta = base_folder + "/seqs/" + name + conf.inference.file_ending + ".fa"
            output_backbones = base_folder + "/backbones/"
            output_packed = base_folder + "/packed/"
            output_stats_path = base_folder + "stats/" + name + conf.inference.file_ending + ".pt"

            out_dict = {}
            out_dict["generated_sequences"] = S_stack.cpu()
            out_dict["sampling_probs"] = sampling_probs_stack.cpu()
            out_dict["log_probs"] = log_probs_stack.cpu()
            out_dict["decoding_order"] = decoding_order_stack.cpu()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu()
            out_dict["mask"] = feature_dict["mask"][0].cpu()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
            out_dict["seed"] = seed
            out_dict["temperature"] = conf.inference.temperature
            full_output_dict[pdb] = out_dict

            if conf.inference.save_stats and not design_run:
                torch.save(out_dict, output_stats_path)

            if conf.inference.pack_side_chains:
                if conf.inference.verbose:
                    print("Packing side chains...")
                feature_dict_ = featurize(
                    protein_dict,
                    cutoff_for_score=8.0,
                    use_atom_context=conf.inference.pack_with_ligand_context,
                    number_of_ligand_atoms=16,
                    model_type="ligand_mpnn",
                )
                sc_feature_dict = copy.deepcopy(feature_dict_)
                B = conf.inference.batch_size
                for k, v in sc_feature_dict.items():
                    if k != "S":
                        try:
                            num_dim = len(v.shape)
                            if num_dim == 2:
                                sc_feature_dict[k] = v.repeat(B, 1)
                            elif num_dim == 3:
                                sc_feature_dict[k] = v.repeat(B, 1, 1)
                            elif num_dim == 4:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                            elif num_dim == 5:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
                        except:
                            pass
                X_stack_list = []
                X_m_stack_list = []
                b_factor_stack_list = []
                for _ in range(conf.inference.number_of_packs_per_design):
                    X_list = []
                    X_m_list = []
                    b_factor_list = []
                    for c in range(conf.inference.number_of_batches):
                        sc_feature_dict["S"] = S_list[c]
                        sc_dict = pack_side_chains(
                            sc_feature_dict,
                            model_sc,
                            conf.inference.sc_num_denoising_steps,
                            conf.inference.sc_num_samples,
                            conf.inference.repack_everything,
                        )
                        X_list.append(sc_dict["X"])
                        X_m_list.append(sc_dict["X_m"])
                        b_factor_list.append(sc_dict["b_factors"])

                    X_stack = torch.cat(X_list, 0)
                    X_m_stack = torch.cat(X_m_list, 0)
                    b_factor_stack = torch.cat(b_factor_list, 0)

                    X_stack_list.append(X_stack)
                    X_m_stack_list.append(X_m_stack)
                    b_factor_stack_list.append(b_factor_stack)
            
            for ix in range(S_stack.shape[0]):
                ix_suffix = ix
                if not conf.inference.zero_indexed:
                    ix_suffix += 1
                seq_rec_print = np.format_float_positional(
                    rec_stack[ix].cpu().numpy(), unique=False, precision=4
                )
                loss_np = np.format_float_positional(
                    np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4
                )
                loss_XY_np = np.format_float_positional(
                    np.exp(-loss_XY_stack[ix].cpu().numpy()),
                    unique=False,
                    precision=4,
                )
                seq = "".join(
                    [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
                )

                # write new sequences into PDB with backbone coordinates
                seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])[
                    None,
                ].repeat(4, 1)
                bfactor_prody = (
                    loss_per_residue_stack[ix].cpu().numpy()[None, :].repeat(4, 1)
                )
                backbone.setResnames(seq_prody)
                backbone.setBetas(
                    np.exp(-bfactor_prody)
                    * (bfactor_prody > 0.01).astype(np.float32)
                )
                
            if not design_run:
                # Disable outputs if design_run
                with open(output_fasta, "w") as f:
                    f.write(
                        ">{}, T={}, seed={}, num_res={}, num_ligand_res={}, use_ligand_context={}, ligand_cutoff_distance={}, batch_size={}, number_of_batches={}, model_path={}\n{}\n".format(
                            name,
                            conf.inference.temperature,
                            seed,
                            torch.sum(rec_mask).cpu().numpy(),
                            torch.sum(combined_mask[:1]).cpu().numpy(),
                            bool(conf.inference.ligand_mpnn_use_atom_context),
                            float(conf.inference.ligand_mpnn_cutoff_for_score),
                            conf.inference.batch_size,
                            conf.inference.number_of_batches,
                            checkpoint_path,
                            seq_out_str,
                        )
                    )
                    for ix in range(S_stack.shape[0]):
                        if other_atoms:
                            writePDB(
                                output_backbones
                                + name
                                + "_"
                                + str(ix_suffix)
                                + conf.inference.file_ending
                                + ".pdb",
                                backbone + other_atoms,
                            )
                        else:
                            writePDB(
                                output_backbones
                                + name
                                + "_"
                                + str(ix_suffix)
                                + conf.inference.file_ending
                                + ".pdb",
                                backbone,
                            )

                        # write full PDB files
                        if conf.inference.pack_side_chains:
                            for c_pack in range(conf.inference.number_of_packs_per_design):
                                X_stack = X_stack_list[c_pack]
                                X_m_stack = X_m_stack_list[c_pack]
                                b_factor_stack = b_factor_stack_list[c_pack]
                                write_full_PDB(
                                    output_packed
                                    + name
                                    + conf.inference.packed_suffix
                                    + "_"
                                    + str(ix_suffix)
                                    + "_"
                                    + str(c_pack + 1)
                                    + conf.inference.file_ending
                                    + ".pdb",
                                    X_stack[ix].cpu().numpy(),
                                    X_m_stack[ix].cpu().numpy(),
                                    b_factor_stack[ix].cpu().numpy(),
                                    feature_dict["R_idx_original"][0].cpu().numpy(),
                                    protein_dict["chain_letters"],
                                    S_stack[ix].cpu().numpy(),
                                    other_atoms=other_atoms,
                                    icodes=icodes,
                                    force_hetatm=conf.inference.force_hetatm,
                                )
                        # -----

                        # write fasta lines
                        seq_np = np.array(list(seq))
                        seq_out_str = []
                        for mask in protein_dict["mask_c"]:
                            seq_out_str += list(seq_np[mask.cpu().numpy()])
                            seq_out_str += [conf.inference.fasta_seq_separation]
                        seq_out_str = "".join(seq_out_str)[:-1]
                        if ix == S_stack.shape[0] - 1:
                            # final 2 lines
                            f.write(
                                ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}".format(
                                    name,
                                    ix_suffix,
                                    conf.inference.temperature,
                                    seed,
                                    loss_np,
                                    loss_XY_np,
                                    seq_rec_print,
                                    seq_out_str,
                                )
                            )
                        else:
                            f.write(
                                ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}\n".format(
                                    name,
                                    ix_suffix,
                                    conf.inference.temperature,
                                    seed,
                                    loss_np,
                                    loss_XY_np,
                                    seq_rec_print,
                                    seq_out_str,
                                )
                            )
            else:
                    
                for ix in range(S_stack.shape[0]):   
                    # extract fasta lines
                    seq_np = np.array(list(seq))
                    seq_out_str = []
                    for mask in protein_dict["mask_c"]:
                        seq_out_str += list(seq_np[mask.cpu().numpy()])
                        seq_out_str += [conf.inference.fasta_seq_separation]
                    seq_out_str = "".join(seq_out_str)[:-1]
                    out_seqs.append(seq_out_str)
                            
    
    if design_run:
        return out_seqs
                        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--config_file",
        type=str,
        default="./mpnn_basic.yaml",
        help="path to yaml config file to load options",
    )
    argparser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="path to json file to load design specs. overwrites config file json file",
    )
    argparser.add_argument(
        "--pdb_path",
        type=str,
        default=None,
        help="path to pdb file to load backbone for design. overwrites config file pdb path",
    )
    args = argparser.parse_args()
    
    # Load config
    jsondata = None
    if args.json_file:
        with open(args.json_file, "r") as fh:
            jsondata = json.load(fh)
    conf = omegaconf.OmegaConf.load(args.config_file)
    print(main(conf, design_run=True, json_data=jsondata, pdb_paths=[args.pdb_path]))
    # print(main(conf, design_run=True, json_data=args.json_file, pdb_paths=args.pdb_path))