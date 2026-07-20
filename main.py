from pathlib import Path
import time
import re
from functools import partial
import warnings

from data_import import (
    load_fasta_sequences,
    load_negative_binders)
from preprocessing import (
    deduplicate_per_cow,
    deduplicate_global,
    flatten_raw_sequences,
    check_negative_occurrence,
    extract_negative_binder_ids_from_fasta,
    build_candidate_set)
from io_utils import (
    save_negative_binders,
    save_unique_per_cow,
    save_unique_global_csv,
    save_raw_sequences,
    save_candidates,
    save_unique_global_fasta,
    build_seq_to_id_map,
    save_clusters,
    load_clusters,
    export_clusters_to_txt)
from general_sequence_analysis import (
    plot_length_distribution,
    plot_occurrence_distribution,
    run_clustalw_alignment_from_fasta,
    plot_gap_distribution,
    plot_entropy_distribution,
    plot_sequence_logo)
from cystein_sequence_analysis import (
    process_alignment_DSATYY_WGXG,
    export_gapfree_sequences_from_alignment,
    process_alignment_by_conserved_c_positions,
    evaluate_dedup_thresholds,
    deduplicate_and_filter_fasta,
    cysteine_clustering,
    plot_cluster_logos,
    cluster_summary_to_latex,
    plot_cysteine_position_heatmap,
    plot_cysteine_spacing_violin,
    compute_sequence_identity_matrix,
    export_full_knob_excel)
from structure_prediction_prep import (
    write_clustered_fasta,
    write_clustered_yaml,
    collect_disulfides_from_folder,
    map_disulfides_to_knobs,
    generate_valid_disulfide_constraints)
from structure_prediction_analysis import (
    get_negative_binder_ids_from_vx_name,
    build_comparison_and_tested_id_sets,
    collect_plddt_stats,
    get_max_residue_length,
    build_model_to_cluster_from_fasta,
    normalize_model_id,
    plot_plddt_landscape,
    collect_boltz_best_models,
    collect_af_best_models,
    build_af_model_to_cluster,
    filter_plddt_stats_by_ids,
    plot_plddt_landscape_groups,
    build_global_display_order,
    plot_mean_plddt_multi_models,
    collect_esm_chain_models,
    collect_boltz_chain_models,
    collect_af_chain_models,)
from MD_prep import (
    archive_and_clean_md_root,
    prepare_md_input_for_seq,
    distribute_md_scripts)
from MD_analysis import (
    get_md_analysis_runs,
    create_conditions_file,
    run_md_analysis,
    create_rmsd_analysis,
    create_rmsf_analysis,
    create_binder_target_distance_analysis,)


# ---------------Switches-----------------------
run_all_bool = False
DEBUG = False

general_sequence_analysis_bool = False
cysteine_sequence_analysis_bool = False
structure_prediction_prep_bool = False
structure_prediction_analysis_bool = False # does not work if old pdb files are present
MD_prep_bool = True
MD_analysis_bool = False

selected_seqs_MD = [
    {
        "seq_id": "seq_25",
        "model": "best",
        "ligand": False,
    },
    {
        "seq_id": "seq_1",
        "model": "best",
        "ligand": False,
    },
    {
        "seq_id": "seq_12",
        "model": "best",
        "ligand": False,
    },
    {
        "seq_id": "seq_78",
        "model": "best",
        "ligand": False,
    },
    {
        "seq_id": "seq_25",
        "model": "best",
        "ligand": True,
    },
    {
        "seq_id": "seq_1",
        "model": "best",
        "ligand": True,
    },
    {
        "seq_id": "seq_12",
        "model": "best",
        "ligand": True,
    },
    {
        "seq_id": "seq_78",
        "model": "best",
        "ligand": True,
    },
]


# ---------------Paths--------------------------
FASTA_PATH_SOURCE = Path(r"\\nas.ads.mwn.de\ge63laz\TUM-PC\Desktop\Masterthesis_Picobodies\Source_Structure Thesis\Picobody_Sequences_Anti-mClover\Fasta-Files")
EXCEL_PATH_SOURCE = Path(r"\\nas.ads.mwn.de\ge63laz\TUM-PC\Desktop\Masterthesis_Picobodies\Source_Structure Thesis\Picobody_Sequences_Anti-mClover\Overview_Sequences_Picobody_anti-mClover.xlsx")
aln_path_knobs_internship = Path(r"\\nas.ads.mwn.de\ge63laz\TUM-PC\Desktop\Masterthesis_Picobodies\Source_Structure Thesis\knobs_internship.aln")
pdb_folder_knobs_internship = Path(r"\\nas.ads.mwn.de\ge63laz\TUM-PC\Desktop\Masterthesis_Picobodies\Source_Structure Thesis\knobs_internship_structures")
txt_seq_mclover3 = Path(r"\\nas.ads.mwn.de\ge63laz\TUM-PC\Desktop\Masterthesis_Picobodies\Source_Structure Thesis\Seq_mClover3.txt")
seq_mclover3 = "MVSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTFGYGVACFSRYPDHMKQHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHYVYITADKQKNCIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSHQSKLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
clustalw_path = r"C:\Program Files (x86)\ClustalW2\clustalw2.exe"

md_root = Path(r"C:\Users\ge63laz\PycharmProjects\Masterthesis_Picobodies\pc_cluster_scripts\MolecularDynamics")
archive_root = Path(r"C:\Users\ge63laz\PyCharmProjects\Masterthesis_Picobodies\data_saved\molecular_dynamics")

save_dir_data = Path("data_saved")
save_dir_plots = Path(save_dir_data / "plots")
save_dir_variable_data = Path(save_dir_data / "variable_data_etc")
save_dir_structure_prediction = Path(save_dir_data / "structure_prediction")
save_dir_maestro = Path(r"\\nas.ads.mwn.de\ge63laz\TUM-PC\Desktop\Masterthesis_Picobodies\Maestro_Pred_Strc_Picobodies")

save_dir_esm = Path(save_dir_structure_prediction / "esm_fold")
save_dir_boltz = Path(save_dir_structure_prediction / "boltz2")
save_dir_af = Path(save_dir_structure_prediction / "af3")

save_dir_variable_data.mkdir(exist_ok=True)
save_dir_plots.mkdir(exist_ok=True)

if run_all_bool:
    general_sequence_analysis_bool = True
    cysteine_sequence_analysis_bool = True
    structure_prediction_prep_bool = True
    structure_prediction_analysis_bool = True
    MD_prep_bool = True
    MD_analysis_bool = True

print("--------------------Import--------------------")
start_time_main = time.time()
picobody_sequences = load_fasta_sequences(FASTA_PATH_SOURCE)
negative_binder_sequences = load_negative_binders(EXCEL_PATH_SOURCE)


# ---------------Preprocessing--------------------------
print("\n--------------------Preprocessing--------------------")
unique_per_cow = deduplicate_per_cow(picobody_sequences)
unique_global = deduplicate_global(picobody_sequences)
for cow, seq_set in unique_per_cow.items():
    print(f"{cow}: {len(seq_set)} unique sequences")
print(f"General: {len(unique_global)} unique sequences")

raw_sequences = flatten_raw_sequences(picobody_sequences)
neg_check = check_negative_occurrence(
    raw_sequences,
    unique_global,
    negative_binder_sequences)

save_negative_binders(negative_binder_sequences, save_dir_variable_data)
save_unique_per_cow(unique_per_cow, save_dir_variable_data)
save_unique_global_csv(unique_global, save_dir_variable_data)
unique_global_fasta = save_unique_global_fasta(unique_global, save_dir_variable_data)
save_raw_sequences(raw_sequences, save_dir_variable_data)

seq_to_id = build_seq_to_id_map(unique_global_fasta)
negative_binder_ids = extract_negative_binder_ids_from_fasta(unique_global_fasta,negative_binder_sequences=negative_binder_sequences)
candidate_sequences = build_candidate_set(unique_global, negative_binder_ids, seq_to_id)
save_candidates(candidate_sequences, save_dir_variable_data, seq_to_id)

print("Negative binders in raw data:", len(neg_check["neg_in_raw"]))
print("Negative binders in global unique:", len(neg_check["neg_in_global"]))
print("Negative binders NOT in global unique:", len(neg_check["neg_not_in_global"]))
print(f"Number of potential binders (candidates): {len(candidate_sequences)}")


# ---------------general sequence analysis--------------------------
if general_sequence_analysis_bool:
    print("\n--------------------General Sequence analysis--------------------")
    plot_length_distribution(unique_global, save_dir_plots)
    plot_occurrence_distribution(unique_global, save_dir_plots)
    print("Plots for length distribution and occurrence distribution were saved.")

    alignment_path = run_clustalw_alignment_from_fasta(
        fasta_files=[(save_dir_variable_data /"unique_global.fasta")],
        clustalw_path=clustalw_path,
        output_dir=save_dir_variable_data,
        output_prefix="unique_global_alignment")

    plot_gap_distribution(alignment_path, save_dir_plots)
    plot_entropy_distribution(alignment_path, save_dir_plots)
    plot_sequence_logo(alignment_path, save_dir_plots, include_gaps=False, plot_title="Logo of Global Unique Sequences")
    plot_sequence_logo(alignment_path, save_dir_plots, include_gaps=True, plot_title="Logo of Global Unique Sequences (with gaps)")
    print(f"Plots for {alignment_path} with the gap distribution, the entropy distribution and the logos (with and without gaps) were saved.")


# ---------------cystein sequence analysis--------------------------
if cysteine_sequence_analysis_bool:
    print("\n--------------------Cystein Sequence analysis--------------------")
    # delete obvious aa and align wird literature knobs, compare
    process_alignment_DSATYY_WGXG(
        aln_path=(save_dir_variable_data / "unique_global_alignment.aln"),
        output_fasta=(save_dir_variable_data /"cut_DSATYY_WGXG_unique.fasta"))
    export_gapfree_sequences_from_alignment( # knobs were only stored as aln data, converting to fasta
        aln_path=aln_path_knobs_internship,
        output_fasta=(save_dir_variable_data / f"{aln_path_knobs_internship.stem}_gapfree.fasta"))
    alignment_path = run_clustalw_alignment_from_fasta(
        fasta_files=[
            save_dir_variable_data / "knobs_internship_gapfree.fasta",
            save_dir_variable_data / "cut_DSATYY_WGXG_unique.fasta"],
        clustalw_path=clustalw_path,
        output_dir=save_dir_variable_data,
        output_prefix="cut_DSATYY_WGXG_unique_vs_structure_picobodies")

    plot_gap_distribution(alignment_path, save_dir_plots)
    plot_entropy_distribution(alignment_path, save_dir_plots)
    plot_sequence_logo(alignment_path, save_dir_plots, include_gaps=False, plot_title="Logo of Cut Global Unique Sequences")
    plot_sequence_logo(alignment_path, save_dir_plots, include_gaps=True, plot_title="Logo of Cut Global Unique Sequences (with gaps)")
    print(f"Plots for {alignment_path} with the gap distribution, the entropy distribution and the logos (with and without gaps) were saved.\n")

    # from previously aligned knobs, isolate knob-domain and compare again
    process_alignment_by_conserved_c_positions(
        aln_path=Path(save_dir_variable_data / "cut_DSATYY_WGXG_unique_vs_structure_picobodies.aln"),
        output_dir=Path(save_dir_variable_data),
        output_suffix="unique") # add "cut_knobs_" and ".fasta"

    # filter sequences (exact + near diff)
    evaluate_dedup_thresholds((save_dir_variable_data / "cut_knobs_unique.fasta"), diff_range=range(0, 5))
    dedup_report = deduplicate_and_filter_fasta(
        input_fasta=save_dir_variable_data / "cut_knobs_unique.fasta",
        output_fasta=save_dir_variable_data / "cut_knobs_unique_deduplicated.fasta",
        max_aa_difference=1, # decision-based
        negative_binder_ids=set(negative_binder_ids.keys()),
        debug=DEBUG)

    alignment_path = run_clustalw_alignment_from_fasta(
        fasta_files=[
            save_dir_variable_data / "knobs_internship_gapfree.fasta",
            save_dir_variable_data / "cut_knobs_unique_deduplicated.fasta"],
        clustalw_path=clustalw_path,
        output_dir=save_dir_variable_data,
        output_prefix="cut_deduplicated_knobs_unique_vs_structure_picobodies")

    plot_gap_distribution(alignment_path, save_dir_plots)
    plot_entropy_distribution(alignment_path, save_dir_plots)
    plot_sequence_logo(alignment_path, save_dir_plots, include_gaps=False, plot_title="Logo with Isolated Knobs of Global Unique Sequences")
    plot_sequence_logo(alignment_path, save_dir_plots, include_gaps=True, plot_title="Logo with Isolated Knobs of Global Unique Sequences (with gaps)")
    print(
        f"Plots for {alignment_path} with the gap distribution, the entropy distribution and the logos (with and without gaps) were saved.\n")

    # cysteine clustering with only knobs
    print("Cysteine clustering with only knobs:")
    clusters, _, _, _, _ = cysteine_clustering(aln_path=Path(save_dir_variable_data /"cut_deduplicated_knobs_unique_vs_structure_picobodies.aln"),
                        output_dir=Path(save_dir_plots),
                        output_prefix="cut_deduplicated_knobs_unique",
                        n_ignore = 32,
                        cluster_range = range(2,10),
                        debug=DEBUG)
    plot_cluster_logos(clusters, save_dir_plots, output_prefix="cut_deduplicated_knobs_unique", include_gaps = True)
    plot_cluster_logos(clusters, save_dir_plots, output_prefix="cut_deduplicated_knobs_unique", include_gaps = True, highlight_aa="C")

    # cystein clustering with knobs + experimental structures
    print("\nCysteine clustering with knobs + experimental structures:")
    clusters, Z, ids, near_knobs_to_knobs, knob_ids = cysteine_clustering(
        aln_path=Path(save_dir_variable_data / "cut_deduplicated_knobs_unique_vs_structure_picobodies.aln"),
        output_dir=Path(save_dir_plots),
        output_prefix=f"cut_deduplicated_knobs_unique_vs_structure_picobodies",
        n_ignore=0,
        cluster_range=range(2, 20),
        highlight_mode="first_n",
        highlight_n=32,
        negative_binder_ids=negative_binder_ids,
        near_knob_similarity_factor=0.5,
        debug=DEBUG)
    save_clusters(clusters, save_dir_variable_data / "clusters.pkl")
    export_clusters_to_txt(clusters, save_dir_variable_data / "clusters_full_table.txt", truncate_seq=False)

    plot_cluster_logos(clusters, save_dir_plots, output_prefix="cut_deduplicated_knobs_unique_vs_structure_picobodies", include_gaps=True)
    plot_cluster_logos(clusters, save_dir_plots, output_prefix="cut_deduplicated_knobs_unique_vs_structure_picobodies", include_gaps=True,
                       highlight_aa="C")
    cluster_summary_to_latex(
        clusters=clusters,
        output_tex_path=(save_dir_variable_data / "cluster_summary_cysteine_topology_cut_deduplicated_knobs_unique_vs_structure_picobodies.tex"))
    plot_cysteine_position_heatmap(
        clusters=clusters,
        save_path=(save_dir_plots / "cysteine_position_heatmap_cut_deduplicated_knobs_unique_vs_structure_picobodies.png"))
    plot_cysteine_spacing_violin(
        clusters=clusters,
        save_path=save_dir_plots / "cysteine_spacing_violin_cut_deduplicated_knobs_unique_vs_structure_picobodies.png")

    identity_results = compute_sequence_identity_matrix(
        alignment_path=save_dir_variable_data / "cut_deduplicated_knobs_unique_vs_structure_picobodies.aln",
        near_knobs_to_knobs=near_knobs_to_knobs,
        debug=DEBUG)

    export_full_knob_excel(
        near_knobs_to_knobs=near_knobs_to_knobs,
        clusters=clusters,
        aln_path_knobs=aln_path_knobs_internship,
        output_excel_path=Path(save_dir_variable_data / "clustering_summary.xlsx"),
        negative_binder_ids=negative_binder_ids,
        knob_ids=knob_ids,
        identity_results=identity_results)

# ---------------structure prediction--------------------------
if structure_prediction_prep_bool:
    print("\n--------------------Structure Prediction Prep--------------------")
    print("Structure prediction performed on cluster/server, not local. So only Preparation and Analysis here.")
    clusters = load_clusters(save_dir_variable_data / "clusters.pkl")

    write_clustered_fasta(clusters=clusters, save_path=(save_dir_esm / "esm_input.fasta"))
    write_clustered_yaml(clusters=clusters, save_dir=Path(save_dir_boltz / "Input_Boltz"))

    write_clustered_fasta(clusters=clusters, save_path=(save_dir_esm / "esm_input.fasta"), ligand_sequence=seq_mclover3)
    write_clustered_yaml(clusters=clusters, save_dir=Path(save_dir_boltz / "Input_Boltz"), ligand_sequence=seq_mclover3)

    disulfides_structures = collect_disulfides_from_folder(pdb_folder_knobs_internship, cutoff=2.1, debug=DEBUG)
    disulfides_mapped_knobs = map_disulfides_to_knobs(
        pdb_folder_knobs_internship,
        (save_dir_variable_data / "knobs_internship_gapfree.fasta"),
        disulfides_structures,
        debug=DEBUG)

    # not finished yet, but also not needed
    """
    validated_constraints = generate_valid_disulfide_constraints(
        near_knobs_to_knobs=near_knobs_to_knobs,
        disulfides_mapped_knobs=disulfides_mapped_knobs,
        clusters=clusters,
        alignment_path=save_dir_variable_data / "cut_deduplicated_knobs_unique_vs_structure_picobodies.aln",
        debug=DEBUG)
    """


if structure_prediction_analysis_bool:
    print("\n--------------------Structure Prediction Analysis--------------------")
    neg_ids_for_comparison = get_negative_binder_ids_from_vx_name(
        save_dir_variable_data / "clustering_summary.xlsx",
        ["v5", "v8", "v9", "v11"])
    tested_id_v1 = get_negative_binder_ids_from_vx_name(
        save_dir_variable_data / "clustering_summary.xlsx",
        "v1")

    pure_comparison_ids, pure_tested_ids = build_comparison_and_tested_id_sets(
        comparison_ids=list(neg_ids_for_comparison.values()),
        tested_ids=list(tested_id_v1.values()),
        additional_comparison_ids=[],
        additional_tested_ids=["seq_1", "seq_12", "seq_78",])

    MODEL_CONFIGS = {
        "esm": {
            "runs": [
                ("esmFold_simple_all", "without ligand"),
                ("esmFold_with_ligand_all", "with ligand"),
            ],
            "collector": collect_plddt_stats,
            "collector_chain": collect_esm_chain_models,
            "base_dir": save_dir_esm,
            "cluster_builder": lambda stats: build_model_to_cluster_from_fasta(
                save_dir_esm / "esm_input.fasta"
            ),
            "suffix_clean": lambda x: x,
            "label": "ESMFold",
        },
        "boltz": {
            "runs": [
                ("boltz_simple_all", "without ligand"),
                ("boltz_with_ligand_all", "with ligand"),
            ],
            "collector": lambda d: collect_boltz_best_models(d / "predictions"),
            "collector_chain": lambda d: collect_boltz_chain_models(d / "predictions"),
            "base_dir": save_dir_boltz,
            "cluster_builder": lambda stats: {
                normalize_model_id(mid): int(
                    re.search(r"cluster_?(\d+)", mid).group(1)
                ) for mid in stats
            },
            "suffix_clean": lambda x: x.replace("boltz_results_", ""),
            "label": "Boltz-2",
        },
        "af3": {
            "runs": [
                ("af3_simple_near", "without ligand"),
                ("af3_with_ligand_near", "with ligand"),
            ],
            "collector": collect_af_best_models,
            "collector_chain": collect_af_chain_models,
            "base_dir": save_dir_af,
            "cluster_builder": lambda stats: build_af_model_to_cluster(stats, clusters),
            "suffix_clean": lambda x: x,
            "label": "AlphaFold 3",
        },
    }

    clusters = load_clusters(save_dir_variable_data / "clusters.pkl")
    display_index = build_global_display_order(clusters)

    for model_name, cfg in MODEL_CONFIGS.items():

        subset_without = None
        subset_chain = None

        for run_dir, ligand_state in cfg["runs"]:
            print(f"------{cfg['label']} ({ligand_state})------")

            full_path = cfg["base_dir"] / run_dir

            # --- normal ---
            plddt_stats = cfg["collector"](full_path)
            global_max_len = get_max_residue_length(plddt_stats)
            model_to_cluster = cfg["cluster_builder"](plddt_stats)

            if ligand_state == "without ligand":
                subset_without = filter_plddt_stats_by_ids(
                    plddt_stats,
                    pure_tested_ids +
                    pure_comparison_ids)

            suffix = cfg["suffix_clean"](run_dir)

            plot_plddt_landscape(
                plddt_stats,
                model_to_cluster,
                display_index=display_index,
                save_path=(save_dir_plots / f"plddt_landscape_{suffix}.png"),
                max_residue_len=global_max_len,
                model_name=f"{cfg['label']} ({ligand_state})"
            )

            # CHAIN-ONLY (without mClover)
            if "with_ligand" in run_dir:
                print(f"------{cfg['label']} ({ligand_state}) CHAIN A------")

                chain_stats = cfg["collector_chain"](full_path)
                subset_chain = filter_plddt_stats_by_ids(
                    chain_stats,
                    pure_tested_ids +
                    pure_comparison_ids)

                global_max_len_chain = get_max_residue_length(chain_stats)
                full_mapping = cfg["cluster_builder"](plddt_stats)

                model_to_cluster_chain = {
                    normalize_model_id(mid): full_mapping[normalize_model_id(mid)]
                    for mid in chain_stats
                    if normalize_model_id(mid) in full_mapping
                }

                plot_plddt_landscape(
                    chain_stats,
                    model_to_cluster_chain,
                    display_index=display_index,
                    save_path=(save_dir_plots / f"plddt_landscape_{suffix}_chainA.png"),
                    max_residue_len=None,
                    model_name=f"{cfg['label']} ({ligand_state}, chain A)")

                if subset_without is not None and subset_chain is not None:
                    plot_plddt_landscape_groups(
                        stats_without=subset_without,
                        stats_chainA=subset_chain,
                        tested_ids=pure_tested_ids,
                        comparison_ids=pure_comparison_ids,
                        save_path=(save_dir_plots / f"{model_name}_tested_vs_comparison.png"),
                        model_name=cfg["label"],)

    CONDITIONS = [
        ("without ligand", {
            "ESMFold": ("esm", "esmFold_simple_all"),
            "Boltz-2": ("boltz", "boltz_simple_all"),
            "AF3": ("af3", "af3_simple_near"),
        }),
        ("with ligand", {
            "ESMFold": ("esm", "esmFold_with_ligand_all"),
            "Boltz-2": ("boltz", "boltz_with_ligand_all"),
            "AF3": ("af3", "af3_with_ligand_near"),
        }),
    ]

    for ligand_state, mapping in CONDITIONS:

        print(f"\n------Mean comparison ({ligand_state})------")

        stats_dict = {}
        stats_chain_dict = {}

        for label, (model_key, run_dir) in mapping.items():

            cfg = MODEL_CONFIGS[model_key]
            full_path = cfg["base_dir"] / run_dir

            stats = cfg["collector"](full_path)
            stats_dict[label] = stats

            # --- nur wenn ligand ---
            if "with_ligand" in run_dir:
                stats_chain = cfg["collector_chain"](full_path)
                stats_chain_dict[label] = stats_chain

        # normal plot
        plot_mean_plddt_multi_models(
            stats_dict,
            labels=list(stats_dict.keys()),
            save_path=(save_dir_plots /
                       f"plddt_mean_comparison_{ligand_state.replace(' ', '_')}.png"),
            display_index=display_index,
            max_structure_index=max(display_index.values()),
            title=ligand_state,
        )

        # chain-only plot
        if stats_chain_dict:
            plot_mean_plddt_multi_models(
                stats_chain_dict,
                labels=list(stats_chain_dict.keys()),
                save_path=(save_dir_plots /
                           f"plddt_mean_comparison_{ligand_state.replace(' ', '_')}_chainA.png"),
                display_index=display_index,
                max_structure_index=max(display_index.values()),
                title=f"{ligand_state} (chain A)",
            )


if MD_prep_bool:
    print("\n--------------------MD Prep--------------------")
    archive_and_clean_md_root(md_root, archive_root)

    MODEL_DATABASE = {
        ("AF3", True): collect_af_best_models(save_dir_af / "af3_with_ligand_near"),
        ("AF3", False): collect_af_best_models(save_dir_af / "af3_simple_near"),
        ("Boltz", True): collect_boltz_best_models(save_dir_boltz / "boltz_with_ligand_all" / "predictions"),
        ("Boltz", False): collect_boltz_best_models(save_dir_boltz / "boltz_simple_all" / "predictions")}

    template_dir = Path(r"C:\Users\ge63laz\PycharmProjects\Masterthesis_Picobodies\pc_cluster_scripts/MolecularDynamics")

    for config in selected_seqs_MD:
        prepare_md_input_for_seq(
            seq_id=config["seq_id"],
            requested_model=config["model"],
            ligand=config["ligand"],
            model_database=MODEL_DATABASE,
            md_root=md_root,)

    distribute_md_scripts(
        md_root=md_root,
        template_dir=template_dir)



if MD_analysis_bool:

    print("\n--------------------MD Analysis--------------------")

    md_analyses = {
        "default": {
            "outputs": [
                "rmsd.csv",
                "rmsd.png",
                "rmsf.csv",
                "rmsf.png",],
            "analyses": [
                create_rmsd_analysis,
                create_rmsf_analysis]},

        "with_ligand": {
            "outputs": [
                "binder_target_distance.csv",
                "binder_target_distance.png",],
            "analyses": [
                partial(
                    create_binder_target_distance_analysis,
                    binder_chain="B",
                    target_chain="A")]}}

    # turn off warnings after double check
    warnings.filterwarnings(
        "ignore",
        message="DCDReader currently makes independent timesteps*",
        category=DeprecationWarning,)
    warnings.filterwarnings(
        "ignore",
        message="PDB file is missing resid information*",
        category=UserWarning,)

    analysis_runs = get_md_analysis_runs(
        archive_root=archive_root,
        md_analyses=md_analyses,
        force_reanalysis=True,)

    for (analysis_dir, topology_file, trajectory_file, analyses, run_name) in analysis_runs:
        create_conditions_file(
            run_dir=topology_file.parent,
            output_dir=analysis_dir,
            run_name=run_name,
        )

        run_md_analysis(
            topology_file=topology_file,
            trajectory_file=trajectory_file,
            output_dir=analysis_dir,
            analyses=analyses,
            run_name=run_name)

        print(f"[MD Analysis] Finished {run_name}")






print("\n--------------------Runtime--------------------")
end_time_main = time.time()
elapsed_main = end_time_main - start_time_main
print(f"\033[92m✔ Script finished with a total runtime of {elapsed_main:.1f} seconds\033[0m")






