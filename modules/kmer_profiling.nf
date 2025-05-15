/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Processes - Bin profiling and core identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

process profile_kmers {

    label "env_python"

    input:
    path bin_files
    val param_kmer_size
    val param_fragment_size

    output:
    path profile,   emit: profile
    path fragments, emit: fragments

    script:
    fragments = "fragments.fna"
    profile   = "kmer_tally.parquet"
    """
    python ${projectDir}/bin/compute_kmer_profile.py \
        --threads ${task.cpus} \
        --kmer ${param_kmer_size} \
        --window ${param_fragment_size} \
        --output ${profile} \
        --fasta ${fragments} \
        ${bin_files}
    """
}

process normalise_features {

    label "env_python"

    input:
    path depth_profile
    path bin_profile
    val param_coverage_weight
    val param_normalise_kmers

    output:
    path feature_matrix, emit: feature_matrix
    path feature_table,  emit: feature_table

    script:
    feature_matrix = "feature_matrix.tsv"
    feature_table  = "raw_bins.parquet"
    """
    python ${projectDir}/bin/project_ordination.py \
        --weighting ${param_coverage_weight} \
        --normalise ${param_normalise_kmers} \
        --kmer ${bin_profile} \
        --coverage ${depth_profile} \
        --store_features ${feature_matrix} \
        --output ${feature_table}
    """
}

process identify_cores {

    label "env_python"

    input:
    path projection
    val param_core_threshold

    output:
    path projection_core, emit: projection_core
    path bin_plots,       emit: bin_plots

    script:
    projection_core = "projection_core.parquet"
    bin_plots       = "*.html"
    """
    python ${projectDir}/bin/identify_bin_cores.py --plot_traces \
        --threshold ${param_core_threshold} \
        -i ${projection} \
        -o ${projection_core}
    """
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

workflow wf_kmer_profiling {
    take:
    mapping_profile
    bin_pattern
    param_kmer_size
    param_fragment_size
    param_coverage_weight
    param_normalise_kmers
    param_core_threshold

    main:
    gathered_bins = bin_pattern | collect
    
    profile_kmers(
        gathered_bins,
        param_kmer_size,
        param_fragment_size
    )

    normalise_features(
        mapping_profile,
        profile_kmers.out.profile,
        param_coverage_weight,
        param_normalise_kmers
    )

    identify_cores(
        normalise_features.out.feature_table,
        param_core_threshold
    )

    emit:
    kmer_profile   = profile_kmers.out.profile
    kmer_fragments = profile_kmers.out.fragments
    bin_matrix     = normalise_features.out.feature_matrix
    bin_features   = normalise_features.out.feature_table
    bin_core       = identify_cores.out.projection_core
    bin_plots      = identify_cores.out.bin_plots
}
