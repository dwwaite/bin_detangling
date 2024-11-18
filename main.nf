/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Bin detangling pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
nextflow.enable.dsl=2


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    User inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Mandatory inputs
params.assembly      = ""
params.fastq_pattern = ""
params.bin_pattern   = ""

// Optional inputs
params.fragment_size      = 10000
params.kmer_size          = 4
params.kmer_normalise     = "yeojohnson"
params.coverage_weighting = 0.5
params.core_threshold     = 0.8
params.recruit_threshold  = 0.8
params.model_score        = "Score_ROC"

params.help = false
if (params.help) {
    help_message()
    exit 0
}

// Output file names
OUTPUT_FRAGMENTS_FNA  = "fragments.fna"
OUTPUT_FEATURE_MATRIX = "feature_matrix.tsv"
OUTPUT_PROJECTION     = "projection_core.parquet"
OUTPUT_RECRUITMENT    = "recruitment_table.tsv"

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Help
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

def help_message() {
    log.info """
        Version:
            ${workflow.manifest.version}

        Usage:
            nextflow run main.nf --assembly asm.fna --fastq_pattern "data/*.fq.gz" --bin_pattern "bins/*.fna"

        Mandatory arguments:
            --assembly           Path to the original assembly file from which binning was performed
            --fastq_pattern      Wildcard pattern to capture paired-end sequencing files used to produce bin coverage profiles
            --bin_pattern        Wildcard pattern to all bins to be refined, stored as one fasta file per bin

       Optional arguments:
            --n_cpus             The number of CPUs to use for high-demand tasks (Default: ${params.n_cpus})
            --fragment_size      The fragment size to use when slicing contigs (Default: ${params.fragment_size})
            --kmer_size          The k-mer size when producing frequency profiles (Default: ${params.kmer_size})
            --kmer_normalise     The method to use when normalising k-mer profiles (Default: ${params.kmer_normalise})
            --coverage_weighting The weighting to assign to coverage profiles when performing ordination
                                 (Default: ${params.coverage_weighting}, which gives half weighting divide amongst all depth profiles,
                                 and half coverage divided amongst all k-mer frequencies)
            --core_threshold     The MCC cutoff for defining the core of each bin (Default: ${params.core_threshold})
            --recruit_threshold  The minimum proportion of assignments to a bin for a recruitment even to be accepted (Default: ${params.recruit_threshold})
            --model_score        The metric to use when determinig the 'best' model for recruitment (Default: ${params.model_score})
            --output             The output folder path for all files and logs to be stored (Default: ${params.output})

            --help               This usage statement.
    """
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Processes - Read mapping bowtie2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

process bwt2_index_assembly {

    conda "${projectDir}/envs/mapping.yaml"
    label "low_cpu"

    input:
    file assembly_fna

    output:
    val  "${index_prefix}",   emit: index_name
    path "${index_prefix}.*", emit: index_path

    script:
    index_prefix = "asm_index"
    """
    bowtie2-build ${assembly_fna} ${index_prefix}
    """
}

process bwt2_interleave {

    conda "${projectDir}/envs/utilities.yaml"
    label "low_cpu"

    input:
    tuple val(pair_id), path(reads)

    output:
    path fq_file

    script:
    fq_file = "${pair_id}.fq.gz"
    """
    seqtk mergepe ${reads[0]} ${reads[1]} | gzip -c > ${fq_file}
    """
}

process bwt2_map_reads {

    conda "${projectDir}/envs/mapping.yaml"

    input:
    path index_files, stageAs: "bt2_index/*"
    each path(fq_file)

    output:
    path sam_file

    script:
    def idx = index_files[0].getBaseName(2)
    sam_file = "${fq_file.getSimpleName()}.sam"
    """
    bowtie2 --sensitive-local --threads ${task.cpus} -x ${idx} --interleaved ${fq_file} > ${sam_file}
    """
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Processes - Read mapping with minimap2

    minimap2=2.28
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*
process minimap_index_assembly {

    conda "${projectDir}/envs/mapping.yaml"
    label "low_cpu"

    input:
    file assembly_fna

    output:
    path assembly_mmi

    script:
    assembly_mmi = "asm_index.mmi"
    """
    minimap2 -d ${assembly_mmi} ${assembly_fna}
    """
}

process minimap_map_reads {

    conda "${projectDir}/envs/mapping.yaml"

    input:
    path index_file
    each tuple val(pair_id), path(reads)

    output:
    path sam_file

    script:
    sam_file = "${pair_id}.sam"
    """
    minimap2 -t ${task.cpus} -ax sr ${index_file} ${reads[0]} ${reads[1]} > ${sam_file}
    """
}
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Processes - Mapping summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

process mapping_compute_depth {

    conda "${projectDir}/envs/mapping.yaml"

    input:
    path sam_file

    output:
    path depth_file

    script:
    depth_file = "${sam_file.getSimpleName()}.txt"
    """
    samtools view -@ ${task.cpus} -bS ${sam_file} | samtools sort -o sample.bam
    samtools depth -a sample.bam > ${depth_file}
    """
}

process mapping_consolidate_profiles {

    conda "${projectDir}/envs/python3.yaml"

    input:
    path depth_files

    output:
    path depth_table

    script:
    depth_table = "depth_profile.parquet"
    """
    python ${projectDir}/bin/compute_depth_profile.py -o ${depth_table} ${depth_files}
    """
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Processes - Bin profiling and core identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

process bin_profile_kmers {

    conda "${projectDir}/envs/python3.yaml"
    publishDir params.output, mode: "copy", pattern: "${OUTPUT_FRAGMENTS_FNA}"

    input:
    path bin_files

    output:
    path bin_profile,    emit: profile
    path kmer_fragments, emit: fasta

    script:
    kmer_fragments = "${OUTPUT_FRAGMENTS_FNA}"
    bin_profile    = "kmer_tally.parquet"
    """
    python ${projectDir}/bin/compute_kmer_profile.py \
        --threads ${task.cpus} \
        --kmer ${params.kmer_size} \
        --window ${params.fragment_size} \
        --output ${bin_profile} \
        --fasta ${kmer_fragments} \
        ${bin_files}
    """
}

process bin_normalise_features {

    conda "${projectDir}/envs/python3.yaml"
    publishDir params.output, mode: "copy", pattern: "${OUTPUT_FEATURE_MATRIX}"

    input:
    path depth_profile
    path bin_profile

    output:
    path bin_matrix,    emit: user_matrix
    path feature_table, emit: feature_table

    script:
    bin_matrix    = "${OUTPUT_FEATURE_MATRIX}"
    feature_table = "raw_bins.parquet"
    """
    python ${projectDir}/bin/project_ordination.py \
        --normalise ${params.kmer_normalise} \
        --weighting ${params.coverage_weighting} \
        --kmer ${bin_profile} \
        --coverage ${depth_profile} \
        --store_features ${bin_matrix} \
        --output ${feature_table}
    """
}

process bin_identify_cores {

    conda "${projectDir}/envs/python3.yaml"
    publishDir params.output, mode: "copy"

    input:
    path projection

    output:
    path projection_core, emit: core
    path bin_plots,       emit: plots

    script:
    projection_core = "${OUTPUT_PROJECTION}"
    bin_plots       = "*.html"
    """
    python ${projectDir}/bin/identify_bin_cores.py --plot_traces \
        --threshold ${params.core_threshold} \
        -i ${projection} \
        -o ${projection_core}
    """
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Processes - Model training, selection, and contig recruitment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

process models_train {

    conda "${projectDir}/envs/python3.yaml"
    publishDir params.output, mode: "copy"

    input:
    path projection_core

    output:
    path model_files
    path summary_files

    script:
    model_files   = "*.pkl"
    summary_files = "*.tsv"
    """
    python ${projectDir}/bin/recruit_by_ml.py train \
        --neural_network --random_forest --svm_linear --svm_radial \
        -i ${projection_core} \
        -o ./
    """
}

process models_select {

    conda "${projectDir}/envs/python3.yaml"
    label "low_cpu"

    input:
    tuple val(model_type), path(summary_file), path(model_files)

    output:
    tuple val(model_type), path(top_pick)

    shell:
    top_pick = "top_pick.*.pkl"
    """
    #!/usr/bin/env python3

    import shutil
    import polars as pl

    model, _ = (
        pl
        .scan_csv('!{summary_file}', separator='\t')
        .select('Model', '!{params.model_score}')
        .sort('!{params.model_score}', descending=True)
        .collect()
        .row(0)
    )

    shutil.copyfile(f"{model}.pkl", f"top_pick.{model}.pkl")
    """
}

process models_recruit {

    conda "${projectDir}/envs/python3.yaml"
    publishDir params.output, mode: "copy"

    input:
    path projection_core
    path selected_models

    output:
    path recruitment_table

    script:
    recruitment_table = "${OUTPUT_RECRUITMENT}"
    """
    python ${projectDir}/bin/recruit_by_ml.py recruit \
        --models ${selected_models} \
        --threshold ${params.recruit_threshold} \
        -i ${projection_core} \
        -o ${recruitment_table}
    """
}

process export_refined_bins {

    conda "${projectDir}/envs/python3.yaml"
    label "low_cpu"
    publishDir params.output, mode: "copy"

    input:
    path assembly_fna
    path recruitment_table

    output:
    path bin_files

    shell:
    bin_files = "refined.*.fna"
    """
    #!/usr/bin/env python3

    import polars as pl
    from Bio import SeqIO

    assembly_dict = SeqIO.to_dict(SeqIO.parse('!{assembly_fna}', 'fasta'))
    bin_groups = (
        pl
        .scan_csv('!{recruitment_table}', separator='\t')
        .filter(pl.col('Bin').ne('unassigned'))
        .collect()
        .group_by(['Bin'])
    )

    for (target_bin,), df in bin_groups:
        with open(f"refined.{target_bin}", 'w') as fna_writer:
            for contig in df['Contig']:
                _ = SeqIO.write(assembly_dict[contig], fna_writer, 'fasta')
    """
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

workflow {

    // Index the reference assembly and produce the depth tables
    Channel.fromPath(params.assembly)
        | bwt2_index_assembly

    Channel.fromFilePairs(params.fastq_pattern)
        | bwt2_interleave

    // KEEP THIS LINE FOR WHEN MINIMAP2 IS ADDED!
    //mapping_ch = index_assembly.out.index_name.combine(fq_pairs).view()

    // bowtie2 mapping workflow
    bwt2_map_reads(bwt2_index_assembly.out.index_path, bwt2_interleave.out)
        | mapping_compute_depth
        | collect
        | mapping_consolidate_profiles

    // Prepare the k-mer profile and project the TSNE coordinates
    Channel.fromPath(params.bin_pattern)
        | collect
        | bin_profile_kmers

    // Identify the core and train models.
    // Using some explicit labelling of channels to clarify the workflow
    depth_ch   = mapping_consolidate_profiles.out
    profile_ch = bin_profile_kmers.out.profile
    bin_normalise_features(depth_ch, profile_ch)

    bin_identify_cores(bin_normalise_features.out.feature_table)
    models_train(bin_identify_cores.out.core)

    models_train.out
        | concat()
        | flatten()
        // Map each output into key/value tuples, using the model name as key
        | map { v ->
                def key = v.getSimpleName().toString().replaceAll(/_(\d+)/, '')
                return tuple(key, v)
            }
        | groupTuple()
        // Reorganise the grouped tuple, from tuple(key, [files]) to tuple(key, summary, [files])
        | map { v ->
                def key = v[0]
                def summary = v[1].find{ it =~ /.*.tsv/ }
                def models = v[1].grep(~/.*.pkl/)
                return tuple(key, summary, models)
            }
        | models_select

    models_select.out
        // Report the selection to the user
        | view{ v ->
                def model = v[0]
                def top_pick = v[1].getBaseName().toString().replace('top_pick.', '')
                "Model: $model, top pick was $top_pick"
            }
        // Strip out the model from the results and combine to a new channel
        | map { v -> v[1] }
        | collect
        | set { models_ch }

    // Perform the recruitment step, then produce a refined set of bins.
    // No filtering to the threshold is done here, this is a user-specific task.
    models_recruit(bin_identify_cores.out.core, models_ch)
    export_refined_bins(
        Channel.fromPath(params.assembly),
        models_recruit.out
    )
}
