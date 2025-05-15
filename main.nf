/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Bin detangling pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
nextflow.enable.dsl     = 2
nextflow.preview.output = true

include { wf_read_mapping      } from "${projectDir}/modules/read_mapping.nf"
include { wf_kmer_profiling    } from "${projectDir}/modules/kmer_profiling.nf"
include { wf_model_selection   } from "${projectDir}/modules/ml_models.nf"
include { wf_model_recruitment } from "${projectDir}/modules/ml_models.nf"

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
    Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

workflow {
    // Prepare channels from input data.
    input_assembly          = Channel.fromPath(params.assembly)
    input_fastq_pattern     = Channel.fromFilePairs(params.fastq_pattern)
    input_bin_pattern       = Channel.fromPath(params.bin_pattern)

    param_kmer_size         = Channel.of(params.kmer_size)
    param_fragment_size     = Channel.of(params.fragment_size)
    param_coverage_weight   = Channel.of(params.coverage_weighting)
    param_normalise_kmers   = Channel.of(params.kmer_normalise)
    param_core_threshold    = Channel.of(params.core_threshold)
    param_model_score       = Channel.of(params.model_score)
    param_recruit_threshold = Channel.of(params.recruit_threshold)

    // Index the reference assembly and produce the depth tables
    wf_read_mapping(input_assembly, input_fastq_pattern)

    // Prepare the k-mer profile and project the TSNE coordinates
    wf_kmer_profiling(
        wf_read_mapping.out.mapping_profile,
        input_bin_pattern,
        param_kmer_size,
        param_fragment_size,
        param_coverage_weight,
        param_normalise_kmers,
        param_core_threshold
    )

    // Train ML models and recruit new contigs
    wf_model_selection(wf_kmer_profiling.out.bin_core, param_model_score)
    wf_model_recruitment(
        input_assembly,
        wf_kmer_profiling.out.bin_core,
        wf_model_selection.out.selected_models,
        param_recruit_threshold
    )

    publish:
    wf_kmer_profiling.out.bin_core             >> "features"
    wf_kmer_profiling.out.bin_matrix           >> "features"
    wf_kmer_profiling.out.bin_plots            >> "plots"
    wf_kmer_profiling.out.kmer_fragments       >> "features"
    wf_model_selection.out.all_models          >> "ml_models"
    wf_model_selection.out.summary_tables      >> "ml_models"
    wf_model_recruitment.out.recruitment_table >> "features"
    wf_model_recruitment.out.refined_bins      >> "bin_files"
}

output {
    bin_files { mode "move" }
    features  { mode "copy" }
    ml_models { mode "copy" }
    plots     { mode "move" }
}
