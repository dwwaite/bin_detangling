/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Processes - Model creation and selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

process models_train {

    label "env_python"

    input:
    path projection_core

    output:
    path model_files,   emit: model_files
    path summary_files, emit: summary_files

    script:
    model_files   = "*.pkl"
    summary_files = "*.tsv"
    """
    python ${projectDir}/bin/recruit_by_ml.py train \
        --neural_network --random_forest --svm_linear --svm_radial \
        --threads ${task.cpus} \
        -i ${projection_core} \
        -o ./
    """
}

process models_select {

    label "env_python"
    label "low_cpu"

    input:
    tuple val(model_type), path(summary_file), path(model_files), val(score_method)

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
        .select('Model', '!{score_method}')
        .sort('!{score_method}', descending=True)
        .collect()
        .row(0)
    )

    shutil.copyfile(f"{model}.pkl", f"top_pick.{model}.pkl")
    """
}

process models_recruit {

    label "env_python"
    label "low_cpu"

    input:
    path projection_core
    path selected_models
    val param_recruit_threshold

    output:
    path recruitment_table

    script:
    recruitment_table = "recruitment_table.tsv"
    """
    python ${projectDir}/bin/recruit_by_ml.py recruit \
        --models ${selected_models} \
        --threshold ${param_recruit_threshold} \
        -i ${projection_core} \
        -o ${recruitment_table}
    """
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Processes - Contig recruitment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

process export_refined_bins {

    label "env_python"
    label "low_cpu"

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
    Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

workflow wf_model_selection {
    take:
    bin_core
    param_model_score

    main:
    trained_models = bin_core
        | models_train
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

    selection_criteria = trained_models.combine(param_model_score)

    top_models = selection_criteria
        | models_select
        // Report the selection to the user
        | view{ v ->
              def model = v[0]
              def top_pick = v[1].getBaseName().toString().replace('top_pick.', '')
              "Model: $model, top pick was $top_pick"
          }
        // Strip out the model from the results and combine to a new channel
        | map { v -> v[1] }
        | collect

    emit:
    all_models      = models_train.out.model_files
    summary_tables  = models_train.out.summary_files
    selected_models = top_models
}

workflow wf_model_recruitment {
    take:
    base_assembly
    projection_core
    selected_models
    param_recruit_threshold

    main:
    // Perform the recruitment step, then produce a refined set of bins.
    // No filtering to the threshold is done here, this is a user-specific task.
    models_recruit(projection_core, selected_models, param_recruit_threshold)
    export_refined_bins(base_assembly, models_recruit.out)

    emit:
    recruitment_table = models_recruit.out
    refined_bins      = export_refined_bins.out
}
