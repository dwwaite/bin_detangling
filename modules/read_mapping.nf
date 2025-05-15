/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Processes - Read mapping bowtie2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

process bwt2_index_assembly {

    label "env_mapping"
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

    label "env_utilities"
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

    label "env_mapping"

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

process compute_depth {

    label "env_samtools"

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

process consolidate_profiles {

    label "env_python"

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
    Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

workflow wf_read_mapping {
    take:
    base_assembly
    fastq_pattern

    main:
    bwt2_index_assembly(base_assembly)
    bwt2_interleave(fastq_pattern)

    bwt2_map_reads(bwt2_index_assembly.out.index_path, bwt2_interleave.out)
        | compute_depth
        | collect
        | consolidate_profiles

    // KEEP THIS LINE FOR WHEN MINIMAP2 IS ADDED!
    //mapping_ch = index_assembly.out.index_name.combine(fq_pairs).view()

    emit:
    mapping_profile = consolidate_profiles.out
}
