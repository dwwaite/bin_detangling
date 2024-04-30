# Bin detangling workflow

This is a work-in-progress collection of scripts for the refinement of metagenome-assembled genomes (MAGs) using a combination of emergent self-organising maps (ESOM) and machine-learning algorithms. This is not intended to work as a single all-in-one package, as there are too many stages to refinement that require user intervention and judgement calls. Instead, this repository is a collection of scripts that are called together to walk through a series of pre-computed bins and coverage files to achieve the outcome.

The documentation will improve as scripts are finalised and the workflow is validated.

## Introduction

The idea behind this analysis is that you have obtained a metagenomic assembly, and are trying to identify clusters of contigs which belong to the same organism, or group of closely related organisms. There are a number of excellent automated pipelines for the recovery of these organism bins (or MAGs; **M**etagenomic-**A**ssembled **G**enomes), and the outputs of these pipelines are a good place to begin this workflow.

Three binning tools I frequently use are:

* [MetaBAT](https://bitbucket.org/berkeleylab/metabat) ([Kang et al., 2015](https://peerj.com/articles/1165/))
* [MaxBin](https://sourceforge.net/projects/maxbin/) ([Wu et al., 2014](https://microbiomejournal.biomedcentral.com/articles/10.1186/2049-2618-2-26))
* [CONCOCT](https://github.com/BinPro/CONCOCT) ([Alneberg et al., 2014](https://www.ncbi.nlm.nih.gov/pubmed/25218180))
* [GroopM](https://github.com/Ecogenomics/GroopM) ([Imelfort et al., 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4183954/))

This workflow employs some biology-agnostic clustering techniques to evaluate the evidence for bin formation and membership and is not intended as a replacement for any of these tools. These scripts are for refining problematic bins and ensuring the quality of bins obtained from these software suites.

*Note - technically you **could** perform binning with this workflow, but I wouldn't recommend it.*

## Quick-fire use

Using the initial bins produced in the Genomics Aotearoa [Metagenomics Summer School](https://genomicsaotearoa.github.io/metagenomics_summer_school/).

Start by mapping the data to produce the coverage table required for the binning refinement. There are two binned data sets here, the raw bins (`bin_*.fna`), and those that have been through [DAS_Tool](https://github.com/cmks/DAS_Tool) refinement.

### Raw bins

```bash
bowtie2-build data/spades_assembly.m1000.fna data/spades_assembly.m1000

for i in {1..4};
do
    bowtie2 --sensitive-local --threads 4 -x data/spades_assembly.m1000 -1 data/sample${i}_R1.fastq.gz -2 data/sample${i}_R2.fastq.gz > sample${i}.sam
    samtools view -bS sample${i}.sam | samtools sort -o sample${i}.bam
    samtools depth -a sample${i}.bam > sample${i}.depth.txt
done

# Create per-contig summary of the depths
python bin/compute_depth_profile.py -o results/depth.parquet sample{1..4}.depth.txt
```

Fortunately, the refined bins have a different extension to the raw versions, so they are easy to sort by wildcard.

```bash
# Raw bins
python bin/compute_kmer_profile.py -k 4 -o results/raw_bins.parquet -f results/raw_bins.fna -t 4 data/bin_*.fna

python bin/project_ordination.py -n yeojohnson -w 0.5 --store_features results/raw_bins.matrix.tsv -k results/raw_bins.parquet -c results/depth.parquet -o results/raw_bins.tsne.parquet
```
