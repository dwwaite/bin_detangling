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

Grab a few example genomes:

```bash
# Campylobacter subantarcticus LMG 24377
datasets download genome accession GCF_000816305.1 --include genome

# Campylobacter ureolyticus
datasets download genome accession GCF_013372225.1 --include genome

# Campylobacter sputorum bv. paraureolyticus LMG 11764
datasets download genome accession GCF_002220755.1 --include genome

# Campylobacter fetus
datasets download genome accession GCF_011600945.2 --include genome

# Campylobacter jejuni
datasets download genome accession GCF_000009085.1 --include genome
```

```bash
python bin/compute_kmer_profile.py -k 4 -o kmers.parquet -f fragments.fna -t 4 data/*.fna

#python project_tsne.py -w 0.5 -o binned_contigs.weighted.tsne binned_contigs.tsv



```
