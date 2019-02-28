# Optmisiation notes

## Background

In order to speed up the creation of the kmer table, I use multithreading to pass off sequences to different processors. However, since there's processing overhead with spawning a new thread there is probably a lower limit at which this is worthwhile. For testing purposes, there is a blockSize parameter in the *computeKmerProfile.py* script which I can tweak to profile the run times.

### Step 1 - Defining the dataset

I can't profile with the fasta file found at *tests/kmer.input.chomp1500.fna* and the run times are too short. Since the file only contains 1,896 sequences and they are all around 1.5 kbp length, the total runtime for the script is in the seconds range.

Instead pull a much larger data set of long reads, so that there is real run time.

```bash
seqmagick convert --sample 100000 ../e1e2.prok.fna mock_reads.fna
```

### Step 2 - Profiling the change in runtime as blockSize increases

Ignoring parameters that have no effect on blockSize manipulation (normalisation, coverage table). I am including the reverse complement process because this is really something that should be happening on every run, and is a single-thread function that will change with kmer size or number of input sequences, but not thread count.

```bash
for i in 1 2 4 8 16 32 64 128 256 512;
do
    time python computeKmerProfile_blocksize.py -k 4 -t 10 -n none mock_reads.fna
done
```

### Step 3 - Controls

At blocksize = 1, the amount of overhead needed to generate the sequence blocks is actually a performance hit. Use a slightly different version of the code for profiling run times at blocksize = 1.

```bash
time python computeKmerProfile.py -k 4 -t 10 -n none mock_reads.fna
```

### Step 4 - Compare

|Block size|Run time (s)|
|:---|:---:|
|1, fixed|206.141|
|1, dynamic|257.888|
|2, dynamic|241.145|
|4, dynamic|240.061|
|8, dynamic|241.698|
|16, dynamic|244.368|
|32, dynamic|241.251|
|64, dynamic|241.674|
|128, dynamic|241.469|
|256, dynamic|241.76|
|512, dynamic|241.296|

Pretty simple, block size of 1 without the dictionary reformatting overhead is a better choice for now.
