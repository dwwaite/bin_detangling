import argparse
from multiprocessing import Pool
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Callable, Generator, List

import polars as pl

from Bio import SeqIO
from Bio.Seq import Seq

@dataclass
class Fragment:
    source: str
    contig: str
    name: str
    seq: Seq
    kmer_size: int

    def create_line(self) -> str:
        return f">{self.name}\n{str(self.seq)}\n"

def main():

    # Set up the options
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-w', '--window', type=int, default=10_000,
        help='Window size for fragmenting input contigs (Default: 10k)'
    )
    parser.add_argument(
        '-k', '--kmer', metavar='KMER_SIZE', dest='kmer_size', type=int, default=4,
        help='Kmer size for profiling (Default: 4)'
    )
    parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads to use (Default: 1)')
    parser.add_argument('-o', '--output', help='Output file for k-mer profile results')
    parser.add_argument('-f', '--fasta', help='Output file for fragmented fasta sequences')
    parser.add_argument('input_files', metavar='FASTA_FILES', nargs='+', help='Fasta files (one per MAG) to be profiled')

    options = parser.parse_args()

    # Create an input list from the file(s) provided and deploy across the available threads
    fragment_list = []
    for input_file in options.input_files:
        fragment_list += [fragment for fragment in slice_fasta_file(input_file, options.window, options.kmer_size)]

    write_fragmented_reads(fragment_list, options.fasta) 

    # Spawn and execute background processes if required, otherwise compute kmers in the main process
    if options.threads > 1:
        fragment_profiles = deploy_multithreaded_function(
            fragment_list,
            compute_kmer_profile,
            options.threads
        )

    else:
        fragment_profiles = [compute_kmer_profile(fragment) for fragment in fragment_list]

    # Gather the individual outputs into a single DataFrame and save into parquet format
    final_df = combine_data_frames(fragment_profiles)
    final_df.write_parquet(options.output)

#region Fasta and sequence handling

def slice_fasta_file(file_path: str, window_size: int, kmer_size: int) -> List[Fragment]:
    """ Return a list of Fragments derived from the fasta contigs sliced into
        non-overlapping sections and the size of the kmers that each fragment will
        be assessed using.

        If contigs are shorter than the window size, they processed as-is. If the terminal
        fragment in a contig is shorter than the window size, it is appended to the second
        to last fragment so as to avoid the generation of fragments too short to yield
        meaningful information.

        Arguments:
        file_path   -- input fasta file of sequences to be broken into fragments
        window_size -- the sliding window size for the windowed walk along each contig
        kmer_size   -- the k-mer size to use for computing k-mer frequency
    """

    records = []

    for record in SeqIO.parse(file_path, 'fasta'):

        # If the sequence is shorter than the window size, add it as a fragment.
        if len(record.seq) < window_size:
            records.append(
                Fragment(
                    source=file_path,
                    contig=record.id,
                    name=f"{record.id}__0",
                    seq=record.seq,
                    kmer_size=kmer_size
                )
            )

        # If the length if longer than the window size, break it into fragments
        else:
            fragment_list = [
                Fragment(
                    source=file_path,
                    contig=record.id,
                    name=f"{record.id}__{i}",
                    seq=record.seq[i:i+window_size],
                    kmer_size=kmer_size
                )
                for i in range(0, len(record.seq), window_size)
            ]

            # Assess the last entry, and if the last fragment is shorter than the window_size,
            # remove it and join to the second to last fragment.
            if len(fragment_list[-1].seq) < window_size:
                final_seq = fragment_list.pop(-1)
                fragment_list[-1].seq += final_seq.seq

            records.extend(fragment_list)

    return records

def sequence_to_kmers(sequence: Seq, kmer_size: int) -> Generator[str, None, None]:
    """ Create a Generator over the k-mers from the input sequence. K-mers are compared
        with their reverse complement and sorted alphabetically. The first in the sort
        is returned. K-mers with N character are skipped.

        Arguments:
        sequence  -- a nucleotide sequence represetentation
        kmer_size -- the k-mer size for creating the k-mer frequency profile
    """

    for i in range(0, len(sequence) - kmer_size + 1, 1):

        kmer = Seq(sequence[i:i+kmer_size])

        if not 'N' in kmer:
            k_reverse = kmer.reverse_complement()
            selected_kmer = sorted([kmer, k_reverse])[0]
            yield str(selected_kmer)

def compute_kmer_profile(fragment: Fragment) -> pl.DataFrame:
    """ For each Fragment return all k-mers according to the k-mer size.

        Arguments:
        fragment -- a Fragment representation of a sequence from which a frequency profile
                    is generated
    """

    kmer_counter = defaultdict(lambda: 0)
    for kmer in sequence_to_kmers(fragment.seq, fragment.kmer_size):
        kmer_counter[kmer] += 1

    kmers = list(kmer_counter.keys())
    kmer_counts = list(kmer_counter.values())
    return pl.DataFrame([
        pl.Series('Source', [fragment.source] * len(kmer_counter)),
        pl.Series('Contig', [fragment.contig] * len(kmer_counter)),
        pl.Series('Fragment', [fragment.name] * len(kmer_counter)),
        pl.Series('Kmer', kmers),
        pl.Series('Count', kmer_counts)
    ])

#endregion

#region Other functions

def write_fragmented_reads(fragment_list: List[Fragment], file_path: str) -> None:
    """ Save the fragmented sequences into a fasta file, using the fragment name as sequence names.

        Arguments:
        fragment_list -- a list of Fragment elements to be written to the output file
        file_path     -- the destination for Fragment elements to be saved
    """

    with open(file_path, 'w') as fna_writer:
        for fragment in fragment_list:
            fna_writer.write(fragment.create_line())

def deploy_multithreaded_function(argument_list: List[Any], callback: Callable, n_threads: int) -> List[Any]:
    """ Distribute the list of sequence dictionaries over the function across a user-specified number of threads.
        Function to be multithreaded is provided as a callback for unit testing.

        Arguments:
        argument_list -- a list of arguments to be passed into the callable function
        callback      -- the function to be enacted over the user-specified number of threads
        n_threads     -- the number of threads to use in parallelisation
    """

    with Pool(processes=n_threads) as pool:
        results = pool.map(callback, argument_list)

    return results

def combine_data_frames(input_frames: List[pl.DataFrame]) -> pl.DataFrame:
    """ Combine all fragment dataframes, sort the results and return a single DataFrame.
    
        Arguments:
        input_frames -- a list of pl.DataFrame objects to be concatenated and sorted
    """

    return (
        pl
        .concat(input_frames, how='vertical')
        .sort(by=['Source', 'Contig', 'Fragment', 'Kmer'], descending=False)
    )

#endregion

if __name__ == '__main__':
    main()
