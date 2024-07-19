import argparse
from typing import Generator, List, Tuple

import polars as pl

def main():

    # Set up the options
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='Output file for depth profile results',)
    parser.add_argument(
        'input_files', metavar='MAPPING_FILES', nargs='+',
        help='Mapping files (bam) for examining coverage profiles'
    )
    options = parser.parse_args()

    # Process the data
    depth_frames = [parse_depth_file(file_path, depth_label) for file_path, depth_label in depth_label_generator(options.input_files)]
    pl.concat(depth_frames, how='vertical').write_parquet(options.output)

def depth_label_generator(file_paths: List[str]) -> Generator[Tuple[str, str], None, None]:
    """ Creates a generator returning tuples of the file path and depth label for each file
        in the input sequence. 
    """
    for i, file_path in enumerate(file_paths):
        yield (file_path, f"Depth_{i+1}")

def parse_depth_file(file_path: str, depth_label: str) -> pl.DataFrame:
    """ Reads and summarises a headerless mapping file produced by samtools. Returns a DataFrame with
        the columns `Contig`, `Coverage`, and `Label`.
        Reports median depth over each contig to allow some weighting to the linkage across contig
        fragments produced under compute_kmer_profile.py.
    """
    return (
        pl
        .scan_csv(file_path, separator='\t', has_header=False, new_columns=['Contig', 'pos', 'depth'])
        .group_by(['Contig'])
        .agg(
            pl.col('depth').median().alias('Coverage')
        )
        .with_columns(
            Label=pl.lit(depth_label)
        )
        .collect()
    )

if __name__ == '__main__':
    main()
