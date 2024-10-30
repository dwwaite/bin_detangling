import argparse
from typing import Counter, Dict

import polars as pl

def main():

    # Parse options
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='A recruitment file produced by the `recruit_by_ml.py recruit` script')
    parser.add_argument('-c', '--core', help='The core contig file produced by the `identify_bin_cores.py` script')
    parser.add_argument('-t', '--threshold', type=float, help='The minimum support to recruit a contig to a bin')
    parser.add_argument(
        '-o', '--output',
        help=(
            'A summary table reporting the classification of each non-core contig and its support as a proportion '
            'of fragments recruited to the bin.'
        )
    )

    options = parser.parse_args()

    # Read data and summarise recruitment
    #recruit_map = import_recruitment(pl.scan_parquet(options.input))
    recruit_map = import_recruitment(pl.scan_parquet('results/raw_bins.recruited.parquet'))

    #df = build_assignment_table(pl.scan_parquet(options.core), recruit_map)
    df = build_assignment_table('results/raw_bins.tsne_core.parquet', recruit_map)

#region Functions

def import_recruitment(lf: pl.LazyFrame) -> Dict[str, str]:
    """ Cast the columns of a lazy frame into a dictionary mapping the Fragment column to
        the assigned bin (Bin).

        Arguments:
        lf -- the lazy representation of the data to be parsed
    """
    return {row[0]: row[1] for row in lf.select('Fragment', 'Bin').collect().iter_rows()}

def build_assignment_table(lf: pl.LazyFrame, recruit_map: Dict[str, str]) -> pl.DataFrame:
    """ Import the original core recruitment data and assign bin membership for each fragment. Original
        bin assignment (Source) is used for core fragments, and the recruited value (or unassigned) is used
        for non-core fragments.

        Arguments:
        lf          -- the lazy representation of the core mapping data
        recruit_map -- a dictionary mapping fragments names to assigned bins for non-core fragments
    """

    return (
        lf
        .with_columns(
            Bin=pl
            .when(pl.col('Core'))
            .then(pl.col('Source'))
            .otherwise(pl.col('Fragment').map_elements(lambda x: recruit_map.get(x), return_dtype=pl.Utf8))
        )
        .select('Source', 'Contig', 'Fragment', 'Bin')
        .collect()
    )

def get_top_bin(bin_series: pl.Series) -> pl.Struct:
    """ Tally the unique entries in a Series and return a pl.Struct representing the most abundant
        selection and the frequency at which is occurs in the Series.

        Arguments:
        bin_series -- a series or iterable of string values
    """

    sorted_counts = Counter(bin_series).most_common()
    top_bin, abund = sorted_counts[0]

    return pl.Series([{'Bin': top_bin, 'Support': abund / len(bin_series)}])[0]

def finalise_recruitment(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """
    """

    return (
        df
        .group_by(['Source', 'Contig'])
        .agg(
            pl.col('Bin').map_elements(lambda x: get_top_bin(x), return_dtype=pl.Struct)
        )
        .unnest('Bin')
    )

#endregion

if __name__ == '__main__':
    main()
