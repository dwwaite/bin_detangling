#import sys, os
#import pandas as pd
#from multiprocessing import Pool
from dataclasses import dataclass, field
from optparse import OptionParser
from typing import List, Tuple, Set

import numpy as np
import polars as pl
from sklearn.metrics import matthews_corrcoef

@dataclass
class GenomeBin:
    coordinates: pl.DataFrame
    target_bin: str
    mcc_values: list[float] = field(default_factory=list)

    def __init__(self, df: pl.DataFrame, target_bin: str):

        self.coordinates = GenomeBin._sort_bin_view(df, target_bin)
        self.target_bin = target_bin
        self.mcc_values = [0.0] * df.shape[0]

    def map_mcc_values(self) -> None:
        """ Identify the contig fragments required to build a minimum viable bin core. Sets the value
            of self.mcc_values for the MCC scores for each incremental inclusion of fragments from the
            first to last in the data frame.
        """

        known_sequence = [b == self.target_bin for b in self.coordinates.get_column('Source')]
        masked_sequence = [False] * len(known_sequence)

        for i in range(0, len(known_sequence)):
            masked_sequence[i] = True
            self.mcc_values[i] = matthews_corrcoef(known_sequence, masked_sequence)

    def report_core_fragments(self, mcc_threshold: float) -> Set[str]:
        """ Returns a set of fragments representing the maximum MCC value for the bin. If this
            value does not surpass the specified threshold, an empty set is returned.
        """

        max_mcc = max(self.mcc_values)

        if max_mcc < mcc_threshold:
            return set([])

        else:

            max_index = self.mcc_values.index(max_mcc)
            retain_mask = [i <= max_index for i, _ in enumerate(self.mcc_values)]

            return set(
                self.coordinates
                .with_columns(pl.Series('mask', retain_mask))
                .filter(
                    pl.col('mask')
                )
                .get_column('Fragment')
            )

    @staticmethod
    def _identify_centroid(df: pl.DataFrame, target_bin: str) -> Tuple[float, float]:
        """ Identify the centroid coordinates for the bin of interest, returning a tuple of the x- and y-coordinates.
        """
        return (
            df
            .filter(pl.col('Source').eq(target_bin))
            .select(
                pl.col('TSNE_1').median(),
                pl.col('TSNE_2').median(),
            )
            .row(0)
        )

    @staticmethod
    def _calculate_distance(dist_x: float, dist_y: float) -> float:
        """ Return the diagonal distance between the x- and y-edges of a triangle.
        """
        return np.sqrt(dist_x ** 2 + dist_y ** 2)

    @staticmethod
    def _sort_bin_view(df: pl.DataFrame, target_bin: str) -> pl.DataFrame:
        """ Calculate the Pythagorean distance between the TSNE_1 and TSNE_2 coordinates and the
            provided coordinates. The data is then sorted according to ascending distance from the
            coordinates.
        """

        x_value, y_value = GenomeBin._identify_centroid(df, target_bin)

        return (
            df
            .with_columns(
                delta_1=pl.col('TSNE_1').map_elements(lambda v: abs(v - x_value)),
                delta_2=pl.col('TSNE_2').map_elements(lambda v: abs(v - y_value)),
            )
            .with_columns(
                distance=pl.struct(['delta_1', 'delta_2']).map_elements(lambda x: GenomeBin._calculate_distance(*x.values()))
            )
            .drop('delta_1', 'delta_2')
            .sort('distance', descending=False)
        )

def main():
    
    # Set up the options
    parser = OptionParser()

    parser.add_option('-i', '--input', help='The output parquet file produced by project_ordination.py', dest='input')
    parser.add_option('--threshold', help='The minimum MCC value for accepting a bin core (Default: 0.8)', dest='threshold', default=0.8, type=float)
    parser.add_option('-o', '--output', help='An output table with bin cores identified', dest='output')

    options, _ = parser.parse_args()

    # Import data and extract the list of bin names
    df = pl.read_parquet(options.input)

    core_fragments = identify_core_members(df, options.threshold)
    master_df = apply_core_members(df, core_fragments)
    master_df.write_parquet(options.output)

def identify_core_members(df: pl.DataFrame, mcc_threshold: float) -> Set[str]:
    """ Build a set of the combined fragments over all bins which form the core, respecting
        the threshold of GenomeBin.report_core_fragments(), then return the total set
        of fragments.
    """

    core_fragments = set()

    for bin_entry in df.get_column('Source').unique():
        my_bin = GenomeBin(df, bin_entry)
        my_bin.map_mcc_values()
        core_fragments |= my_bin.report_core_fragments(mcc_threshold)

    return core_fragments

def apply_core_members(df: pl.DataFrame, core_names: Set[str]) -> pl.DataFrame:
    """ Attach the core fragment identities to the original DataFrame and return.
    """
    return df.with_columns(Core=pl.col('Fragment').is_in(core_names))

if __name__ == '__main__':
    main()