import os
import argparse
from typing import List, Tuple, Set
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import plotly.express as px
from sklearn.metrics import matthews_corrcoef

@dataclass
class GenomeBin:
    coordinates: pl.DataFrame
    target_bin: str
    mcc_values: List[float] = field(default_factory=list)

    def __init__(self, df: pl.DataFrame, target_bin: str) -> 'GenomeBin':

        self.coordinates = GenomeBin._sort_bin_view(df, target_bin)
        self.target_bin = target_bin
        self.mcc_values = [0.0] * df.shape[0]

    def __eq__(self, other_bin: 'GenomeBin') -> bool:

        coord_eq = self.coordinates.equals(other_bin.coordinates)
        target_eq = self.target_bin == other_bin.target_bin
        mcc_eq = self.mcc_values == other_bin.mcc_values

        return coord_eq & target_eq & mcc_eq

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
                .filter(pl.col('mask'))
                .get_column('Fragment')
            )

    def plot_mcc_trace(self, core_fragments: Set[str], bin_name: str, file_path: str) -> None:
        """ Produce a bar plot of the MCC values of the bin, denoting the threshold and membership of each contig.
        """

        target_core = 'Target (core)'
        target_noncore = 'Target (non-core)'
        non_target = 'Non-target'

        # Apply additional plotting columns to the core DataFrame
        plot_df = (
            self
            .coordinates
            .with_columns(
                pl.Series('x_values', [i+1 for i in range(0, len(self.mcc_values))]),
                pl.when(
                    pl.col('Source').eq(self.target_bin),
                    pl.col('Fragment').is_in(core_fragments)
                )
                .then(pl.lit(target_core))
                .when(pl.col('Source').eq(self.target_bin))
                .then(pl.lit(target_noncore))
                .otherwise(pl.lit(non_target))
                .alias('x_colours')
            )
        )

        fig = px.bar(
            x=plot_df['x_values'], y=self.mcc_values, color=plot_df['x_colours'], title=f"MCC progression: {bin_name}",
            labels={
                'x': 'Contig inclusion',
                'y': 'MCC',
                'colour': 'Classification'
            },
            color_discrete_map={
                target_core: 'rgb(0, 136, 55)',
                target_noncore: 'rgb(166, 219, 160)',
                non_target: 'rgb(194, 165, 207)'
            }
        )
        fig.write_html(file_path)

    @staticmethod
    def _identify_centroid(df: pl.DataFrame, target_bin: str) -> Tuple[float, float]:
        """ Identify the centroid coordinates for the bin of interest, returning a tuple of the x- and y-coordinates.
        """
        return (
            df
            .filter(pl.col('Source').eq(target_bin))
            .select(pl.col('TSNE_1').median(), pl.col('TSNE_2').median())
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
                delta_1=pl.col('TSNE_1').map_elements(lambda v: abs(v - x_value), return_dtype=pl.Float64),
                delta_2=pl.col('TSNE_2').map_elements(lambda v: abs(v - y_value), return_dtype=pl.Float64),
            )
            .with_columns(
                distance=pl.struct(['delta_1', 'delta_2']).map_elements(lambda x: GenomeBin._calculate_distance(*x.values()), return_dtype=pl.Float64)
            )
            .drop('delta_1', 'delta_2')
            .sort('distance', descending=False)
        )

    @staticmethod
    def extract_bin_name(genome_bin: 'GenomeBin') -> str:
        """ Strip the genome bin `target_bin` parameter of file path and extension and return the result
            as a proxy for the bin name.
        """
        _, bin_file = os.path.split(genome_bin.target_bin)
        bin_name, _ = os.path.splitext(bin_file)

        return bin_name


def main():
    
    # Set up the options
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='The output parquet file produced by project_ordination.py')
    parser.add_argument('--threshold', default=0.8, type=float, help='The minimum MCC value for accepting a bin core (Default: 0.8)')
    parser.add_argument('--plot_traces', action='store_true', help='Store the MCC traces for each examined bin (Default: False)')
    parser.add_argument('-o', '--output', help='An output table with bin cores identified')

    options = parser.parse_args()

    # Import data and create GenomeBin records
    df = pl.read_parquet(options.input)
    bin_instances = spawn_bin_instances(df)

    # Identify the MCC progression and core contigs, then save it required.
    core_fragments = set()

    for bin_instance in bin_instances:

        bin_instance.map_mcc_values()
        bin_core = bin_instance.report_core_fragments(options.threshold)

        core_fragments |= bin_core

        if options.plot_traces:
            bin_name = GenomeBin.extract_bin_name(bin_instance)
            bin_instance.plot_mcc_trace(bin_core, bin_name, f"{options.output}.{bin_name}.html")

    # Map the results back to the input DataFrame, and store.
    master_df = apply_core_members(df, core_fragments)
    master_df.write_parquet(options.output)

def spawn_bin_instances(df: pl.DataFrame) -> List[GenomeBin]:
    """ Iterate through the input data frame and create an alphabetical list of
        GenomeBin objects representing the contents.
    """

    bin_sequence = sorted(df.get_column('Source').unique())
    return [GenomeBin(df, bin_entry) for bin_entry in bin_sequence]

def apply_core_members(df: pl.DataFrame, core_names: Set[str]) -> pl.DataFrame:
    """ Attach the core fragment identities to the original DataFrame and return.
    """
    return df.with_columns(Core=pl.col('Fragment').is_in(core_names))

if __name__ == '__main__':
    main()