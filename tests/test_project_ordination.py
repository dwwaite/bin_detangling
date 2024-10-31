import sys
import unittest
import polars as pl
from polars.testing import assert_frame_equal

from bin.project_ordination import counts_to_frequencies
from bin.project_ordination import map_coverage_to_fragments
from bin.project_ordination import project_to_matrix
from bin.project_ordination import normalise_matrix
from bin.project_ordination import build_weighting_mask
from bin.project_ordination import compute_dist
from bin.project_ordination import project_tsne

class TestProjectOrdination(unittest.TestCase):

    @staticmethod
    def count_matrix_long() -> pl.DataFrame:
        return pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'b', 'b', 'b', 'b', 'c']),
            pl.Series('Contig', ['a1', 'a1', 'b1', 'b1', 'b1', 'b1', 'c1']),
            pl.Series('Fragment', ['a1_1', 'a1_1', 'b1_1', 'b1_1', 'b1_1', 'b1_2', 'c1_1']),
            pl.Series('Kmer', ['AAAA', 'AAAC', 'AAAA', 'AAAC', 'AAAG', 'AAAA', 'AAAA']),
            pl.Series('Count', [1, 1, 1, 1, 2, 2, 1]),
        ])

    @staticmethod
    def frequency_matrix_long() -> pl.DataFrame:
        return pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'b', 'b', 'b', 'b', 'c']),
            pl.Series('Contig', ['a1', 'a1', 'b1', 'b1', 'b1', 'b1', 'c1']),
            pl.Series('Fragment', ['a1_1', 'a1_1', 'b1_1', 'b1_1', 'b1_1', 'b1_2', 'c1_1']),
            pl.Series('Feature', ['Freq_AAAA', 'Freq_AAAC', 'Freq_AAAA', 'Freq_AAAC', 'Freq_AAAG', 'Freq_AAAA', 'Freq_AAAA']),
            pl.Series('Value', [0.5, 0.5, 0.25, 0.25, 0.5, 1.0, 1.0]),
        ])

    @staticmethod
    def frequency_matrix_wide() -> pl.DataFrame:
        return pl.DataFrame([
            pl.Series('Source', ['a', 'b', 'b', 'c']),
            pl.Series('Contig', ['a1', 'b1', 'b1', 'c1']),
            pl.Series('Fragment', ['a1_1', 'b1_1', 'b1_2', 'c1_1']),
            pl.Series('Freq_AAAA', [0.5, 0.25, 1.0, 1.0]),
            pl.Series('Freq_AAAC', [0.5, 0.25, 0.0, 0.0]),
            pl.Series('Freq_AAAG', [0.0, 0.5, 0.0, 0.0])
        ])

# region Normalisation and transformation

    def test_map_coverage_to_fragments(self):

        kmer_df = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'a', 'b', 'c']),
            pl.Series('Contig', ['a1', 'a1', 'a2', 'b1', 'c1']),
            pl.Series('Fragment', ['a1_1', 'a1_2', 'a2_1', 'b1_1', 'c1_1']),
        ])

        input_df = pl.LazyFrame([
            pl.Series('Contig', ['a1', 'a2', 'b1', 'c1', 'a1']),
            pl.Series('Label', ['Depth_1', 'Depth_1', 'Depth_1', 'Depth_1', 'Depth_2']),
            pl.Series('Coverage', [1, 2, 3, 4, 5])
        ])

        exp_df = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'a', 'a', 'a', 'b', 'c']),
            pl.Series('Contig', ['a1', 'a1', 'a1', 'a1', 'a2', 'b1', 'c1']),
            pl.Series('Fragment', ['a1_1', 'a1_1', 'a1_2', 'a1_2', 'a2_1', 'b1_1', 'c1_1']),
            pl.Series('Feature', ['Depth_1', 'Depth_2', 'Depth_1', 'Depth_2', 'Depth_1', 'Depth_1', 'Depth_1']),
            pl.Series('Value', [1, 5, 1, 5, 2, 3, 4]),
        ])

        obs_df = map_coverage_to_fragments(input_df, kmer_df)
        assert_frame_equal(exp_df, obs_df.sort(['Source', 'Contig', 'Fragment', 'Feature']))

    def test_counts_to_frequencies(self):

        input_df = TestProjectOrdination.count_matrix_long()
        exp_df = TestProjectOrdination.frequency_matrix_long()

        obs_df = counts_to_frequencies(input_df)
        assert_frame_equal(exp_df, obs_df)

    def test_project_to_matrix(self):

        input_df = TestProjectOrdination.frequency_matrix_long()
        exp_df = TestProjectOrdination.frequency_matrix_wide()

        obs_df = project_to_matrix(input_df)
        assert_frame_equal(exp_df, obs_df)

    def test_normalise_matrix_unit(self):

        input_df = TestProjectOrdination.frequency_matrix_wide().drop('Source', 'Contig', 'Fragment')
        exp_df = (
            input_df
            .with_columns(
                Freq_AAAA=pl.Series('Freq_AAAA', [-0.57735, -1.347151, 0.96225, 0.96225]),
                Freq_AAAC=pl.Series('Freq_AAAC', [1.507557, 0.301511, -0.9045340, -0.904534]),
                Freq_AAAG=pl.Series('Freq_AAAG', [-0.57735, 1.732051, -0.57735, -0.57735]))
        )

        obs_df = normalise_matrix(input_df, 'unit')
        assert_frame_equal(exp_df, obs_df)

    def test_normalise_matrix_yeojohnson(self):

        input_df = TestProjectOrdination.frequency_matrix_wide().drop('Source', 'Contig', 'Fragment')
        exp_df = (
            input_df
            .with_columns(
                Freq_AAAA=pl.Series('Freq_AAAA', [-0.617343, -1.319965, 0.968654, 0.968654]),
                Freq_AAAC=pl.Series('Freq_AAAC', [1.354266, 0.566621, -0.960444, -0.960444]),
                Freq_AAAG=pl.Series('Freq_AAAG', [-0.57735, 1.732051, -0.57735, -0.57735])
            )
        )

        obs_df = normalise_matrix(input_df, 'yeojohnson')
        assert_frame_equal(exp_df, obs_df)

# endregion

# region Clustering

    def test_build_weighting_mask(self):

        column_order = ['Depth_1', 'Depth_2', 'Freq_A', 'Freq_T', 'Freq_G', 'Freq_C']
        coverage_weighting = 0.5

        # Split 50% of weight over 2 depths, and 50% over 4 frequencies
        exp_weights = [0.25, 0.25, 0.125, 0.125, 0.125, 0.125]

        obs_weights = build_weighting_mask(column_order, coverage_weighting)
        self.assertTrue(exp_weights, obs_weights)

    def test_compute_dist_uniform(self):

        input_df = pl.DataFrame([
            pl.Series('Depth_1', [1, 2, 3, 4]),
            pl.Series('Freq_AAAA', [1, 2, 3, 4])
        ])

        exp_df = pl.DataFrame([
            pl.Series('column_0', [0.0, 1.414214, 2.828427, 4.242641]),
            pl.Series('column_1', [1.414214, 0.0, 1.414214, 2.828427]),
            pl.Series('column_2', [2.828427, 1.414214, 0.0, 1.414214]),
            pl.Series('column_3', [4.242641, 2.828427, 1.414214, 0.0])
        ])

        obs_df = compute_dist(input_df)
        assert_frame_equal(exp_df, obs_df)

    def test_compute_dist_weighted(self):

        input_df = pl.DataFrame([
            pl.Series('Depth_1', [1, 2, 3, 4]),
            pl.Series('Freq_AAAA', [1, 2, 3, 4])
        ])

        exp_df = pl.DataFrame([
            pl.Series('column_0', [0.0, 1.0, 2.0, 3.0]),
            pl.Series('column_1', [1.0, 0.0, 1.0, 2.0]),
            pl.Series('column_2', [2.0, 1.0, 0.0, 1.0]),
            pl.Series('column_3', [3.0, 2.0, 1.0, 0.0])
        ])

        obs_df = compute_dist(input_df, weight_coverage=0.8)
        assert_frame_equal(exp_df, obs_df)

    def test_project_tsne(self):

        label_df = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'b', 'b']),
            pl.Series('Contig', ['a1', 'a1', 'b1', 'b2']),
            pl.Series('Fragment', ['a1_1', 'a1_2', 'b1_1', 'b2_1'])
        ])

        input_dist = pl.DataFrame([
            pl.Series('column_0', [0.0, 1.0, 2.0, 3.0]),
            pl.Series('column_1', [1.0, 0.0, 1.0, 2.0]),
            pl.Series('column_2', [2.0, 1.0, 0.0, 1.0]),
            pl.Series('column_3', [3.0, 2.0, 1.0, 0.0])
        ])

        exp_df = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'b', 'b']),
            pl.Series('Contig', ['a1', 'a1', 'b1', 'b2']),
            pl.Series('Fragment', ['a1_1', 'a1_2', 'b1_1', 'b2_1']),
            pl.Series('TSNE_1', [-59.168961, -52.248138, -44.48210, -37.561005]),
            pl.Series('TSNE_2', [-153.83548, -40.11422, 87.493782, 201.214996])
        ])

        obs_df = project_tsne(input_dist, label_df, perplexity=2, seed=5)
        assert_frame_equal(exp_df, obs_df, check_dtypes=False)

# endregion

if __name__ == '__main__':
    unittest.main()
