import unittest
import polars as pl
from polars.testing import assert_frame_equal

from bin.identify_bin_cores import GenomeBin
from bin.identify_bin_cores import spawn_bin_instances
from bin.identify_bin_cores import apply_core_members

class TestIdentifyBinCores(unittest.TestCase):

    @staticmethod
    def input_dataframe() -> pl.DataFrame:
        return pl.DataFrame([
            pl.Series('Source', ['bin_1', 'bin_1', 'bin_2', 'bin_1']),
            pl.Series('Fragment', ['1_1', '1_2', '2_1', '1_3']),
            pl.Series('TSNE_1', [1.0, 3.0, 10.0, 4.0]),
            pl.Series('TSNE_2', [2.0, 7.0, -1.0, 8.0])
        ])

    @staticmethod
    def sorted_dataframe() -> pl.DataFrame:
        return pl.DataFrame([
            pl.Series('Source', ['bin_1', 'bin_1', 'bin_1', 'bin_2']),
            pl.Series('Fragment', ['1_2', '1_3', '1_1', '2_1']),
            pl.Series('TSNE_1', [3.0, 4.0, 1.0, 10.0]),
            pl.Series('TSNE_2', [7.0, 8.0, 2.0, -1.0]),
            pl.Series('distance', [0.0, 1.414214, 5.385165, 10.630146]),
        ])

#region GenomeBin tests

    def test_constructor(self):

        target_bin = 'bin_1'
        exp_values = [0.0, 0.0, 0.0, 0.0]
        exp_df = TestIdentifyBinCores.sorted_dataframe()

        my_bin = GenomeBin(TestIdentifyBinCores.input_dataframe(), target_bin)

        self.assertEqual(target_bin, my_bin.target_bin)
        self.assertListEqual(exp_values, my_bin.mcc_values)
        assert_frame_equal(my_bin.coordinates, exp_df)

    def test_map_mcc_values(self):

        exp_results = [0.3333333333333333, 0.5773502691896258, 1.0, 0.0]

        my_bin = GenomeBin(TestIdentifyBinCores.input_dataframe(), 'bin_1')
        my_bin.map_mcc_values()

        self.assertListEqual(exp_results, my_bin.mcc_values)

    def test_report_core_fragments_pass(self):

        my_bin = GenomeBin(TestIdentifyBinCores.input_dataframe(), 'bin_1')
        my_bin.map_mcc_values()
        obs_results = my_bin.report_core_fragments(0.99)

        self.assertSetEqual({'1_1', '1_2', '1_3'}, obs_results)

    def test_report_core_fragments_fail(self):

        my_bin = GenomeBin(TestIdentifyBinCores.input_dataframe(), 'bin_1')
        my_bin.map_mcc_values()
        obs_results = my_bin.report_core_fragments(1.1)

        self.assertSetEqual(set([]), obs_results)

    
    def test_report_core_fragments_list(self):

        # Manually create a messier data set than above. Expectation is that fragment 1_1 is not
        # recruited to the core.
        input_df = pl.DataFrame([
            pl.Series('Source', ['bin_1', 'bin_1', 'bin_2', 'bin_1', 'bin_2', 'bin_3']),
            pl.Series('Fragment', ['1_1', '1_2', '2_1', '1_3', '2_2', '3_1']),
            pl.Series('TSNE_1', [1.0, 3.0, 10.0, 4.0, 11.1, 1.1]),
            pl.Series('TSNE_2', [2.0, 7.0, -1.0, 8.0, -1.1, 2.1])
        ])
        input_bins = [GenomeBin(input_df, 'bin_1'), GenomeBin(input_df, 'bin_2'), GenomeBin(input_df, 'bin_3')]

        exp_results = [set(['1_2', '1_3']), set(['2_1', '2_2']), set(['3_1'])]

        for input_bin, exp_result in zip(input_bins, exp_results):

            input_bin.map_mcc_values()
            obs_result = input_bin.report_core_fragments(0.7)

            self.assertSetEqual(exp_result, obs_result)

    def test_identify_centroid(self):

        input_df = TestIdentifyBinCores.input_dataframe()
        exp_result = (3.0, 7.0)

        obs_result = GenomeBin._identify_centroid(input_df, 'bin_1')
        self.assertTupleEqual(exp_result, obs_result)

    def test_calculate_distance(self):

        obs_result = GenomeBin._calculate_distance(3.0, 4.0)
        self.assertEqual(5.0, obs_result)

    def test_sort_bin_view(self):

        input_df = TestIdentifyBinCores.input_dataframe()
        exp_result = TestIdentifyBinCores.sorted_dataframe()

        obs_result = GenomeBin._sort_bin_view(input_df, 'bin_1')
        assert_frame_equal(exp_result, obs_result)

    def test_extract_bin_name(self):

        my_bin = GenomeBin(TestIdentifyBinCores.input_dataframe(), 'bin_1')
        my_bin.target_bin = 'abc/def.ghi.txt'

        self.assertEqual('def.ghi', GenomeBin.extract_bin_name(my_bin))

#endregion

    def test_spawn_bin_instances(self):

        input_df = TestIdentifyBinCores.input_dataframe()
        exp_results = [GenomeBin(input_df, 'bin_1'), GenomeBin(input_df, 'bin_2')]
        obs_results = spawn_bin_instances(input_df)
        self.assertListEqual(exp_results, obs_results)

    def test_apply_core_members(self):

        input_df = TestIdentifyBinCores.input_dataframe()
        exp_df = input_df.with_columns(pl.Series('Core', [True, True, True, False]))

        obs_df = apply_core_members(input_df, set(['1_1', '1_2', '2_1']))
        assert_frame_equal(exp_df, obs_df)

if __name__ == '__main__':

    unittest.main()
