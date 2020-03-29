import sys
import io
import os
import unittest
import pandas as pd
import numpy as np

class TestGenomeBin(unittest.TestCase):

    def setUp(self):
        #@unittest.skip('Not implemented yet')
        self.temp_file_buffer = []

    def tearDown(self):

        for temp_file in self.temp_file_buffer:
            os.remove(temp_file)

    # region Capture stdout for evaluating print() statements

    def start_logging_stdout(self):
            self.print_capture = io.StringIO()
            sys.stdout = self.print_capture

    def stop_logging_stdout(self):
            sys.stdout = sys.__stdout__
            return self.print_capture.getvalue()

    # endregion

    # region Tests for the constructor

    def spawn_mock_table(self, save_file=None):

        df = pd.DataFrame([ { 'ContigBase': 'contig_1', 'ContigName': 'contig_1|1', 'V1': 0.1, 'V2': 0.1, 'BinID': 'bin_1', },
                            { 'ContigBase': 'contig_1', 'ContigName': 'contig_1|2', 'V1': 0.4, 'V2': 0.2, 'BinID': 'bin_1', },
                            { 'ContigBase': 'contig_1', 'ContigName': 'contig_1|3', 'V1': 0.5, 'V2': 0.1, 'BinID': 'bin_1', },
                            { 'ContigBase': 'contig_1', 'ContigName': 'contig_1|4', 'V1': 0.2, 'V2': -0.3, 'BinID': 'bin_1', },
                            { 'ContigBase': 'contig_1', 'ContigName': 'contig_1|5', 'V1': 0.3, 'V2': -0.2, 'BinID': 'bin_1', },
                            { 'ContigBase': 'contig_2', 'ContigName': 'contig_2|1', 'V1': 0.5, 'V2': -0.2, 'BinID': 'bin_2', },
                            { 'ContigBase': 'contig_1', 'ContigName': 'contig_1|6', 'V1': 0.6, 'V2': -0.3, 'BinID': 'bin_1', },
                            { 'ContigBase': 'contig_3', 'ContigName': 'contig_3|1', 'V1': 0.6, 'V2': 0.4, 'BinID': 'bin_1', },
                            { 'ContigBase': 'contig_3', 'ContigName': 'contig_3|2', 'V1': 0.8, 'V2': 0.3, 'BinID': 'bin_1', } ] )

        if save_file:
            df.to_csv(save_file, sep='\t', index=False)
            self.temp_file_buffer.append(save_file)

        return df

    def instantiate_bin(self):
        df = self.spawn_mock_table(save_file='mock.txt')
        return GenomeBin(bin_name='bin_1', esom_path='mock.txt', bias_threshold=1.0, number_of_slices=3, output_path='debug'), df

    def test_constructor(self):

        genome_bin, _ = self.instantiate_bin()

        self.assertEqual(genome_bin.bin_name, 'bin_1')
        self.assertEqual(genome_bin.output_path, 'debug')
        self.assertEqual(genome_bin.bias_threshold, 1.0)
        self.assertEqual(genome_bin.number_of_slices, 3)
        self.assertIsNotNone(genome_bin.esom_table)

        self.assertListEqual(genome_bin.mcc_expectation, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertListEqual(genome_bin._iteration_scores, [])
        self.assertDictEqual(genome_bin._slice_df_lookup, {})

        self.assertIsNone(genome_bin._core_contigs)

    def test_esom_df(self):

        genome_bin, df = self.instantiate_bin()

        ''' Test all columns are preserved and number of rows persists. Columns should be df + 1 due to the addition of the Distance column '''

        ''' Check columns as a set, so that I do't fail on order '''
        self.assertSetEqual( set(genome_bin.esom_table.columns), set(['BinID', 'ContigBase', 'ContigName', 'V1', 'V2', 'Distance']) )

        self.assertEqual( df.shape[0], genome_bin.esom_table.shape[0] )
        self.assertEqual( df.shape[1] + 1, genome_bin.esom_table.shape[1] )

        ''' Test that the Distance column is sorted correctly '''
        for i in range(1, genome_bin.esom_table.shape[0]):
            self.assertLessEqual( genome_bin.esom_table.Distance[i-1], genome_bin.esom_table.Distance[i] )

        ''' Test the centroid value '''
        cen_x, cen_y = genome_bin.centroid
        self.assertEqual(cen_x, 0.45)
        self.assertEqual(cen_y, 0.1)

    # endregion

    # region Tests for the section Internal manipulation functions and constructor

    def test_calc_dist(self):

        genome_bin, _ = self.instantiate_bin()

        # Test two options, one using distance from 0, and one using negative values
        obs_dist = genome_bin._calc_dist(0, 0, 3, 4)
        self.assertEqual(obs_dist, 5.0)

        obs_dist = genome_bin._calc_dist(-1, 0, 2, -4)
        self.assertEqual(obs_dist, 5.0)

    def test_get_next_slice(self):

        genome_bin, _ = self.instantiate_bin()
        exp_sizes = [4, 7, 9]

        obs_sizes = [ df.shape[0] for df in genome_bin._get_next_slice() ]
        self.assertListEqual(obs_sizes, exp_sizes)

    def test_compute_mcc(self):

        genome_bin, _ = self.instantiate_bin()
        exp_mcc = [-0.3952847075210474, -0.1889822365046136, 0.0]

        obs_mcc = [ genome_bin._compute_mcc(df) for df in genome_bin._get_next_slice() ]
        self.assertListEqual(obs_mcc, exp_mcc)

    def test_store_quality_values(self):

        genome_bin, _ = self.instantiate_bin()
        exp_dict = { 'MCC': 0.75, 'Key': 'abcd', 'Area': 2.0, 'Perimeter': 1.0 }

        genome_bin._store_quality_values(exp_dict['MCC'], exp_dict['Key'], exp_dict['Area'], exp_dict['Perimeter'], [])
        self.assertDictEqual( genome_bin._iteration_scores[0], exp_dict )
        self.assertListEqual( genome_bin._slice_df_lookup[ exp_dict['Key'] ], [] )

    def store_slices(self, g_bin):

        for key, df_slice in enumerate( g_bin._get_next_slice() ):
            g_bin._store_quality_values(1.0, str(key), 2.0, 3.0, df_slice )

        return key + 1

    def test_slice_contigs_bin(self):

        genome_bin, _ = self.instantiate_bin()
        max_key = self.store_slices(genome_bin)

        exp_sizes = [3, 6, 8]

        for i in range(0, max_key):
            df = genome_bin._slice_contigs_bin( str(i) )

            self.assertEqual(df.shape[0], exp_sizes[i])
            self.assertSetEqual( set(df.BinID), set(['bin_1']) )

    def test_slice_contigs_nonbin(self):

        genome_bin, _ = self.instantiate_bin()
        max_key = self.store_slices(genome_bin)

        for i in range(0, max_key):
            df = genome_bin._slice_contigs_nonbin( str(i) )

            self.assertEqual(df.shape[0], 1)
            self.assertSetEqual( set(df.BinID), set(['bin_2']) )

    # endregion

    # region Tests for the section Externally exposed functions

    def test_computeCloudPurity(self):

        ''' Set a list of dicts of expected values '''
        exp_scores = [ {'MCC': -0.395, 'Area': 0.05, 'Perimeter': 1.30, 'Rows': 4 },
                       {'MCC': -0.189, 'Area': 0.23, 'Perimeter': 1.95, 'Rows': 7 },
                       {'MCC': 0.0, 'Area': 0.325, 'Perimeter': 2.25, 'Rows': 9 } ]

        genome_bin, _ = self.instantiate_bin()
        genome_bin.ComputeCloudPurity()

        ''' Test each entry saved by the ComputeCloudPurity() function '''
        for exp_dict, obs_dict in zip(exp_scores, genome_bin._iteration_scores):

            self.assertTrue( np.isclose(exp_dict['MCC'], obs_dict['MCC'], atol=3))
            self.assertTrue( np.isclose(exp_dict['Area'], obs_dict['Area'], atol=3) )
            self.assertTrue( np.isclose(exp_dict['Perimeter'], obs_dict['Perimeter'], atol=3) )

            self.assertIn('Key', obs_dict)
            obs_slice = genome_bin._slice_df_lookup[ obs_dict['Key'] ]
            self.assertEqual( exp_dict['Rows'], obs_slice.shape[0])

    def test_ResolveUnstableContigs(self):

        ''' Set a list of dicts of expected values '''
        exp_bins = set(['bin_1'])
        exp_contigs = set(['contig_1', 'contig_3'])

        genome_bin, df = self.instantiate_bin()
        genome_bin.ComputeCloudPurity()

        mgmr = ThreadManager(2, print)
        genome_bin.ResolveUnstableContigs({'contig_1': 6, 'contig_2': 1, 'contig_3': 2}, mgmr.queue)

        ''' Need to package the output, to avoid getting a false negative due to sorting issues '''
        resolved_contigs = mgmr.results
        obs_bins = set([ d['Bin'] for d in resolved_contigs ])
        obs_contigs = set([ d['Contig'] for d in resolved_contigs ])

        self.assertSetEqual(exp_bins, obs_bins)
        self.assertSetEqual(exp_contigs, obs_contigs)

    # endregion

if __name__ == '__main__':

    ''' Import the parent path, so that we can import the scripts folder '''
    sys.path.insert(0, '..')
    from scripts.GenomeBin import GenomeBin
    from scripts.ThreadManager import ThreadManager

    unittest.main()