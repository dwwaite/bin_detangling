import sys
import io
import os
import unittest
import pandas as pd

class TestProjectTsne(unittest.TestCase):

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
        self.assertListEqual( list(genome_bin.esom_table.columns), ['BinID', 'ContigBase', 'ContigName', 'V1', 'V2', 'Distance'])
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

        pass
        #genome_bin, _ = self.instantiate_bin()
        #genome_bin.ComputeCloudPurity()



    """
    def ComputeCloudPurity(self):

        for frame_slice in self._get_next_slice():

            ''' Create a UUID for storing contig records '''
            slice_key = uuid.uuid4()          

            ''' Calculate the MCC '''
            slice_mcc = self._compute_mcc(frame_slice)

            ''' Record the perimeter and area of the point slice.
                Note that these are special cases of the ConvexHull for a 2D shape. If we project to more dimensions, this will no longer be valid '''
            q_hull = ConvexHull( frame_slice[ ['V1', 'V2'] ].values )
            slice_area = q_hull.volume
            slice_perimeter = q_hull.area

            self._store_quality_values(slice_mcc, slice_key, slice_area, slice_perimeter, frame_slice )

    def ResolveUnstableContigs(self, fragment_count_dict, qManager):

        ''' Find the key that corresponds to the top MCC, then cast out a list of the ContigBase names within this '''
        top_df = self._slice_df_lookup[ self.top_key ]

        top_contigs = top_df[ top_df.BinID == self.bin_name ].ContigBase.unique()

        ''' For each fo these contigs, remove it if it does not pass the bias_threshold '''
        core_contigs = set(top_contigs)
        for contig in top_contigs:

            contig_fragment_bin_dist = top_df[ top_df.ContigBase == contig ].BinID.value_counts()

            self_fragments = contig_fragment_bin_dist[ self.bin_name ]
            total_fragments = fragment_count_dict[contig]

            if float(self_fragments) / total_fragments < self.bias_threshold:
                core_contigs.remove(contig)

        ''' Store dicts of bin/contig, for return to the main process '''
        for c in core_contigs:
            qManager.put( { 'Bin': self.bin_name, 'Contig': c } )

    """

    # endregion

if __name__ == '__main__':

    ''' Import the parent path, so that we can import the scripts folder '''
    sys.path.insert(0, '..')
    from scripts.GenomeBin import GenomeBin

    unittest.main()