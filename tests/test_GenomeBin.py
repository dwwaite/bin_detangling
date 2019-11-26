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

    # endregion

    # region Tests for the section Internal manipulation functions and constructor
    
    def spawn_mock_table(self, save_file=None):

        df = pd.DataFrame([ { 'ContigBase': 'contig_1', 'ContigName': 'contig_1|1', 'V1': 0.50, 'V2': 0.50, 'BinID': 'bin_1', },
                            { 'ContigBase': 'contig_1', 'ContigName': 'contig_1|2', 'V1': 0.55, 'V2': 0.45, 'BinID': 'bin_1', },
                            { 'ContigBase': 'contig_2', 'ContigName': 'contig_2|1', 'V1': 0.70, 'V2': 0.55, 'BinID': 'bin_2', },
                            { 'ContigBase': 'contig_3', 'ContigName': 'contig_3|1',  'V1': 0.80, 'V2': 0.65, 'BinID': 'bin_1', } ] )

        if save_file:
            df.to_csv(save_file, sep='\t', index=False)
            self.temp_file_buffer.append(save_file)

        return df

    def test_constructor(self):

        df = self.spawn_mock_table(save_file='mock.txt')
        genome_bin = GenomeBin(bin_name='bin_1', esom_path='mock.txt', bias_threshold=0.5, number_of_slices=1, output_path='debug')

        self.assertEqual(genome_bin.bin_name, 'bin_1')
        self.assertEqual(genome_bin.output_path, 'debug')
        self.assertEqual(genome_bin.bias_threshold, 0.5)
        self.assertEqual(genome_bin.number_of_slices, 1)
        self.assertIsNotNone(genome_bin.esom_table)

        self.assertListEqual(genome_bin.mcc_expectation, [1.0, 1.0, 0.0, 1.0])
        self.assertListEqual(genome_bin._iteration_scores, [])
        self.assertDictEqual(genome_bin._slice_df_lookup, {})

        self.assertIsNone(genome_bin._core_contigs)

    def test_esom_df(self):

        df = self.spawn_mock_table(save_file='mock.txt')
        genome_bin = GenomeBin(bin_name='bin_1', esom_path='mock.txt', bias_threshold=0.5, number_of_slices=1, output_path='debug')

        ''' Test all columns are preserved and number of rows persists. Columns should be df + 1 due to the addition of the Distance column '''
        self.assertListEqual( list(genome_bin.esom_table.columns), ['BinID', 'ContigBase', 'ContigName', 'V1', 'V2', 'Distance'])
        self.assertEqual( df.shape[0], genome_bin.esom_table.shape[0] )
        self.assertEqual( df.shape[1] + 1, genome_bin.esom_table.shape[1] )

        ''' Test that the Distance column is sorted correctly '''
        for i in range(1, genome_bin.esom_table.shape[0]):
            self.assertLessEqual( genome_bin.esom_table.Distance[i-1], genome_bin.esom_table.Distance[i] )

        ''' Test the centroid value '''
        cen_x, cen_y = genome_bin.centroid
        self.assertEqual(cen_x, 0.55)
        self.assertEqual(cen_y, 0.50)
   
    """
        def _calc_dist(self, xCen, yCen, xPos, yPos):
            dX = np.array(xCen - xPos)
            dY = np.array(yCen - yPos)    
            return np.sqrt( dX ** 2 + dY ** 2 )

        def _get_next_slice(self):

            ''' Pull the indices for the contig fragments in the bin '''
            index_list = list( self.esom_table[ self.esom_table.BinID == self.bin_name ].index )
            n_contigs = len(index_list)

            ''' Divide the index list into N slices, following a sine function '''
            for x in range(1, self.number_of_slices + 1):

                slice_index = np.sin( x / self.number_of_slices * np.pi/2 ) * n_contigs
                slice_index = int(slice_index)
                curr_slice = index_list[ slice_index ]

                yield self.esom_table.iloc[ 0:curr_slice, ]

                ''' Break the loop if we've hit the end of the curve early.
                    This can happen for the last entry due to int rounding of the index '''
                if slice_index + 1 == n_contigs:
                    break

        def _compute_mcc(self, slice_df):

            ''' There is an edge case where if there are no false contigs in a bin the MCC encounters a divide by zero.
                    Generally this doesn't matter, because it results in a vector of 0.0 for the MCC, and the larger slice is reported
                    for MCC ties.
                    That said, this should be revised in the future.

            '''
            obs_vector = [0.0] * len( self.mcc_expectation )
            for i in range(0, slice_df.shape[0]): obs_vector[i] = 1.0

            '''
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    mcc = ...
                except:
                    return ...
            '''
            return matthews_corrcoef(self.mcc_expectation, obs_vector)

        def _store_quality_values(self, mcc, slice_key, area, perimeter, frame_slice):

            self._iteration_scores.append( { 'MCC': mcc, 'Key': slice_key, 'Area': area, 'Perimeter': perimeter })
            self._slice_df_lookup[ slice_key ] = frame_slice

        def _slice_contigs_bin(self, slice_key):

            df = self._slice_df_lookup[ slice_key ]
            return df[ df.BinID == self.bin_name ]

        def _slice_contigs_nonbin(self, slice_key):

            df = self._slice_df_lookup[ slice_key ]
            return df[ df.BinID != self.bin_name ]
    """

    # endregion

    # region Tests for the section Externally exposed functions

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