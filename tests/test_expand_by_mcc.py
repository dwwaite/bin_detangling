'''
    There are no tests for the section Bin refinement functions since these are just calls to the GenomeBin class.
'''
import sys
import io
import os
import unittest
from test_project_tsne import spawn_mock_table

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

    # region Tests for the section Input, parsing, and validation functions

    def test_validate_input_data():
        pass

    """

    def validate_input_data(bin_names, esom_table_name, number_of_slices, bias_threshold, output_file_prefix):

        ''' Common case, char - is passed denoting the use of all bins in the input table '''
        if len(bin_names) == 1 and bin_names[0] == '-':
            bin_names = list( pd.read_csv(esom_table_name, sep='\t').BinID.unique() )

        bin_precursors = GenomeBin.ParseStartingVariables(bin_names, esom_table_name, number_of_slices, bias_threshold, output_file_prefix)

        ''' Ensure there is at least one valid result '''
        if len(bin_precursors) == 0:
            print('Unable to locate any valid contig lists. Aborting...')
            sys.exit()

        return bin_precursors
    """

    # endregion

if __name__ == '__main__':

    ''' Import the project_tsne.py library '''
    sys.path.insert(0, '..')
    import project_tsne

    unittest.main()