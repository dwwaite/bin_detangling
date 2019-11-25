'''
    There are no tests for the section Bin refinement functions since these are just calls to the GenomeBin class.
'''
import sys
import io
import os
import unittest
import pandas as pd

class TestExpandByMcc(unittest.TestCase):

    def setUp(self):
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

    def spawn_mock_table(self, save_file=None):

        df = pd.DataFrame([ { 'ContigName': 'contig_1', 'V1': 0.50, 'V2': 0.50, 'BinID': 'bin_1', },
                            { 'ContigName': 'contig_2', 'V1': 0.55, 'V2': 0.45, 'BinID': 'bin_1', },
                            { 'ContigName': 'contig_3', 'V1': 1.10, 'V2': 2.00, 'BinID': 'bin_2', } ] )

        if save_file:
            df.to_csv(save_file, sep='\t', index=False)
            self.temp_file_buffer.append(save_file)

        return df

    def test_validate_input_data_all(self):

        df = self.spawn_mock_table(save_file='mock.txt')
        exp_tuples = [ ('bin_1', 'mock.txt', 1.0, 1, 'abcbin_1.refined'),
                       ('bin_2', 'mock.txt', 1.0, 1, 'abcbin_2.refined') ]

        bin_names = ['bin_1', 'bin_2']
        obs_tuples = expand_by_mcc.validate_input_data(bin_names, 'mock.txt', 1, 1.0, 'abc')

        self.assertEqual( len(exp_tuples), len(obs_tuples) )

        for e, o in zip(exp_tuples, obs_tuples):
            self.assertEqual(o, e)

    def test_validate_input_data_subset(self):

        df = self.spawn_mock_table(save_file='mock.txt')
        exp_tuples = [ ('bin_1', 'mock.txt', 1.0, 1, 'abcbin_1.refined') ]

        obs_tuples = expand_by_mcc.validate_input_data(['bin_1'], 'mock.txt', 1, 1.0, 'abc')
        self.assertEqual( 1, len(obs_tuples) )
        self.assertEqual( exp_tuples[0], obs_tuples[0] )

    def test_validate_input_data_autoselect(self):

        df = self.spawn_mock_table(save_file='mock.txt')
        exp_tuples = [ ('bin_1', 'mock.txt', 1.0, 1, 'abcbin_1.refined'),
                       ('bin_2', 'mock.txt', 1.0, 1, 'abcbin_2.refined') ]

        obs_tuples = expand_by_mcc.validate_input_data('-', 'mock.txt', 1, 1.0, 'abc')

        for e, o in zip(exp_tuples, obs_tuples):
            self.assertEqual(e, o)

    def test_validate_input_data_bad_selection(self):

        df = self.spawn_mock_table(save_file='mock.txt')

        self.start_logging_stdout()
        obs_tuples = expand_by_mcc.validate_input_data(['not_here'], 'mock.txt', 1, 1.0, 'abc')
        print_capture = self.stop_logging_stdout()

        self.assertIsNone(obs_tuples)
        self.assertIn('Unable to locate any valid contig lists. Aborting...', print_capture)

    # endregion

if __name__ == '__main__':

    ''' Import the project_tsne.py library '''
    sys.path.insert(0, '..')
    import expand_by_mcc

    unittest.main()