'''
    Automated testing of the creation, insertion, update, and delete functionality of the DatabaseManipulator class
'''
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

    # region Tests for the section Pre-workflow overhead

    def spawn_mock_table(self, save_file=None):

        df = pd.DataFrame([ { 'Contig': 'contig_1', 'AAAA': 0.5, 'AAAC': 0.5, 'Cov1': 10 },
                            { 'Contig': 'contig_2', 'AAAA': 1.0, 'AAAC': 0.0, 'Cov1': 20 },
                            { 'Contig': 'contig_3', 'AAAA': 0.7, 'AAAC': 0.3, 'Cov1': 5 } ] )

        if save_file:
            df.to_csv(save_file, sep='\t', index=False)
            self.temp_file_buffer.append(save_file)

        return df

    def test_read_and_validate_table(self):

        mock_table = self.spawn_mock_table(save_file='mock.txt')

        df = project_tsne.read_and_validate_table('mock.txt', 'Cov')

        ''' Test all columns are preserved, and then the content of the first row. '''
        for col in mock_table.columns:
            self.assertIn(col, df.columns)

        exp_row = mock_table.iloc[0,:]
        obs_row = df.iloc[0,:]

        for col in mock_table.columns:
            self.assertEqual( exp_row[col], obs_row[col] )

    def test_read_and_validate_table_notexist(self):

        self.start_logging_stdout()
        df = project_tsne.read_and_validate_table('does_not_exist.txt', 'Cov')
        print_capture = self.stop_logging_stdout()

        self.assertIsNone(df)
        self.assertIn('Warning: Unable to detect feature table does_not_exist.txt, aborting...', print_capture)

    @unittest.skip('Not implemented yet')
    def test_read_and_validate_table_missing_col(self):
        pass
    """
    def read_and_validate_table(ftable_name, cov_prefix):

        ftable = pd.read_csv(ftable_name, sep='\t')

        # Assume there must be able least one coverage column
        ValidateDataFrameColumns(ftable, ['Contig', '{}1'.format(cov_prefix) ])

        return ftable
    """

    def test_parse_and_validate_weighting(self):

        x = project_tsne.parse_and_validate_weighting(0.5)

        self.assertEqual(x, 0.5)

    def test_parse_and_validate_weighting_notnumeric(self):

        ''' This is really a test of part of the functionality under the OptionValidator.ValidateFloat function.
            As invoked in project_tsne.py, it is being called in the form:

            ValidateFloat(userChoice=weight_value, parameterNameWarning='coverage weighting', behaviour='abort')
        '''
        self.start_logging_stdout()
        project_tsne.parse_and_validate_weighting('abc')
        print_capture = self.stop_logging_stdout()

        self.assertIn('Unable to accept value abc for coverage weighting, aborting...', print_capture)

    def test_parse_and_validate_weighting_high(self):

        self.start_logging_stdout()
        project_tsne.parse_and_validate_weighting(1.1)
        print_capture = self.stop_logging_stdout()

        self.assertIn('Error: Trying to weight coverage for more than 100% of data, setting to 1.0.', print_capture)

    def test_parse_and_validate_weighting_low(self):

        self.start_logging_stdout()
        project_tsne.parse_and_validate_weighting(-1.0)
        print_capture = self.stop_logging_stdout()

        self.assertIn('Error: Trying to weight coverage for less than 0% of data, setting to uniform.', print_capture)

    # endregion

if __name__ == '__main__':

    ''' Import the project_tsne.py library '''
    sys.path.insert(0,'..')
    import project_tsne

    unittest.main()