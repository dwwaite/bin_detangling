'''
    Automated testing of the creation, insertion, update, and delete functionality of the DatabaseManipulator class
'''
import sys
import io
import unittest
#import os

class TestProjectTsne(unittest.TestCase):

    def setUp(self):
        #@unittest.skip('Not implemented yet')
        pass

    # region Capture stdout for evaluating print() statements

    def start_logging_stdout(self):
            self.print_capture = io.StringIO()
            sys.stdout = self.print_capture

    def stop_logging_stdout(self):
            sys.stdout = sys.__stdout__
            return self.print_capture.getvalue()

    # endregion

    # region Tests for the section Pre-workflow overhead
    @unittest.skip('Not implemented yet')
    def test_read_and_validate_table(self):
        pass

    @unittest.skip('Not implemented yet')
    def test_read_and_validate_table_notexist(self):
        pass

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

        """
        def ValidateFloat(userChoice, parameterNameWarning, behaviour, defaultValue=None):

            try:

                f = float(userChoice)
                return f

            except:

                if behaviour == 'default':

                    print( 'Unable to accept value {} for {}, using default ({}) instead.'.format(userChoice, parameterNameWarning, defaultValue) )
                    return defaultValue

                if behaviour == 'abort':

                    print( 'Unable to accept value {} for {}, aborting...'.format(userChoice, parameterNameWarning, defaultValue) )
                    sys.exit()
        """
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