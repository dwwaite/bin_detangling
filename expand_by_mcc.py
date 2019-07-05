'''
    A refactored version of the original script, so make use of multithreading

    Debug line
    cd C:/Users/dwait/bin_detangling
    python expand_by_mcc.py -e tests/mock.table.txt -o tests/debug -
'''

import sys, os
import pandas as pd
import numpy as np
from optparse import OptionParser

# My functions and classes
from scripts.ThreadManager import ThreadManager
from scripts.GenomeBin import GenomeBin
from scripts.OptionValidator import ValidateFile, ValidateInteger, ValidateFloat, ValidateDataFrameColumns

def main():
    
    ''' Process the user input via the OptionParser library '''
    options, bin_names = process_user_input()

    ''' Validate user choices '''
    options.esomTable = ValidateFile(inFile=options.esomTable, fileTypeWarning='ESOM table', behaviour='abort')
    ValidateDataFrameColumns( pd.read_csv(options.esomTable, sep='\t'), ['V1', 'V2', 'BinID', 'ContigName', 'ContigBase'])

    options.threads = ValidateInteger(userChoice=options.threads, parameterNameWarning='threads', behaviour='default', defaultValue=1)
    options.slices = ValidateInteger(userChoice=options.slices, parameterNameWarning='slices', behaviour='default', defaultValue=50)
    options.bias_threshold = ValidateFloat(userChoice=options.bias_threshold, parameterNameWarning='bias threshold', behaviour='default', defaultValue=0.9)

    ''' Parse the values into a list of per-bin settings
        Check that bins specified are actually correct, and if a default value is given, overwrite the list '''
    bin_precursors = validate_input_data(bin_names, options.esomTable, options.slices, options.bias_threshold, options.output)

    fragment_counts = count_all_fragments(options.esomTable)  

    ''' Distribute the jobs over the threads provided
        Because this mapped function stores results, not distributed jobs, I can't count the progress using the optional parameters of the ThreadManager '''
    tManager = ThreadManager(options.threads, RefineAndPlotBin)
    
    funcArgList = [ (bP, fragment_counts, tManager.queue) for bP in bin_precursors ]
    tManager.ActivateMonitorPool(sleepTime=10, funcArgs=funcArgList)

    ''' Resolve the final bin memberships into an output table '''
    GenomeBin.CreateCoreTable(options.esomTable, tManager.results )

###############################################################################

# region Input, parsing, and validation functions

def process_user_input():

    ''' Create and utilise the OptionParser '''
    usageString = "usage: %prog [options] [bin names]"
    parser = OptionParser(usage=usageString)

    parser.add_option('-e', '--esom-table', help='A table produced by the vizbin_files_to_table.py script', dest='esomTable')
    parser.add_option('-o', '--output', help='An output prefix for all generated files (Default: None)', dest='output', default=None)
    parser.add_option('-s', '--slices', help='Number of slices of each bin to take (Default: 50)', dest='slices', default=50)
    parser.add_option('-b', '--bias-threshold', help='The weighting at which contigs are assigned to a bin when fragments appear across multiple bins (Default: 0.9)', dest='bias_threshold', default=0.9)
    parser.add_option('-t', '--threads', help='Number of threads', dest='threads', default=1)

    return  parser.parse_args()

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

# endregion

#region Bin refinement functions

def RefineAndPlotBin(argTuple):

    '''
        2019/03/11 - As a future point, it might make sense to rewrite ComputeCloudPurity as a static function of GenomeBin,
                     with specific functions for the steps within the MCC expansion.
                     This change would be cosmetic only, so remains TODO.
    '''
    (bin_name, esom_path, bias_threshold, number_of_slices, output_path), fragment_count_dict, q = argTuple

    bin_instance = GenomeBin(bin_name, esom_path, bias_threshold, number_of_slices, output_path)

    bin_instance.ComputeCloudPurity()
    bin_instance.ResolveUnstableContigs(fragment_count_dict, q)

    bin_instance.PlotTrace()
    bin_instance.PlotScatter()
    bin_instance.SaveMccTable()

def cast_bin_list_to_dict(bin_instance_list):

    bin_instance_dict = {}
    for bI in bin_instance_list:
        bin_instance_dict[ bI.bin_name ] = bI
        bin_instance_dict[ bI.bin_name ].build_contig_base_set()

    return bin_instance_dict

def count_all_fragments(esom_table_name):

    return { contig: df.shape[0] for contig, df in pd.read_csv(esom_table_name, sep='\t').groupby('ContigBase') }

#endregion

###############################################################################
if __name__ == '__main__':
    main()