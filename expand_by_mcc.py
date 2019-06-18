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
from scripts.GenomeBin import GenomeBin, ContaminationRecordManager, ContaminationRecord
from scripts.OptionValidator import ValidateFile, ValidateInteger, ValidateFloat, ValidateDataFrameColumns

def main():
    
    # Parse options
    usageString = "usage: %prog [options] [bin names]"
    parser = OptionParser(usage=usageString)

    parser.add_option('-e', '--esom-table', help='A table produced by the vizbin_files_to_table.py script', dest='esomTable')
    parser.add_option('-o', '--output', help='An output prefix for all generated files (Default: None)', dest='output', default=None)
    parser.add_option('-s', '--slices', help='Number of slices of each bin to take (Default: 50)', dest='slices', default=50)
    parser.add_option('-b', '--bias-threshold', help='The weighting at which contigs are assigned to a bin when fragments appear across multiple bins (Default: 0.9)', dest='biasThreshold', default=0.9)
    parser.add_option('-t', '--threads', help='Number of threads', dest='threads', default=1)
    options, binNames = parser.parse_args()

    ''' Validate user choices '''
    options.esomTable = ValidateFile(inFile=options.esomTable, fileTypeWarning='ESOM table', behaviour='abort')
    ValidateDataFrameColumns( pd.read_csv(options.esomTable, sep='\t'), ['V1', 'V2', 'BinID', 'ContigName', 'ContigBase'])

    options.threads = ValidateInteger(userChoice=options.threads, parameterNameWarning='threads', behaviour='default', defaultValue=1)
    options.slices = ValidateInteger(userChoice=options.slices, parameterNameWarning='slices', behaviour='default', defaultValue=50)
    options.biasThreshold = ValidateFloat(userChoice=options.biasThreshold, parameterNameWarning='bias threshold', behaviour='default', defaultValue=0.9)

    ''' Parse the values into a list of per-bin settings
        Check that bins specified are actually correct, and if a default value is given, overwrite the list '''
    if len(binNames) == 1 and binNames[0] == '-':
        binNames = list( pd.read_csv(options.esomTable, sep='\t').BinID.unique() )

    binPrecursors = GenomeBin.ParseStartingVariables(options.esomTable, options.slices, binNames, options.output)
    if len(binPrecursors) == 0:
        print('Unable to locate any valid contig lists. Aborting...')
        sys.exit()

    #
    # Debugging - working without threading to keep errors on main process
    #
    from multiprocessing import Queue, Manager
    q = Manager().Queue()
    for bP in binPrecursors:
        RefineAndPlotBin( (bP, q) )

    _results = []
    while not q.empty():
        _results.append( q.get(True) )

    #
    # DEBUGGING - UP TO HERE
    #   SEEM TO BE RETURNING NO CONTAM CONTIGS
    #
    print(_results)

    #
    # Debugging - commented out for now
    #
    ''' Distribute the jobs over the threads provided '''
    '''
    tManager = ThreadManager(options.threads, RefineAndPlotBin)
    
    funcArgList = [ (bP, tManager.queue) for bP in binPrecursors ]
    #tManager.ActivateMonitorPool(sleepTime=30, funcArgs=funcArgList, trackingString='Completed MCC growth for {} of {} bins.', totalJobSize=len(funcArgList))
    tManager.ActivateMonitorPool(sleepTime=10, funcArgs=funcArgList)

    '''
    ''' Parse the results into the contamination record '''
    #binInstances, contaminationInstanceRecord = ExtractQueuedResults(tManager.results)
    binInstances, contaminationInstanceRecord = ExtractQueuedResults(_results)

    contaminationInstanceRecord.IndexRecords()

    print( contaminationInstanceRecord._recordFrame.head() )
    sys.exit()

    contaminationInstanceRecord.CalculateContigDistributions(options.esomTable)

    ''' Recasting the binInstances list as a dict, so I can access specific bins at will '''
    binInstances = { bI.binIdentifier: bI for bI in binInstances }
    revisedBinInstances = contaminationInstanceRecord.ResolveContaminationByAbundance(binInstances, options.biasThreshold)

    #''' For each bin, write out the core contigs that are trusted at this stage. '''
    #for binInstance in revisedBinInstances.values():
    #    GenomeBin.SaveCoreContigs(binInstance)
    ''' Write a table of the core contigs for each bin. '''
    coreTable = GenomeBin.CreateCoreTable( revisedBinInstances.values() )
    coreTable.to_csv( options.output + '.core_table.txt', sep='\t', index=False)

###############################################################################

#region Bin refinement functions

def RefineAndPlotBin(argTuple):

    '''
        2019/03/11 - As a future point, it would make sense to rewrite ComputeCloudPurity as a static function of GenomeBin,
                     with specific functions for the steps within the MCC expansion.
                     This change would be cosmetic only, so remains TODO.
    '''
    (bin_name, esom_path, number_of_slices, output_path), q = argTuple

    bin_instance = GenomeBin(bin_name, esom_path, number_of_slices, output_path)

    bin_instance.ComputeCloudPurity(q)

    bin_instance.PlotTrace()
    bin_instance.PlotContours()
    #GenomeBin.SaveMccTable(bin_instance)

def ExtractQueuedResults(resultsQueue):

    cIR = ContaminationRecordManager()
    updatedBinList = []

    '''
        The Queue object contains tuples of (bool, object), where the bool refers to whether the object is a GenomeBin or not.
        This is used to determine which result container the object is stored in.
    '''
    for isBin, obj in resultsQueue:

        if isBin:
            updatedBinList.append(obj)
        else:
            cIR.AddRecord(obj)

    return updatedBinList, cIR

#endregion

###############################################################################
if __name__ == '__main__':
    main()