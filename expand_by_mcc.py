'''
    A refactored version of the original script, so make use of multithreading

    Debug line
    cd C:/Users/dwai012/Documents/Genomics Aoteoroa/Pipeline Development/ESOM Bin detangler
    python expand_by_mcc.py -e tests/mock.table.txt -o tests/mock.table Bin1 Bin3
'''

import sys, os
import pandas as pd
import numpy as np
from optparse import OptionParser
from collections import namedtuple

# My functions and classes
from scripts.ThreadManager import ThreadManager
from scripts.GenomeBin import GenomeBin, ContaminationRecordManager, ContaminationRecord
from scripts.OptionValidator import ValidateFile, ValidateInteger, ValidateFloat

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
    options.threads = ValidateInteger(userChoice=options.threads, parameterNameWarning='threads', behaviour='default', defaultValue=1)
    options.slices = ValidateInteger(userChoice=options.slices, parameterNameWarning='slices', behaviour='default', defaultValue=50)
    options.biasThreshold = ValidateFloat(userChoice=options.biasThreshold, parameterNameWarning='bias threshold', behaviour='default', defaultValue=0.9)

    ''' Parse the values into a list of per-bin settings '''
    binPrecursors = GenomeBin.ParseStartingVariables(options.esomTable, options.slices, binNames, options.output)

    if not binPrecursors:
        print('Unable to locate any valid contig lists. Aborting...')
        sys.exit()

    ''' Instantiate the bin objects. Failed constructor returns None, so filter these out '''
    binInstances = [ GenomeBin(bP) for bP in binPrecursors ]
    binInstances = [ b for b in binInstances if b]

    ''' Distribute the jobs over the threads provided '''
    tManager = ThreadManager(options.threads, RefineAndPlotBin)
    
    funcArgList = [ (bP, tManager.queue) for bP in binInstances ]
    #tManager.ActivateMonitorPool(sleepTime=30, funcArgs=funcArgList, trackingString='Completed MCC growth for {} of {} bins.', totalJobSize=len(funcArgList))
    tManager.ActivateMonitorPool(sleepTime=10, funcArgs=funcArgList)

    ''' Parse the results into the contamination record '''
    binInstances, contaminationInstanceRecord = ExtractQueuedResults(tManager.results)
    contaminationInstanceRecord.IndexRecords()
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
    binInstance, q = argTuple

    try:

        binInstance.ComputeCloudPurity(q)

        GenomeBin.PlotTrace(binInstance)
        GenomeBin.PlotContours(binInstance)
        #GenomeBin.SaveMccTable(binInstance)

    except:
        print( 'Error processing bin {}, skipping...'.format(binInstance.binIdentifier) )

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