'''
    A refactored version of the original script, so make use of multithreading

    Note that it's trival to update the header of the points output of the gapstatCluster.py script
        by simply changing the Cluster column name to BinID (and renaming the old BinID column)
    Debug line
    cd C:/Users/dwai012/Documents/Genomics Aoteoroa/Pipeline Development/ESOM Bin detangler
    python vizbin_MCCgrow_v3.py -v debug_clusterv3/test.n11.points.txt Cluster_1 Cluster_2 Cluster_3 Cluster_4 Cluster_5 Cluster_6 Cluster_7 Cluster_8 Cluster_9 Cluster_10 Cluster_11
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
    parser = OptionParser()
    usage = "usage: %prog [options] [starter-bin fasta files]"

    parser.add_option('-e', '--esom-table', help='A vizbin table produced by the vizbin_FilesToTable.py script', dest='esomTable')
    parser.add_option('-s', '--slices', help='Number of slices of each bin to take, projected across the asymptotic function [1 - (1 / (1 + s))] Default = 50', dest='slices', default=50)
    parser.add_option('-b', '--bias-threshold', help='The weighting at which contigs are assigned to a bin when fragments appear across multiple bins (Default 0.9)', dest='biasThreshold', default=0.9)
    parser.add_option('-t', '--threads', help='Number of threads', dest='threads', default=1)
    options, binNames = parser.parse_args()

    ''' Validate user choices '''
    options.esomTable = ValidateFile(inFile=options.esomTable, fileTypeWarning='ESOM table', behaviour='abort')
    options.threads = ValidateInteger(userChoice=options.threads, parameterNameWarning='threads', behaviour='default', defaultValue=1)
    options.slices = ValidateInteger(userChoice=options.slices, parameterNameWarning='slices', behaviour='default', defaultValue=50)
    options.biasThreshold = ValidateFloat(userChoice=options.biasThreshold, parameterNameWarning='bias threshold', behaviour='default', defaultValue=0.9)

    ''' Parse the values into a list of per-bin settings '''
    binPrecursors = GenomeBin.ParseStartingVariables(options.esomTable, options.slices, binNames)

    if not binPrecursors:
        print('Unable to locate any valid contig lists. Aborting...')
        sys.exit()

    ''' Instantiate the bin objects '''
    binInstances = [ GenomeBin(bP) for bP in binPrecursors ]

    ''' Distribute the jobs over the threads provided '''
    tManager = ThreadManager(options.threads, RefineAndPlotBin)
    
    funcArgList = [ (bP, tManager.queue) for bP in binInstances ]
    #tManager.ActivateMonitorPool(sleepTime=30, funcArgs=funcArgList, trackingString='Completed MCC growth for {} of {} bins.', totalJobSize=len(funcArgList))
    tManager.ActivateMonitorPool(sleepTime=10, funcArgs=funcArgList)

    ''' Parse the results into the contamination record '''
    binInstances, contaminationInstanceRecord = ExtractQueuedResults(tManager.results)
    contaminationInstanceRecord.IndexRecords()
    contaminationInstanceRecord.CalculateContigDistributions(options.esomTable)

    print(len(binInstances))

    ''' Recasting the binInstances list as a dict, so I can access specific bins at will '''
    binInstances = { bI.binIdentifier: bI for bI in binInstances }
    revisedBinInstances = contaminationInstanceRecord.ResolveContaminationByAbundance(binInstances, options.biasThreshold)

    ''' For each bin, write out the core contigs that are trusted at this stage. '''
    for binInstance in revisedBinInstances.values():
        print( binInstance.to_string() )
        GenomeBin.SaveCoreContigs(binInstance)

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