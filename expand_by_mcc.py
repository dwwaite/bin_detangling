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

    parser.add_option('-v', '--vizbinTable', help='A vizbin table produced by the vizbin_FilesToTable.py script', dest='vizbinTable')
    parser.add_option('-s', '--slices', help='Number of slices of each bin to take, projected across the asymptotic function [1 - (1 / (1 + s))] Default = 50', dest='slices', default=50)
    parser.add_option('-b', '--bias-threshold', help='The weighting at which contigs are assigned to a bin when fragments appear across multiple bins (Default 0.9)', dest='biasThreshold', default=0.9)
    parser.add_option('-t', '--threads', help='Number of threads', dest='threads', default=1)
    options, binNames = parser.parse_args()

    ''' Validate user choices '''
    options.vizbinTable = ValidateFile(inFile=options.vizbinTable, fileTypeWarning='vizbin table', behaviour='abort')
    options.threads = ValidateInteger(userChoice=options.thread, parameterNameWarning='threads', behaviour='default', defaultValue=1)
    options.biasThreshold = ValidateFloat(userChoice=options.biasThreshold, parameterNameWarning='bias threshold', behaviour='default', defaultValue=0.9)

    ''' Parse the values into a list of per-bin settings '''
    binPrecursors = GenomeBin.ParseStartingVariables(options.vizbinTable, options.slices, binNames)

    if not binPrecursors:
        print('Unable to locate any valid contig lists. Aborting...')
        sys.exit()

    binInstances = [ GenomeBin(bP) for bP in binPrecursors ]

    ''' Distribute the jobs over the threads provided '''
    tManager = ThreadManager(nThreads, RefineAndPlotBin)
    funcArgList = [ (bI, tManager.queue) for bI in binInstances ]
    tManager.ActivateMonitorPool(sleepTime=30, funcArgs=funcArgList, trackingString='Completed MCC growth for {} of {} bins.', totalJobSize=len(funcArgList))

    ''' Parse the results into the contamination record '''    
    contaminationInstanceRecord = ContaminationRecordManager()
    for cR in tManager.results:
        contaminationInstanceRecord.AddRecord(cR)
        
    contaminationInstanceRecord.IndexRecords()
    contaminationInstanceRecord.CalculateContigDistributions(options.vizbinTable)

    ''' Recasting the binInstances list as a dict, so I can access specific bins at will '''
    binInstances = { bI.binIdentifier: bI for bI in binInstances }
    revisedBinInstances = contaminationInstanceRecord.ResolveContaminationByAbundance(binInstances, options.biasThreshold)

    ''' For each bin, write out the core contigs that are trusted at this stage. '''
    for binInstance in revisedBinInstances.values():
        GenomeBin.SaveCoreContigs(binInstance)

###############################################################################

#region Bin refinement functions

def RefineAndPlotBin(argTuple):

    binInstance, q = argTuple

    binInstance.ComputeCloudPurity(q)
    GenomeBin.PlotTrace(binInstance)
    GenomeBin.PlotContours(binInstance)
    #GenomeBin.SaveMccTable(binInstance)

#endregion

###############################################################################
if __name__ == '__main__':
    main()