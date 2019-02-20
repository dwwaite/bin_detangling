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
from GenomeBin import GenomeBin, ContaminationRecordManager, ContaminationRecord
from multiprocessing import Pool, Queue, Manager

def main():
    
    # Parse options
    parser = OptionParser()
    usage = "usage: %prog [options] [starter-bin fasta files]"

    parser.add_option('-v', '--vizbinTable', help='A vizbin table produced by the vizbin_FilesToTable.py script', dest='vizbinTable')
    parser.add_option('-s', '--slices', help='Number of slices of each bin to take, projected across the asymptotic function [1 - (1 / (1 + s))] Default = 50', dest='slices', default=50)
    parser.add_option('-b', '--bias-threshold', help='The weighting at which contigs are assigned to a bin when fragments appear across multiple bins (Default 0.9)', dest='biasThreshold', default=0.9)
    parser.add_option('-t', '--threads', help='Number of threads', dest='threads', default=1)
    options, binNames = parser.parse_args()

    # Pre-multithreading workflow
    ''' Parse the input files - the GenomeBin function returns None if it can't parse anything.
        Finally, insantiate a container for all contamination occurrances. '''

    ValidateVizBin(options.vizbinTable)
    ValidateThreads(options.threads) # TO DO
    options.biasThreshold = ValidateBias(options.biasThreshold)

    queueManager, poolManager = InstantiateMultithreadingParameters(2)

    binPrecursors = GenomeBin.ParseStartingVariables(options.vizbinTable, options.slices, binNames)
    if not binPrecursors: TerminateScript('Unable to locate any valid contig lists. Aborting...')
    binInstances = [ GenomeBin(bP) for bP in binPrecursors ]

    ''' This is where multithreading will come in later one '''

    for binInstance in binInstances:

        print( 'Calculating scores for {}'.format(binInstance.binIdentifier) )
        binInstance.ComputeCloudPurity(queueManager) # this will be refactored on mp rewrite.
        GenomeBin.PlotTrace(binInstance)
        GenomeBin.PlotContours(binInstance)
        #GenomeBin.SaveMccTable(binInstance)

    ''' Ready for detangling stage.
        Start by draining out the queueManager of all results ''' 
    contaminationInstanceRecord = ContaminationRecordManager()
    for cR in ExtractQueueResults(queueManager):
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

#region Validation functions
#
# Functions to confirm all input data is correct, and terminate script execution if not.
#
def ValidateVizBin(vbTable):
    if not vbTable: TerminateScript('Unable to proceed without a vizbin table.')
    if not os.path.isfile(vbTable): TerminateScript('Unable to open vizbin table.')

def ValidateThreads(nThreads):
    return True

def ValidateBias(biasThreshold):
    try:
        biasThreshold = float(biasThreshold)
    except:
        TerminateScript('Unable to parse bias threshold to a decimal value.')
    
    if biasThreshold > 1.0: biasThreshold /=  100 
    return biasThreshold

def TerminateScript(oMsg):
    print(oMsg)
    sys.exit()
#endregion

#region multithreading management functions
def InstantiateMultithreadingParameters(nThreads):
    q = Manager().Queue()
    p = Pool(nThreads)
    return q, p

def ExtractQueueResults(q):
    yield q.get()
    while not q.empty():
        yield q.get(True)


#endregion

###############################################################################
if __name__ == '__main__':
    main()