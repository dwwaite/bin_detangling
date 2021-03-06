import sys, os
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from optparse import OptionParser
from sklearn import preprocessing

# My functions and classes
from scripts.ThreadManager import ThreadManager
from scripts.OptionValidator import ValidateFile, ValidateStringParameter, ValidateInteger
#from scripts.SequenceManipulation import IndexFastaFile

def main():

    ''' Set up the options '''
    usageString = "usage: %prog [options] [contig file]"
    parser = OptionParser(usage=usageString)

    parser.add_option('-k', '--kmer', help='Kmer size for profiling (Default: 4)', dest='kmer', default=4)
    parser.add_option('-t', '--threads', help='Number of threads to use (Default: 1)', dest='threads', default=1)
    parser.add_option('-n', '--normalise', help='Method for normalising per-column values (Options: unit variance (\'unit\'), Yeo-Johnson (\'yeojohnson\'), None (\'none\'). Default: Unit variance)', dest='normalise', default='unit')
    parser.add_option('-o', '--output', help='Output file extension for input files (Default: .tsv)', dest='output', default='.tsv')
    parser.add_option('-r', '--ignore-rev-comp', help='Prevent the use of reverse complement kmers to reduce table size (Default: False)', dest='revComp', action='store_false', default=True)

    parser.add_option('-c', '--coverage', help='Append coverage table to data (Default: None)', dest='coverage', default=None)

    options, inputFiles = parser.parse_args()

    ''' Validate any user-specified variables '''
    if options.coverage: options.coverage = ValidateFile(inFile=options.coverage, behaviour='skip', fileTypeWarning='coverage file', )

    options.normalise = ValidateStringParameter(userChoice=options.normalise,
                                                choiceTypeWarning='coverage file',
                                                allowedOptions=['unit', 'yeojohnson', 'none'],
                                                behaviour='default', defBehaviour='unit')

    options.threads = ValidateInteger(userChoice=options.threads, parameterNameWarning='threads', behaviour='default', defaultValue=1)
    options.kmer = ValidateInteger(userChoice=options.kmer, parameterNameWarning='kmer', behaviour='default', defaultValue=4)

    ''' Generally I'll only run this one file at a time, but option is there for more '''
    for inputFile in inputFiles:

        if not ValidateFile(inFile=inputFile, fileTypeWarning='fasta file', behaviour='skip'):
            continue

        ''' Read in and index the fasta contents '''
        sequenceIndex = IndexFastaFile(inputFile)

        ''' These functions are self-describing, so minimal commenting needed here '''
        kmerDataFrame = ComputeKmerTable(sequenceIndex, options.threads, options.kmer)
        kmerDataFrame = OrderColumns(kmerDataFrame)

        if options.revComp:
            kmerDataFrame = ReverseComplement(kmerDataFrame)

        if options.coverage:
            kmerDataFrame = AppendCoverageTable(kmerDataFrame, options.coverage)

        completeDataFrame = NormaliseColumnValues(kmerDataFrame, options.normalise)

        ''' Reorder the rows to match the input order '''
        completeDataFrame = OrderRows(completeDataFrame, sequenceIndex)
        WriteOutputTable(inputFile, options.output, completeDataFrame)

###############################################################################

#region Fasta and sequence handling

def IndexFastaFile(fileName):

    index = OrderedDict()

    content = open(fileName, 'r').read().split('>')[1:]
    for entry in content:

        seqName, *seqContent = entry.split('\n')
        index[seqName] = ''.join(seqContent)

    return index

def ComputeKmerTable(sequenceDict, nThreads, kSize):

    tManager = ThreadManager(nThreads, CreateKmerRecord)

    ''' Prime a list of arguments to distribute over the threads, then execute '''
    funcArgList = [ (contig, sequence, kSize, tManager.queue) for (contig, sequence) in sequenceDict.items() ]
    tManager.ActivateMonitorPool(sleepTime=15, funcArgs=funcArgList)

    ''' Extract the results, as return as a DataFrame '''
    inputList = tManager.results
    return pd.DataFrame(inputList).fillna(0.0)

def CreateKmerRecord(orderedArgs):

    ''' Unpack the arguments from tuple to variables'''
    try:
        contig, sequence, kSize, q = orderedArgs

        ''' Walk through the sequence as kmer steps and record abundance.
            Omit any results with ambiguous sequences since these are not biological signal '''
        kmerMap = []
        for i in range(0, len(sequence)-kSize+1):
            kmer = sequence[i:i+kSize]
            if not 'N' in kmer: kmerMap.append(kmer)          

        ''' Compress the kmerMap list into a dict, and insert the Contig name '''
        kmerCounter = Counter(kmerMap)
        totKmers = np.sum( [x for x in kmerCounter.values() ] )
        kmerMap = { k: float(i) / totKmers for k, i in kmerCounter.items() }
        kmerMap['Contig'] = contig

        q.put( kmerMap )

    except:
        print( '\tError parsing contig \'{}\', skipping...'.format(contig) )

def ReverseComplement(naiveTable):

    refinedTable = pd.DataFrame()
    refinedTable['Contig'] = naiveTable.Contig

    ''' Instantiate the lookup dict here, so that rebuilt for every kmer in the DataFrame '''
    ntLookup = { 'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C' }
    
    ''' Only consider kmers in the original DataFrame, so there are no empty columns in the end result. '''
    addedKmers = set()

    for kmer in naiveTable.columns[1:]:

        rKmer = _computeSequenceComplement(kmer, ntLookup)

        if rKmer in addedKmers:
            refinedTable[rKmer] += naiveTable[kmer]
        else:
            refinedTable[kmer] = naiveTable[kmer]

        addedKmers.add(kmer)

    return refinedTable

def _computeSequenceComplement(sequence, lookupDict):
    sequence = list(sequence)
    revSequence = [ lookupDict[nt] for nt in sequence[::-1] ]
    return ''.join(revSequence)

#endregion

#region DataFrame handling

def OrderColumns(df):
    colNames = list(df.columns)
    colNames.remove('Contig')
    colNames = sorted(colNames)
    colNames.insert(0, 'Contig')
    return df[colNames]

def NormaliseColumnValues(df, normFactor):

    ''' If there is no normalisation needed, just return the DataFrame unmodified '''
    if normFactor == 'none':
        return df

    ''' Only consider numeric data '''
    colsToTransform = list( df.columns )
    colsToTransform.remove('Contig')

    contigNames = df.Contig

    if normFactor == 'unit':

        ''' Straight out of the preprocessing.scale documentation. '''
        df[colsToTransform] = preprocessing.scale(df[colsToTransform], axis=0)
        return df
    elif normFactor == 'yeojohnson':

        '''
            Taken from https://scikit-learn.org/stable/modules/preprocessing.html
            Since df has already been sorted, can just do an iloc slice of the values.
        '''
        pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
        normArray = pt.fit_transform( df.iloc[ : , 1: ] )

        ''' Recast the data as a DataFrame and return'''
        normDf = pd.DataFrame(normArray, columns=colsToTransform)
        normDf.insert(loc=0, column='Contig', value=contigNames)
        return normDf

def AppendCoverageTable(df, covFile):

    ''' Read in the coverage file, then name first column to Contig, others as Coverage1..CoverageN '''
    covFrame = pd.read_csv(covFile, sep='\t')
    colNames = [ 'Coverage{}'.format(i) for i in range(1, covFrame.shape[1]) ]
    colNames.insert(0, 'Contig')
    covFrame.columns = colNames

    ''' https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html '''
    newFrame = df.merge(covFrame, how='inner', on='Contig')
    return newFrame

def OrderRows(completeDataFrame, sequenceIndex):

    ''' sequenceIndex is an OrderedDict, so extract the keys '''
    df = completeDataFrame.set_index('Contig')
    df = df.loc[ list(sequenceIndex.keys()) ]

    return df.reset_index()

def WriteOutputTable(inputFileName, output_ext, df):

    outputName = os.path.splitext(inputFileName)[0] + output_ext
    df.to_csv(outputName, sep='\t', index=False)

#endregion

###############################################################################
if __name__ == '__main__':
    main()
