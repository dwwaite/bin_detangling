import sys, os
import pandas as pd
import numpy as np
from collections import Counter
from optparse import OptionParser
from sklearn import preprocessing
from scripts.ThreadManager import ThreadManager

def main():

    ''' Set up the options '''
    parser = OptionParser()
    parser.add_option('-k', '--kmer', help='Kmer size for profiling (Default: 4)', dest='kmer', default=4)
    parser.add_option('-t', '--threads', help='Number of threads to use (Default: 1)', dest='threads', default=1)
    parser.add_option('-n', '--normalise', help='Method for normalising per-column values (Options: unit variance (\'unit\'), Yeo-Johnson (\'yeojohnson\'), None (\'none\'). Default: Unit variance)', dest='normalise', default='unit')
    parser.add_option('-r', '--ignore-rev-comp', help='Prevent the use of reverse complement kmers to reduce table size (Default: False)', dest='revComp', action='store_false', default=True)

    ''' Options for making this ESOM-ready'''
    parser.add_option('-c', '--coverage', help='Append coverage table to data (Default: None)', dest='coverage', default=None)
    parser.add_option('--to-esomana', help='Add ESOMana prefix to the output table (Default: False)', dest='esomana', action='store_true', default=False)

    options, inputFiles = parser.parse_args()

    ''' Validate any user-specified variables '''
    if options.coverage: options.coverage = Validatecoverage(options.coverage)
    options.normalise = ValidateNormalisation(options.normalise)
    options.threads = ValidateNumeric(options.threads, 'threads', 1)
    options.kmer = ValidateNumeric(options.kmer, 'kmer', 4)

    ''' Generally I'll only run this one file at a time, but option is there for more '''
    for inputFile in inputFiles:

        ''' Check it exists... '''
        if not os.path.isfile(inputFile):
            print( 'Unable to open file {}, skipping...'.format(inputFile) )
            continue

        ''' Read in and index the fasta contents '''
        sequenceIndex = IndexFastaFile(inputFile)

        ''' These functions are self-describing, so minimal commenting needed here '''
        kmerFrequencyTable = ComputeKmerTable(sequenceIndex, options.threads, options.kmer)
        contigNames, kmerDataFrame = OrderColumns(kmerFrequencyTable)

        if options.revComp:
            kmerDataFrame = ReverseComplement(kmerDataFrame)

        completeDataFrame = BuildFinalOutput(kmerDataFrame, contigNames)

        if options.coverage:
            completeDataFrame = AppendCoverageTable(completeDataFrame, options.coverage)

        completeDataFrame = NormaliseColumnValues(completeDataFrame, options.normalise)
        WriteOutputTable(inputFile, completeDataFrame, options.esomana)

###############################################################################

#region Validation functions, to check inputs

def Validatecoverage(covFile):
    if os.path.isfile(covFile):
        return covFile
    else:
        print( 'Warning: Unable to detect coverage file {}, skipping....'.format(covFile) )
        return None

def ValidateNormalisation(normChoice):

    validOptions = set( ['unit', 'yeojohnson', 'none'] )

    if normChoice.lower() in validOptions:
        return normChoice.lower()

    else:
        print( 'Warning: Unable to parse normalisaton choice {}, using unit variance instad.'.format(normChoice) )
        return 'unit'

def ValidateNumeric(submittedValue, parameterName, defaultValue):
    try:
        i = int(submittedValue)
        return i
    except:
        print( 'Unable to accept value {} for parameter {}, using default ({}) instead.'.format(submittedValue, parameterName, defaultValue) )
        return defaultValue

#endregion

#region Fasta and sequence handling

def IndexFastaFile(fileName):
    content = open(fileName, 'r').read()
    content = content.split('>')[1:]
    index = {}
    for entry in content:
        entry = entry.split('\n')
        contig = entry[0]
        sequence = ''.join(entry[1:])
        index[contig] = sequence
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
        print( '\tError parsing conig \'{}\', skipping...'.format(contig) )

def ReverseComplement(naiveTable):

    refinedTable = pd.DataFrame()

    ''' Instantiate the lookup dict here, so that it's not instantiated for every kmer in the DataFrame '''
    ntLookup = { 'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C' }
    
    ''' Only consider kmers in the original DataFrame
        This means we don't end up with empty columns in the end result. '''
    addedKmers = set()
    for kmer in naiveTable.columns:
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
    colValues = df.loc[:,'Contig']
    colNames.remove('Contig')
    return colValues, df[sorted(colNames)]

def NormaliseColumnValues(df, normFactor):

    ''' If there is no normalisation needed, just return the DataFrame unmodified '''
    if normFactor == 'none':
        return df

    ''' Only consider numeric data '''
    colsToTransform = list( df.columns )
    colsToTransform.remove('Contig')

    if normFactor == 'unit':

        ''' Straight out of the preprocessing.scale documentation. '''
        df[colsToTransform] = preprocessing.scale(df[colsToTransform], axis=0)
        return df
    elif normFactor == 'yeojohnson':

        ''' Taken from https://scikit-learn.org/stable/modules/preprocessing.html
            Since df has already been sorted, can just do an iloc slice of the values.
            Fits and transforms for each feature (column) '''
        pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
        normArray = pt.fit_transform( df.iloc[ : , 1: ] )

        ''' Recast the data as a DataFrame and return'''
        normDf = pd.DataFrame(normArray, columns=colsToTransform)
        return BuildFinalOutput(normDf, df['Contig'])
    else:
        ''' Otherwise, none must have been chosen, return the frame unchanged. '''
        return df

def BuildFinalOutput(df, contigNames):
    df.insert(loc=0, column='Contig', value=contigNames)
    return df

def AppendCoverageTable(df, covFile):

    ''' Read in the coverage file, then name first column to Contig, others as Coverage1..CoverageN '''
    covFrame = pd.read_csv(covFile, sep='\t')
    colNames = [ 'Coverage{}'.format(i) for i in range(1, covFrame.shape[1]) ]
    colNames.insert(0, 'Contig')
    covFrame.columns = colNames

    ''' https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html '''
    newFrame = df.merge(covFrame, how='inner', on='Contig')
    return newFrame

def WriteOutputTable(inputFileName, df, esomanaFlag):

    outputName = os.path.splitext(inputFileName)[0] + '.tsv'
    df.to_csv(outputName, sep='\t', index=False)

    ''' The trick to appending header infomation came from https://stackoverflow.com/questions/29233496/write-comments-in-csv-file-with-pandas '''
    if esomanaFlag:

        n, m = df.shape

        ''' Write the names file '''
        outClass = os.path.splitext(inputFileName)[0] + '.names'
        outputWriterClass = open(outClass, 'w')

        outputWriterClass.write( '%{}\n'.format(n) )
        for i, c in zip( range(0, n), df.Contig ):
            outputWriterClass.write( '{}\t{}\n'.format(i, c) )

        outputWriterClass.close()

        ''' Open a stream for the lrn file '''
        outputName = os.path.splitext(inputFileName)[0] + '.lrn'
        outputWriter = open(outputName, 'w')

        ''' Write the header for the lrn file '''
        outputWriter.write( '%{}\n'.format(n) )
        outputWriter.write( '%{}\n'.format(m) )

        colIndentifiers = [ '1' for x in range(1, m) ]
        colIndentifiers.insert(0, '9')
        outputWriter.write( '%{}\n'.format( '\t'.join(colIndentifiers) ) )
        outputWriter.write( '%' )

        ''' Write out the content of the df file '''
        df.Contig = [ i for i in range(0, n) ]
        df.to_csv(outputWriter, sep='\t', index=False)
        outputWriter.close()

#endregion

###############################################################################
if __name__ == '__main__':
    main()