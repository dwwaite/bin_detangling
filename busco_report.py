'''

'''
# Basic imports
import sys, os, glob
import numpy as np
import pandas as pd
from optparse import OptionParser
from collections import namedtuple

# Standard/science imports
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from scripts.OptionValidator import ValidateFolder, ValidateFile, ValidateStringParameter, ValidateInteger

def main():

    ''' Set up the options '''
    parser = OptionParser()
    parser.add_option('-r', '--rank', help='Metric to rank tBLASTn results (Options: identity, evalue. Default: identity)', dest='rank', default='identity')
    parser.add_option('-o', '--output', help='Name for output files (Default: busco_summary)', dest='output', default='busco_summary')
    parser.add_option('-a', '--ali-threshold', help='Minimum number of amino acids to accept for the alignment length (Default: 50)', dest='ali_length', default=50)

    ''' Parse options, walk through each BUSCO folder '''
    options, buscoFolders = parser.parse_args()
    options.ali_length = ValidateInteger(userChoice=options.ali_length, parameterNameWarning='minimum alignment length', behaviour='default', defaultValue=50)
    options.rank = ValidateStringParameter(userChoice=options.rank, choiceTypeWarning='ranking parameter', allowedOptions=['identity', 'evalue'],
                                           behaviour='default', defBehaviour='identity')
    
    scoreDict = {}

    for buscoFolder in buscoFolders:

        buscoDB, buscoPathObj = ValidateBuscoFolder( buscoFolder )
        blastTable = ImportBlastTable(buscoPathObj.tblastn, options.ali_length, options.rank)

        scoreDict[ buscoDB ] = blastTable[options.rank]
        #print( '{} <==> {}'.format(buscoPathObj.tblastn, buscoPathObj.short_summary) )

    bestBuscoResults = IdentifyTopBuscos(scoreDict, options.rank)
    PlotMeasures(scoreDict, options.output, bestBuscoResults)
###############################################################################

# region Input handling

def ValidateBuscoFolder(buscoPath):

    ''' Not a full validation, just checking the files I need are present
        A bit of a sloppy implementation, just a lot of exit statements when required files are absent '''
    buscoPathObj = namedtuple('buscoPaths', ['tblastn', 'short_summary'])
    blastFolder = os.path.join(buscoPath, 'blast_output')

    buscoDB = _extractDatabaseName(buscoPath)

    ''' Check that the BLAST folder exists, abort if it doesn't and load in the files if it does '''
    ValidateFolder(inFile=blastFolder, fileTypeWarning='BLAST folder', behaviour='abort')
    tblastnFile = glob.glob(blastFolder + '/tblastn_BUSCO.*.{}.tsv'.format(buscoDB) )[0]
    ValidateFile(inFile=tblastnFile, fileTypeWarning='tBLASTn table', behaviour='abort')   
    buscoPathObj.tblastn = tblastnFile

    ''' Check for the summary file '''
    shortSummaryFile = glob.glob( '{}/short_summary_BUSCO.*.{}.txt'.format(buscoPath, buscoDB) )[0]
    ValidateFile(inFile=shortSummaryFile, fileTypeWarning='tBLASTn table', behaviour='abort')   
    buscoPathObj.short_summary = shortSummaryFile

    return buscoDB, buscoPathObj

def _extractDatabaseName(buscoFolder):
    buscoDB =  buscoFolder.split('.')[-1]
    buscoDB = buscoDB[0:-1] if buscoDB[-1] == '/' else buscoDB
    return buscoDB

def ImportBlastTable(blastPath, minAlignmentLength, rankMethod):

    prePandasList = []

    for line in open(blastPath, 'r'):

        if not line[0] == '#':

            dbTarget, binProtein, identity, alignment_length, *other, evalue, bit_score = line.strip().split('\t')

            if int(alignment_length) >= minAlignmentLength:

                prePandasList.append( { 'bin_protein': binProtein,
                                        'db_target': dbTarget,
                                        'identity': float(identity),
                                        'alignment_length': float(alignment_length),
                                        'evalue': float(evalue),
                                        'bit_score': float(bit_score) } )

    ''' Now just get the top hit for each BUSCO target
        Sort the DataFrame by identity, then take the first entry each db_target '''
    df = pd.DataFrame(prePandasList)

    orderMode = True if rankMethod == 'identity' else False
    df_sorted = _sortDfByRankingColumn(df, rankMethod)
    df_top = df_sorted.groupby('db_target').head(1).reset_index(drop=True)
    return df_top

def _sortDfByRankingColumn(df, method):
    orderMode = True if method == 'identity' else False
    return df.sort_values([method], ascending=orderMode)

# endregion

# region Plotting and outputs

def IdentifyTopBuscos(scoreDict, rankMethod):

    preDfList = [ { 'k': k, rankMethod: np.median(vals) } for k, vals in scoreDict.items() ]
    df = pd.DataFrame(preDfList)

    df_sorted = _sortDfByRankingColumn(df, rankMethod)
    topHit = df_sorted.k[0]

    ''' For each subsequent BUSCO, if there is no difference in its values compared to the top hit, add it to the set as an equally likely candidate.
        There is no multiple testing correction here, because this is just an informal screen for plotting purposes. '''
    
    topHits = set()
    topHits.add(topHit)

    for k in df.k[1:]:
        
            t, p = ttest_ind( scoreDict[topHit], scoreDict[k])
            #print( '{} vs {}, t={}, p={}'.format(topHit, k, t, p) )
            if p > 0.05: topHits.add(k)

    return topHits

def PlotMeasures(valueDict, outputPath, topBuscos):

    ''' Sort the BUSCos alphabetically '''
    keySort = sorted( valueDict )

    ''' Plot and customise '''
    vPlot = plt.violinplot([ valueDict[k] for k in keySort ], showmeans=False, showmedians=True)

    for k, pc in zip(keySort, vPlot['bodies']):

        pc.set_facecolor('g')
        
        if k in topBuscos:    
            pc.set_alpha(1.0)
        else:   
            pc.set_alpha(0.2)

    ''' https://matplotlib.org/1.5.0/examples/statistics/boxplot_vs_violin_demo.html '''
    ax = plt.gca()
    plt.setp(ax, yticks=[ x for x in range(0, 101, 10) ], ylabel='Identity (%)', xlabel='BUSCO collection',
             xticks=[ y+1 for y in range(len(keySort)) ], xticklabels=keySort)


    plt.show()
    outputFile = '{}.violinplot.png'.format(outputPath)
    #fig.savefig(outputFile, bbox_inches='tight')

# endregion
"""
  ''' Clear the plotting space '''
    plt.clf()

    ''' Plot the points '''
    colValues = [ colourLookup[x] for x in pcaObj.df.Cluster ]
    plt.scatter(pcaObj.df.PC1, pcaObj.df.PC2, c=colValues)

    ''' For each cluster, add a centroid label '''
    clusterNames = _grabClusterNames(pcaObj.df)
    for clusterName in clusterNames:

        tempdf = pcaObj.df[ pcaObj.df.Cluster == clusterName ]
        x, y = _returnCentroid(tempdf)
        plt.text(x, y, clusterName, fontsize=20)

    _produceAndSavePlot(plt, pcaObj.pc1_label, pcaObj.pc2_label, outputName, 'cluster')
"""
###############################################################################
if __name__ == '__main__':
     main()