'''

'''
# Basic imports
import sys, os, glob
import numpy as np
import pandas as pd
from optparse import OptionParser

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

        buscoDB, tblastnFile, shortSummaryFile = ValidateBuscoFolder( buscoFolder )

        if buscoDB:
            blastTable = ImportBlastTable(tblastnFile, options.ali_length, options.rank)
            scoreDict[ buscoDB ] = blastTable[options.rank]

    if len(scoreDict) > 0:

        bestBuscoResults = IdentifyTopBuscos(scoreDict, options.rank)
        PlotMeasures(scoreDict, options.output, bestBuscoResults)

    else: print('There were no valid BUSCO folders provided. No results generated.')
###############################################################################

# region Input handling

def ValidateBuscoFolder(buscoPath):

    ''' Not a full validation, just checking the files I need are present '''
    buscoDB = _extractDatabaseName(buscoPath)

    ''' Check that the required files exist, return None if they don't, which will trigger a skip '''
    blastFolder = os.path.join(buscoPath, 'blast_output')
    blastFolder = ValidateFolder(inFile=blastFolder, behaviour='callback', _callback=_fileCallback, fType='BLAST folder')

    tblastnFile = GetBuscoFile(blastFolder, 'tblastn_BUSCO.*.{}.tsv'.format(buscoDB))
    tblastnFile = ValidateFile(inFile=tblastnFile, behaviour='callback', _callback=_fileCallback, fType='tBLASTn file')

    shortSummaryFile = GetBuscoFile(buscoPath, 'short_summary_BUSCO.*.{}.txt'.format(buscoDB))
    shortSummaryFile = ValidateFile(inFile=shortSummaryFile, behaviour='callback', _callback=_fileCallback, fType='summary file')

    if blastFolder and tblastnFile and shortSummaryFile:
        return buscoDB, tblastnFile, shortSummaryFile
    else:
        return None, None, None

def GetBuscoFile(folder, fileString):

    sPath = os.path.join(folder, fileString)
    return glob.glob(sPath)[0]

def _fileCallback(kwargs):

    print( 'Warning: Unable to detect {}, skipping....'.format(kwargs['fType']) )
    return None

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
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
        
        if k in topBuscos:    
            pc.set_alpha(1.0)
        else:   
            pc.set_alpha(0.2)

    for partname in ['cbars','cmins','cmaxes', 'cmedians']:
        vPlot[partname].set_edgecolor('black')
        vPlot[partname].set_linewidth(0.5)

    ''' https://matplotlib.org/1.5.0/examples/statistics/boxplot_vs_violin_demo.html '''
    ax = plt.gca()
    plt.setp(ax, yticks=[ x for x in range(0, 101, 10) ], ylabel='Identity (%)', xlabel='BUSCO collection',
             xticks=[ y+1 for y in range(len(keySort)) ], xticklabels=keySort)
    plt.xticks(rotation=90)

    outputFile = '{}.violinplot.png'.format(outputPath)
    plt.savefig(outputFile, bbox_inches='tight')

###############################################################################
if __name__ == '__main__':
     main()