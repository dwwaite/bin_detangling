'''
    Options supported by sklearn which may be appropriate are:
        1. K-means clustering (distance between points; produces equal-abundance clusters which is not appropriate)
        2. Affinity propagation (graph distance)
        3. Mean-shift (distance between points; can't specify numbers of clusters)
        4. DBSCAN (distance between points; can't specify numbers of clusters)
        5. Birch (distance between points; uneven clusters allowed)

    Find them at https://scikit-learn.org/stable/modules/clustering.html

    Debug line:
    python gapstat_cluster.py -t 2 --min 2 --max 10 -o mock_cluster.pca --convex --plot tests/kmer.input.chomp1500.tsv
    python gapstat_cluster.py -t 2 --min 2 --max 10 -o mock_cluster.tsne --esom-ordinate --convex --plot tests/kmer.input.chomp1500.tsv
'''
# Basic imports
import sys, os
import numpy as np
import pandas as pd
from optparse import OptionParser
from collections import namedtuple

# Standard/science imports
import matplotlib.pyplot as plt
import matplotlib.colors as cmap
from scipy.spatial import ConvexHull
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Custom imports
from gap_statistic import OptimalK
from scripts.OptionValidator import ValidateInteger


def main():

    usageString = "usage: %prog [options] [feature table]"
    parser = OptionParser(usage=usageString)

    ''' Parse and validate options - simple/expected choices '''
    parser.add_option('-t', '--threads', help='Number of threads to use (Default = 1)', dest='threads')
    parser.add_option('--min', help='Minimum number of clusters to test (Default = 2)', dest='min', default=2)
    parser.add_option('--max', help='Maximum number of clusters to test (Default = 200)', dest='max', default=200)
    parser.add_option('--force-cluster', help='Skip gap-stat evaluation and cluster into N clusters', dest='force_cluster', default=None)
    parser.add_option('--sep', help='Separator value to delimit columns (Default = [tab])', dest='sep', default='\t')
    parser.add_option('-o', '--output', help='Name stub for all output files (Default: Inherited from input table)', dest='output_file')
    parser.add_option('--esom-ordinate', help='Use the tSNE algorithm to create ordination and reduce dimensionality of data (Default: None, PCA for plotting)', dest='esom_ordinate', action='store_true')
    parser.add_option('--esom-dimensions', help='Number of dimensions to project the tSNE ordination. If greater than 2 is specified and plotting requested, only the first 2 axes will be plotted (Default: 2).', dest='esom_dimensions', default=2)
    parser.add_option('--log-contigs', help='Create per-cluster files containing contig names (Default: False)', dest='log_contigs', action='store_true')
    parser.add_option('-p', '--plot', help='Plot the decision process (Default: False, but highly recommended)', dest='plot', action='store_true')
    parser.add_option('--convex', help='Plot convex hulls of the bin outlines. Useful for deciding whether or not to merge bins (Default: False)', dest='convex', action='store_true')

    options, arguments = parser.parse_args()

    ''' Parse the options and arguments '''
    featureTableName = arguments[0]
    featureTable = pd.read_csv(featureTableName, sep=options.sep)

    options.output_file = options.output_file if options.output_file else os.path.splitext(featureTableName)[0]

    options.min = ValidateInteger(options.min, 'minimum number of clusters', behaviour='default', defaultValue=2)
    options.max = ValidateInteger(options.max, 'maximum number of clusters', behaviour='default', defaultValue=200)
    options.threads = ValidateInteger(options.threads, 'number of threads', behaviour='default', defaultValue=1)
    options.force_cluster = ValidateInteger(options.force_cluster, 'user-defined number of clusters', behaviour='skip', defaultValue=None)

    ''' Determine clustering size, or skip if using --force_cluster. Append clustering information to DataFrame.
        If using ESOM to ordinate data, transform the table at this stage. '''

    contigNames = featureTable.pop('Contig')

    if options.esom_ordinate:
        options.esom_dimensions = ValidateInteger(options.esom_dimensions, 'number of ESOM dimensions', behaviour='default', defaultValue=2)
        featureTable = ReduceViaEsom(featureTable, options.esom_dimensions)
        
    else:
        featureTable = featureTable.values
        
    cEngine, n_clusters = OptmiseClustering(featureTable, options.threads, options.min, options.max, options.force_cluster)
    clusterIdentities = ApplyClustering(featureTable, n_clusters)

    '''' Log statistics, and plot if desired - skip if using force_cluster '''
    if not options.force_cluster:
        LogGapStatistics(cEngine, options.output_file)

    ''' If requested, plot ordination of points with cluster information '''
    plotObj = EsomToPlot(featureTable, contigNames, clusterIdentities) if options.esom_ordinate else ReduceToPCA(featureTable, contigNames, clusterIdentities)
    colourLookup = MapColourSpace(clusterIdentities)

    ''' Plot the outputs, if desired.
        Not the most simple method in terms of number of statements evaluated, but makes the logic simple to observe '''


    if options.plot: PlotClusters(plotObj, colourLookup, options.output_file)
    if options.plot and options.convex: PlotConvexHulls(plotObj, colourLookup, options.output_file)
    
    if options.plot and not options.force_cluster: PlotResults(cEngine, n_clusters, options.output_file)

    ''' Write out the results as a table with the ordination coordinates and cluster identifiers, then as accnos files for each bin if requested '''
    WriteOutputFiles(plotObj.df, options.output_file, options.log_contigs)

###############################################################################

# region Clustering functions

def ReduceViaEsom(df, nDimensions):

    ''' Use a 50-dimension PCA as the starting place for tSNE clustering '''
    pcaOrd = PCA(n_components=50).fit_transform(df.values)
    tsneOd = TSNE(n_components=nDimensions, method='barnes_hut').fit_transform(pcaOrd)
    return tsneOd

def OptmiseClustering(featureTable, nThreads, minSize, maxSize, forceOpt):

    if forceOpt: return None, forceOpt

    ''' Instantiate the engine according to thread requirement '''
    clustEngine = OptimalK(parallel_backend='multiprocessing', n_jobs=nThreads) if nThreads > 1 else OptimalK(parallel_backend='None')

    ''' Where multiple best-case instances are reported, the default behaviour is to return the higher number of clusters '''
    nClusters = clustEngine(featureTable, cluster_array=np.arange(minSize, maxSize))
    return clustEngine, nClusters

def ApplyClustering(featureTable, n_clusters):

    ''' Perform the actual clustering '''
    clustObj = Birch(n_clusters=n_clusters).fit( featureTable )

    ''' Create a vector of the clustering information and return it '''
    return [ 'Cluster_{}'.format(b+1) for b in clustObj.labels_ ]

def MapColourSpace(clusterColumn):
    clusterNames = list( set(clusterColumn) )
    colours = plt.cm.Spectral( np.linspace(0, 1, len(clusterNames) ) )
    return { n: c for n, c in zip(clusterNames, colours) }

# endregion

# region Plotting functions

def EsomToPlot(eTable, contigVector, clusterVector):

    ''' Slice the eTable down to just the first two dimensions. Warn the user if this reduces the dimensions '''
    dESOM = namedtuple('plotObj', ['df', 'x', 'y'])
    dESOM.df = pd.DataFrame(eTable.iloc[:,0:2], columns=['X', 'Y'])

    if eTable.shape[1] > dESOM.df.shape[1]: print( 'Warning: Computed ESOM comprised {} dimensions, but only the first 2 are plotted.'.format(eTable.shape[1]) )

    dESOM.x = 'Axis 1'; dESOM.y = 'Axis 2'

    dESOM.df['Contig'] = contigVector
    dESOM.df['Cluster'] = clusterVector

    return dESOM

def ReduceToPCA(mTable, contigVector, clusterVector):

    ''' Fit a 2D PCA '''
    pcaObj = PCA(n_components=2)
    pcCoordinates = pcaObj.fit_transform(mTable)

    ''' Package the results into a namedtuple '''
    dPCA = namedtuple('plotObj', ['df', 'x', 'y'])
    dPCA.df = pd.DataFrame(data=pcCoordinates, columns=['X', 'Y'])
    dPCA.df['Contig'] = contigVector
    dPCA.df['Cluster'] = clusterVector

    dPCA.x = 'Principal Component 1 ({:.2f}% variance)'.format( pcaObj.explained_variance_ratio_[0] * 100 )
    dPCA.y = 'Principal Component 2 ({:.2f}% variance)'.format( pcaObj.explained_variance_ratio_[1] * 100 ) 

    return dPCA

def PlotResults(gapObj, n_clusters, outputName):

    ''' Clear the plotting space '''
    plt.clf()

    ''' Plot the data '''
    plt.plot(gapObj.gap_df.n_clusters, gapObj.gap_df.gap_value, linewidth=3)
    plt.scatter(gapObj.gap_df[gapObj.gap_df.n_clusters == n_clusters].n_clusters,
                gapObj.gap_df[gapObj.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')

    ''' Tidy it up a bit, then save the figures '''
    _produceAndSavePlot(plt, 'Number of bins', 'Gap Value', outputName, 'gap_stat')

def PlotClusters(plotObj, colourLookup, outputName):

    ''' Clear the plotting space '''
    plt.clf()

    ''' Plot the points '''
    colValues = [ colourLookup[x] for x in plotObj.df.Cluster ]
    plt.scatter(plotObj.df.X, plotObj.df.Y, c=colValues)

    ''' For each cluster, add a centroid label '''
    for clusterName in sorted( plotObj.df.Cluster.unique() ):

        tempdf = plotObj.df[ plotObj.df.Cluster == clusterName ]
        x, y = _returnCentroid(tempdf)
        plt.text(x, y, clusterName, fontsize=20)

    _produceAndSavePlot(plt, plotObj.x, plotObj.y, outputName, 'cluster')

def PlotConvexHulls(plotObj, colourLookup, outputName):

    ''' Clear the plotting space '''
    plt.clf()

    ''' Plot the data '''
    for clusterName in sorted( plotObj.df.Cluster.unique() ):

        tempdf = plotObj.df[ plotObj.df.Cluster == clusterName ]
        nRow, nCol = tempdf.shape

        ''' Can only do a ConvexHull if there are at least 3 points '''
        if nRow >= 3:

            hull = ConvexHull( tempdf.loc[ : , ['X', 'Y'] ].values )
            vDF = tempdf.iloc[ hull.vertices , : ]

            plt.fill( vDF.X, vDF.Y, c=colourLookup[clusterName], alpha=0.5 )
            x, y = _returnCentroid(tempdf)
            plt.text(x, y, clusterName.replace('Cluster_', ''), fontsize=10)

        else:
            print( 'Unable to compute spatial hull for cluster {} (requires 3 points, cluster contains {})'.format(clusterName, nRow) )

    _produceAndSavePlot(plt, plotObj.x, plotObj.y, outputName, 'convex')

def _returnCentroid(df):
    return ( np.median(df.X), np.median(df.Y) )

def _produceAndSavePlot(_plt, xLabel, yLabel, outputName, suffix):
    _plt.grid(True)
    _plt.xlabel(xLabel)
    _plt.ylabel(yLabel)
    _plt.savefig('{}.{}.svg'.format(outputName, suffix), bbox_inches='tight')
    _plt.savefig('{}.{}.png'.format(outputName, suffix), bbox_inches='tight')

# endregion

# region Output files

def LogGapStatistics(cEng, outputName):
    cEng.gap_df.to_csv( '{}.gap_stats.txt'.format(outputName), sep='\t', index=False )

def WriteOutputFiles(pointDf, outputName, logSeparate):
    
    ''' The overall table '''
    pointDf.to_csv( '{}.clustered.txt'.format(outputName), sep='\t', index=False)

    ''' Per-cluster lists of contig names, if desired '''
    if logSeparate:

        clusterNames = _grabClusterNames(pointDf)
        for clusterName in clusterNames:

            outputWriter = open( '{}.{}.contigs'.format(outputName, clusterName), 'w')

            for c in pointDf[pointDf.Cluster == clusterName].Contig:
                outputWriter.write( c + '\n' )

            outputWriter.close()

# endregion

###############################################################################
if __name__ == '__main__':
     main()