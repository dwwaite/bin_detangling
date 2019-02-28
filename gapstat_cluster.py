'''
    Options supported by sklearn which may be appropriate are:
        1. K-means clustering (distance between points; produces equal-abundance clusters which is not appropriate)
        2. Affinity propagation (graph distance)
        3. Mean-shift (distance between points; can't specify numbers of clusters)
        4. DBSCAN (distance between points; can't specify numbers of clusters)
        5. Birch (distance between points; uneven clusters allowed)

    Find them at https://scikit-learn.org/stable/modules/clustering.html
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
#from sklearn.datasets.samples_generator import make_blobs # Only for generating test data

# Custom imports
from gap_statistic import OptimalK
from scripts.OptionValidator import ValidateInteger


def main():

    parser = OptionParser()
    usage = "usage: %prog [options] [feature table]"

    ''' Parse and validate options - simple/expected choices '''
    parser.add_option('-t', '--threads', help='Number of threads to use (Default = 1)', dest='threads')
    parser.add_option('--min', help='Minimum number of clusters to test (Default = 2)', dest='min', default=2)
    parser.add_option('--max', help='Maximum number of clusters to test (Default = 200)', dest='max', default=200)
    parser.add_option('--force_cluster', help='Skip gap-stat evaluation and cluster into N clusters', dest='force_cluster', default=None)
    parser.add_option('--sep', help='Separator value to delimit columns (Default = [tab])', dest='sep', default='\t')
    parser.add_option('-o', '--output', help='Name stub for all output files (Default: Inherited from input table)', dest='output_file')
    parser.add_option('--log_contigs', help='Create per-cluster files containing contig names (Default: False)', dest='log_contigs', action='store_true')
    parser.add_option('-p', '--plot', help='Plot the decision process (Default: False, but highly recommended)', dest='plot', action='store_true')
    parser.add_option('--convex', help='Plot convex hulls of the bin outlines. Useful for deciding whether or not to merge bins (Default: False)', dest='convex', action='store_true')


    options, arguments = parser.parse_args()

    ''' Parse the options and arguments '''
    featureTableName = arguments[0]
    featureTable = ParseFeatureTable(featureTableName, options.sep)

    options.output_file = options.output_file if options.output_file else os.path.splitext(featureTableName)[0]

    options.min = ValidateInteger(options.min, 'minimum number of clusters', behaviour='default', defaultValue=2)
    options.max = ValidateInteger(options.max, 'maximum number of clusters', behaviour='default', defaultValue=200)
    options.threads = ValidateInteger(options.threads, 'number of threads', behaviour='default', defaultValue=1)
    options.force_cluster = ValidateInteger(options.force_cluster, 'user-defined number of clusters', behaviour='skip', defaultValue=None)

    ''' Determine clustering size, or skip if using --force_cluster. Append clustering information to DataFrame '''
    cEngine, n_clusters = OptmiseClustering(featureTable, options.threads, options.min, options.max, options.force_cluster)
    mappedFeatureTable = ApplyClustering(featureTable, n_clusters)

    '''' Log statistics, and plot if desired - skip if using force_cluster '''
    if not options.force_cluster:
        LogGapStatistics(cEngine, options.output_file)

    ''' If request, plot ordination of points with cluster information '''
    plotPCA = ReduceToPCA(mappedFeatureTable)
    colourLookup = MapColourSpace( mappedFeatureTable.Cluster )

    if options.plot:

        PlotClusters(plotPCA, colourLookup, options.output_file)

        ''' Plot the clustering decision process, if requested '''
        if not options.force_cluster: PlotResults(cEngine, n_clusters, options.output_file)

    if options.convex: PlotConvexHulls(plotPCA, colourLookup, options.output_file)

    ''' Write out the results as a table with the ordination coordinates and cluster identifiers, then as accnos files for each bin if requested '''
    WriteOutputFiles(plotPCA.df, options.output_file, options.log_contigs)

###############################################################################

# region Clustering functions

def OptmiseClustering(featureTable, nThreads, minSize, maxSize, forceOpt):

    if forceOpt:
        return None, forceOpt

    ''' Instantiate the engine according to thread requirement '''
    clustEngine = OptimalK(parallel_backend='multiprocessing', n_jobs=nThreads) if nThreads > 1 else OptimalK(parallel_backend='None')

    _, featureFrame = ExtractColumn(featureTable, 'Contig')

    ''' Where multiple best-case instances are reported, the default behaviour is to return the higher number of clusters '''
    nClusters = clustEngine(featureFrame.values, cluster_array=np.arange(minSize, maxSize))
    return clustEngine, nClusters

def ApplyClustering(featureTable, n_clusters):

    ''' Perform the actual clustering '''
    _, featureFrame = ExtractColumn(featureTable, 'Contig')
    clustObj = Birch(n_clusters=n_clusters).fit( featureFrame.values )

    ''' Append results to the DataFrame, then return it '''
    featureTable = _bindClustering(featureTable, clustObj)

    return featureTable

def _bindClustering(dTable, clustObj):
    dTable['Cluster'] = [ 'Cluster_{}'.format(b+1) for b in clustObj.labels_ ]
    return dTable

def MapColourSpace(clusterColumn):
    clusterNames = list( set(clusterColumn) )
    colours = plt.cm.Spectral( np.linspace(0, 1, len(clusterNames) ) )
    return { n: c for n, c in zip(clusterNames, colours) }

# endregion

# region DataFrame manipulation

def ParseFeatureTable(pFile, _sep):
    return pd.read_csv(pFile, sep=_sep)

def ExtractColumn(dTable, colName):
    colValues = dTable.loc[:,[colName]]
    dfValues = dTable.drop(colName, axis=1)
    return colValues, dfValues

# endregion

# region Plotting functions

def ReduceToPCA(mTable):

    ''' Split out the contigs, then fit a 2D PCA '''
    contigNames, dFrameCluster = ExtractColumn(mTable, 'Contig')
    clusterNames, dFrame = ExtractColumn(dFrameCluster, 'Cluster')
    dArray = dFrame.values

    pcaObj = PCA(n_components=2)
    pcCordinates = pcaObj.fit_transform(dArray)

    ''' Package the results into a namedtuple '''
    dPCA = namedtuple('dPCA', ['df', 'pc1_label', 'pc2_label'])
    dPCA.df = pd.DataFrame(data=pcCordinates, columns=['PC1', 'PC2'])
    dPCA.df.insert(loc=0, column='Contig', value=mTable.Contig)
    #dPCA.df['Contig'] = mTable.Contig
    dPCA.df['Cluster'] = mTable.Cluster
    dPCA.pc1_label = 'Principal Component 1 ({:.2f}% variance)'.format( pcaObj.explained_variance_ratio_[0] * 100 )
    dPCA.pc2_label = 'Principal Component 2 ({:.2f}% variance)'.format( pcaObj.explained_variance_ratio_[1] * 100 ) 

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

def PlotClusters(pcaObj, colourLookup, outputName):

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

def PlotConvexHulls(pcaObj, colourLookup, outputName):

    ''' Clear the plotting space '''
    plt.clf()

    ''' Plot the data '''
    clusterNames = _grabClusterNames(pcaObj.df)
    for clusterName in clusterNames:

        tempdf = pcaObj.df[ pcaObj.df.Cluster == clusterName ]
        nRow, nCol = tempdf.shape

        ''' Can only do a ConvexHull if there are at least 3 points '''
        if nRow >= 3:

            hull = ConvexHull( tempdf.loc[ : , ['PC1', 'PC2'] ].values )
            vDF = tempdf.iloc[ hull.vertices , : ]

            plt.fill( vDF.PC1, vDF.PC2, c=colourLookup[clusterName], alpha=0.5 )
            x, y = _returnCentroid(tempdf)
            plt.text(x, y, clusterName.replace('Cluster_', ''), fontsize=10)

        else:
            print( 'Unable to compute spatial hull for cluster {} (requires 3 points, cluster contains {})'.format(clusterName, nRow) )

    _produceAndSavePlot(plt, pcaObj.pc1_label, pcaObj.pc2_label, outputName, 'convex')

def _grabClusterNames(pointDf):
    return sorted( pointDf.Cluster.unique() )

def _returnCentroid(df):
    return ( np.median(df.PC1), np.median(df.PC2) )

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