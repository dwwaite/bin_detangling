'''
    ESOMana is a powerful way to detangle metagenomic bins, but comes with the following headache:

    * What I want = A finite, low-dimensional representation of the distance between contig points, from which ML models can be trained
    * What I get = An infinite, low-dimensions representation of the distance between contig points, as the ESOM repeats over the edges of the
        plotting space.

        E.g. For a plot with 10 x 10 neurons, the neuron at [9,1] is 1 x-value away from [8,1] and [10,1], but then 2 x-values away from [7,1] and [1,1].
             The same is true for y-values.

    Therefore, we cannot just use the neuron positions as the V1 and V2 values, because it will give too great a distance between points near the edges of the map.
    I need to reduce the data into a set of coordinates.

    To this end, this script:
    
        1. Takes the *.bm file from ESOMana and creates a distance matrix between all points
        2. Creates an MDS representation of the data
        3. Writes the values out into a V1 - Vn projection of the data
        4. Reports the number of axis required to explain the top X% of variance at various pre-defined cutoffs.
'''

import sys, os
import pandas as pd
import numpy as np
from optparse import OptionParser
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# My functions and classes
from scripts.OptionValidator import ValidateFile, ValidateInteger, ValidateFloat

def main():

    # Parse options
    usageString = "usage: %prog [options] [bm file] [output file]"
    parser = OptionParser(usage=usageString)

    parser.add_option('-u', '--umx', help='An ESOMana umx file of the neuron heights', dest='umx', default='')
    parser.add_option('-n', '--names', help='An ESOMana names file to map contig names to coordinates', dest='names', default=None)
    parser.add_option('-d', '--dimensions', help='Maximum number of dimensions for nMDS', dest='dimensions', default=30)
    parser.add_option('-s', '--stress', help='Maximum allowed stress for nMDS fit', dest='stress', default=0.2)
    parser.add_option('-t', '--threads', help='Number of threads for nMDS calculation', dest='threads', default=1)
    parser.add_option('--add-contig-base', help='TODO', dest='add_base', action='store_true', default=False)

    options, args = parser.parse_args()
    bmFile, outputStub = args[0:2]

    '''
        Step 1. Validate the input files, ensuring that the process ends here if it will not succeed.
    '''
    options.umx = ValidateFile(inFile=options.umx, behaviour='abort', fileTypeWarning='umx (heights) file')

    if options.names:
        options.names = ValidateFile(inFile=options.names, behaviour='skip', fileTypeWarning='names file')

    ValidateFile(inFile=bmFile, behaviour='abort', fileTypeWarning='bm file')

    options.dimensions = ValidateInteger(userChoice=options.dimensions, parameterNameWarning='dimensions', behaviour='default', defaultValue=30)
    options.stress = ValidateFloat(userChoice=options.stress, parameterNameWarning='stress', behaviour='default', defaultValue=0.2)
    options.threads = ValidateInteger(userChoice=options.threads, parameterNameWarning='threads', behaviour='default', defaultValue=1)

    '''
        Step 2. Import the data, and translate the coordinates into Euclidean distances, accounting for the wrap-around of the ESOM space.
                Return values are:
                    1. Distance matrix of each unique neuron pair in the map
                    2. Sequence of neuron coordinates, ordered as per python sorting
                    3. Dict that maps each unique neuron coordinate to the contigs it represents
    '''
    coordinateMatrix = ImportBmMatrix(bmFile, options.names)
    heightMatrix = ImportHeightMatrix(options.umx)

    coordinateDistances, coordinateSequence, coordinateMap = CalculateMatrixDistances(coordinateMatrix, heightMatrix)

    '''
        Step 3. Attempt to reduce the distance matrix into an nMDS ordination that fits within the desired stress and dimensionality constraints.
                If the desired stress cannot be reached within the specified dimensions, script aborts and reports current progress

                Note: There is a known complaint about sklearn.manifold.MDS in that it doesn't report the stress value we normally get in R.
                      As such, it's not necessarily practical to aim for the typical stress < 0.2 marker.
    '''
    coordinateValues, dimensions, stress = OptmiseMDS(coordinateDistances, options.stress, options.dimensions, options.threads)
    print( 'Ordination completed with a stress of {:.3f} in {} dimensions.'.format(stress, dimensions) )

    '''
        Step 4. Save the results as a coordinate table ready for the expand_by_mcc.py script.
    '''
    ordinationTable = PopulateOrdinationTable(coordinateValues, coordinateSequence, coordinateMap)
    ordinationTable.to_csv( '{}.coord_table.txt'.format(outputStub), sep='\t', index=False)

###############################################################################

# region File import

def ImportBmMatrix(bmPath, namesFile=None):

    b = pd.read_csv(bmPath, header=None, names=[ 'Contig', 'X', 'Y' ], sep='\t', skiprows=2)

    if namesFile:
        n = pd.read_csv(namesFile, header=None, names=[ 'Index', 'Contig' ], sep='\t', skiprows=1)
        b.Contig = n['Contig']

    return b

def ImportHeightMatrix(umxPath):

    u = pd.read_csv(umxPath, header=None, sep='\t', skiprows=1)
    return u.values

# endregion

# region Distance calculation

def CalculateMatrixDistances(coordinateMatrix, heightMatrix):

    ''' Start by identifying just the unique points to reduce the numbe of calcuations needed '''
    pointMatrix, pointSequence, pointMap = _reduceMatrixSpace(coordinateMatrix, heightMatrix)

    ''' Convert the point matrix into Euclidean distance matrix, rows/columns ordered the same as the input data '''
    pointDistances = _pointToDist(pointMatrix)
    return pointDistances, pointSequence, pointMap

def _reduceMatrixSpace(cMatrix, heightMatrix):

    '''
        Take a DataFrame of the form [ 'Contig', 'X', 'Y' ] and parse into a matrix of unique [ X, Y, height ] triplets and a dict that maps each pair to all Contig values
            Note that the X and Y values are their original int values, while 'height' is already scaled.
            This must be addressed in the dist calculation, since I can't pre-normalise these values

        This is probably not the most efficient implementation, but it is an easy to understand one, and this is not a particularly heavy script.
    '''
    uniquePairs = []
    uniquePairsOrder = []
    pairMap = {}

    ''' Create a new column to perform groupby operation and cache the heights of each pair in the map '''
    coordMerge = [None] * cMatrix.shape[0]
    heightMap = {}
    for i in range(cMatrix.shape[0]):

        x, y = cMatrix.X[i], cMatrix.Y[i]
        pairKey = '{},{}'.format(x, y)

        coordMerge[i] = pairKey
        heightMap[pairKey] = heightMatrix[x, y]

    cMatrix['coord_fusion'] = coordMerge

    ''' Record the unique pair x/y/height values, and the contigs represented by each neuron '''
    for pair, df in cMatrix.groupby('coord_fusion'):

        ''' Record the pair values '''
        uniquePairs.append( { 'X': list(df.X)[0], 'Y': list(df.Y)[0], 'H': heightMap[pair] } )
        uniquePairsOrder.append(pair)

        ''' Store the contigs '''
        pairMap.setdefault(pair, []).extend( list(df.Contig) )

    ''' Create the return matrix and organise the columns '''
    uniquePairDf = pd.DataFrame(uniquePairs)
    uniquePairDf = uniquePairDf[['X', 'Y', 'H']]

    return uniquePairDf.values, uniquePairsOrder, pairMap

def _pointToDist(pMatrix):
    '''
        This is likely the worst-case implementation of this algorithm.

        For now, I just need something to get started with so am using the quickest-to-work function for getting the distances.
        In the future, there is probably a way to revise this calculation using a meshgrid to speed the process up.
    '''
    maxX = np.max( pMatrix[:,0] )
    maxY = np.max( pMatrix[:,1] )

    nPoints = pMatrix.shape[0]
    distArray = np.zeros(shape=(nPoints,nPoints))

    for i, j in combinations(range(nPoints), 2):

        ''' At this stage I need to scale the X and Y distances by their max total, otherwise I get 2 int deltas weighted against the height.
            Since height it already scaled, the difference in height between two nodes is a simple absolute difference '''
        dX = _calcDist(maxX, pMatrix[i,0], pMatrix[j,0]) / maxX
        dY = _calcDist(maxY, pMatrix[i,1], pMatrix[j,1]) / maxY
        dH = np.abs( pMatrix[i,2] - pMatrix[j,2] )

        d = np.sqrt( dX ** 2 + dY ** 2 + dH ** 2)

        distArray[i,j] = d
        distArray[j,i] = d

    return distArray

def _calcDist(maxA, *vals):
    x1 = max(vals)
    x2 = min(vals)
    return min( x1 - x2, x2 + maxA - x1 )

# endregion

# region Ordination table

def OptmiseMDS(distArray, maxStress, maxComponents, nThreads):

    ''' Tracer for if stress target is not acheived '''
    stressValues = []
    dimensionValues = []

    currentComponents = 2
    distTransformed = MDS(n_components=currentComponents, metric=False, dissimilarity='precomputed', n_jobs=nThreads).fit(distArray)

    if distTransformed.stress_ <= maxStress:
        return distTransformed.embedding_, currentComponents, distTransformed.stress_

    stressValues.append(distTransformed.stress_)
    dimensionValues.append(currentComponents)

    currentIterations = 0
    while distTransformed.stress_ > maxStress and currentComponents <= maxComponents:

        currentComponents += 1
        distTransformed = MDS(n_components=currentComponents, metric=False, dissimilarity='precomputed', n_jobs=nThreads).fit(distArray)

        stressValues.append(distTransformed.stress_)
        dimensionValues.append(currentComponents)

        if distTransformed.stress_ <= maxStress:
            return distTransformed.embedding_, currentComponents, distTransformed.stress_

    print( 'Unable to achieve desired stress of {} in less than {} dimensions. Aborting...'.format(maxStress, maxComponents) )

    for d, s in zip(dimensionValues, stressValues):
        print( '  Dimensions: {}, Stress: {} '.format(d, s) )

    sys.exit()

def PopulateOrdinationTable(coordinateValues, coordinateSequence, coordinateMap):

    resultsList = []

    for i, coordPair in enumerate(coordinateSequence):

        coordList = list( coordinateValues[i,:] )
        pointRecord = { 'V{}'.format(j+1): c for j, c in enumerate(coordList) }

        ''' This loop will add the first entry, then on future entries just update the index '''
        for contig in coordinateMap[coordPair]:

            cRecord = pointRecord.copy()
            pointRecord['Contig'] = contig
            resultsList.append(pointRecord)

    resultsFrame = pd.DataFrame(resultsList)
    return resultsFrame

# endregion

###############################################################################
if __name__ == '__main__':
    main()