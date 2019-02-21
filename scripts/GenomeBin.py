import sys # Debug only?
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from collections import namedtuple
from sklearn.metrics import matthews_corrcoef

# Currently dev/debugging imports
#from sklearn.neighbors import KernelDensity

class GenomeBin:

    def __init__(self, binInstanceTuple):

        self.binIdentifier = binInstanceTuple.binIdent

        self._eTable = pd.read_csv(binInstanceTuple.esomPath, sep='\t')
        
        self.binPoints = self._eTable[ self._eTable.BinID == binInstanceTuple.binIdent ]
        self.nonbinPoints = self._eTable[ self._eTable.BinID != binInstanceTuple.binIdent ]
        self.binPoints._is_copy = None # Just silence the slice warnings
        self.nonbinPoints._is_copy = None

        ''' Add variables for internal use '''
        self._numSlices = binInstanceTuple.numSlices
        self._mccValues = [None] * self._numSlices
        self._simplexArea = [None] * self._numSlices
        self._simplexPerimeter = [None] * self._numSlices
        self._qHulls = [None] * self._numSlices
        self._sliceArray = [None] * self._numSlices
        self._contaminatingPoints = None
        self._bestSlice = -1
        self._outputPath = binInstanceTuple.outputPath

        ''' Set a variable of the expected contig membership for MCC calculation '''
        self._expectedContigs = [ 1 if x == self.binIdentifier else 0 for x in self._eTable.BinID ]

        ''' Organise the binPoints according to distance from centroid
            For each contig, map the displacement band from the centroid that it falls into '''
        self.ComputeDistances()
        self.MapDisplacementBanding()

    #region Externally exposed functions
    def ComputeDistances(self):
        cenX, cenY = self.centroid
        self.binPoints['Distance'] = [ self.CalcDist(cenX, cenY, x, y) for x, y in zip(self.binPoints.V1, self.binPoints.V2)  ]
        self.binPoints = self.binPoints.sort_values('Distance', ascending=True)

    def MapDisplacementBanding(self):

        ''' This needs improvement - the current implementation gives a large jump from [n-1] and 1.
            Instead start on this formula, then once we pass the curve, linear extrapolation to the end. '''
        sliceSequence = [ 1 - ( 1 / (1 + x) ) for x in range(1, self._numSlices) ]
        sliceSequence.append(1.0)

        nRows = self.binPoints.shape[0]
        sliceBand = []
        prevRows = 0
        for sliceSize in sliceSequence:
            newRows = int(sliceSize * nRows) - prevRows
            prevRows = int(sliceSize * nRows)
            sliceBand.extend( [sliceSize] * newRows  )

        self.binPoints['SliceBand'] = sliceBand

    def CalcDist(self, xCen, yCen, xPos, yPos):
        dX = np.array(xCen - xPos)
        dY = np.array(yCen - yPos)    
        return np.sqrt( dX ** 2 + dY ** 2 )

    def ComputeCloudPurity(self, qManager):

        ''' Cast this out first, because it will be accessed for each sliceSize in sliceSequence '''
        nonbinArraySlice = self.nonbinPoints.loc[ : , ['V1', 'V2'] ].values

        ''' Cache the contaminating contigs and parent bin as they get observed '''
        contaminationMarkerStore = []
        topMcc = -1

        for i, sliceSize in enumerate(self.sliceSequence):

            binDfSlice = self.binPoints[ self.binPoints.SliceBand <= sliceSize ]
            self._sliceArray[i] = binDfSlice.loc[ : , ['V1', 'V2'] ].values

            ''' Identify contaminant contigs, log them for future use '''
            dHull = Delaunay(self._sliceArray[i])

            obsBins, contamContigs = self._returnContaminatingContigs(dHull, nonbinArraySlice)
            contaminationMarkerStore.append( (obsBins, contamContigs) )

            ''' Calculate the MCC '''
            obsContigs = self._resolveObservedArray(binDfSlice.ContigName, contamContigs)
            self._mccValues[i] = matthews_corrcoef(self._expectedContigs, obsContigs)
            if self._mccValues[i] >= topMcc:
                self._bestSlice = i
                topMcc = self._mccValues[i]

            ''' Record the perimeter and area of the point slice. Note that these are special cases of
                the QHull object for a 2D shape. If we project to more dimensions, this will no longer be valid '''
            self._qHulls[i] = ConvexHull(self._sliceArray[i])
            self._simplexArea[i] = self._qHulls[i].volume
            self._simplexPerimeter[i] = self._qHulls[i].area

        ''' Log the contaminating contig points at the final extension for plotting purposes '''
        if contamContigs:
            self._contaminatingPoints =  self.nonbinPoints[ self.nonbinPoints['ContigName'].isin(contamContigs) ]
        else:
            self._contaminatingPoints = None

        ''' Log each contaminating contig that was found within this slice '''
        obsBins, contamContigs = contaminationMarkerStore[ self._bestSlice ]

        if contamContigs:
            for binName, contigFragment in zip(obsBins, contamContigs):
                qManager.put( ContaminationRecord(binName, contigFragment, self.sliceSequence[ self._bestSlice ], self.binIdentifier) )

    def DropContig(self, contigName):
        print( self.binPoints.shape)
        self.binPoints = self.binPoints[ self.binPoints.ContigBase != contigName ]
        print( self.binPoints.shape)

    def to_string(self):
        return 'Name: {}, Contigs: {}, Best MCC: {}, '.format(self.binIdentifier, len(self.binPoints.ContigBase.unique()), self._mccValues[ self._bestSlice ])
    #endregion

    #region Internal manipulation functions
    def _resolveObservedArray(self, targetSlice, nontargetSlice):

        binnedContigs = set(targetSlice)

        ''' Need a bit of flow control here, to account for no contaminating contigs '''
        if nontargetSlice: binnedContigs = binnedContigs | set(nontargetSlice)

        return [ 1 if x in binnedContigs else 0 for x in self._eTable.ContigName  ]

    def _returnContaminatingContigs(self, delaunayHullObj, nonbinPointArray):

        bMask = delaunayHullObj.find_simplex(nonbinPointArray) >= 0

        if True in bMask:
            contamEvents = self.nonbinPoints.iloc[ bMask , : ]
            return list(contamEvents.BinID), list(contamEvents.ContigName)

        else:
            return None, None

    #endregion

    #region Properties
    @property
    def centroid(self):
        x = np.median(self.binPoints.V1)
        y = np.median(self.binPoints.V2)
        return (x, y)

    @property
    def bestSlice(self):
        ''' Makes use of a copy to avoid changing None -> np.nan in the _mccValues list '''
        mCopy = [ m if m else np.nan for m in self._mccValues ]
        return np.nanargmax(mCopy)

    @property
    def mccValues(self):
        ''' Returns as masked version of _mccValues, ommiting None values '''
        return [ m for m in self._mccValues if m ]

    @property
    def simplexPerimeter(self):
        ''' Returns as masked version of _simplexPerimeter, ommiting values without a corresponding MCC '''
        return [ s for (m, s) in zip(self._mccValues, self._simplexPerimeter) if m ]

    @property
    def simplexArea(self):
        ''' Returns as masked version of _simplexArea, ommiting values without a corresponding MCC '''
        return [ a for (m, a) in zip(self._mccValues, self._simplexArea) if m ]

    @property
    def sliceSequence(self):
        return self.binPoints.SliceBand.unique()

    @property
    def coreContigs(self):
        bestSliceValue = self.sliceSequence[ self._bestSlice ]
        contigs = list (self.binPoints[ self.binPoints.SliceBand <= bestSliceValue ].ContigBase.unique() )
        if contigs:
            return contigs
        else:
            return None

    #endregion

    #region Static functions

    @staticmethod
    def ParseStartingVariables(ePath, nSlices, binNames):

        ''' Just a QoL method for pre-formating the input data for constructing GenomeBin objects.
            Allows for a nice way to terminate the script early if there are path issues, before getting to the actual grunt work '''

        binDataLoader = namedtuple('binDataLoader', 'esomPath binIdent numSlices outputPath')
        dataStore = []

        esomTable = pd.read_csv(ePath, sep='\t')
        validBins = set( esomTable.BinID.unique() )

        for bN in binNames:

            if bN in validBins:
                b = binDataLoader(esomPath=ePath, binIdent=bN, numSlices=nSlices, outputPath=bN + '.refined')
                dataStore.append(b)

            else:
                print( 'Unable to find contigs associated with {}, skipping...'.format(bN) )
                continue

        dataStore = [d for d in dataStore if d ]
        if len(dataStore) > 0:
            return dataStore
        else:
            return None

    @staticmethod
    def PlotTrace(gBin):

        ''' Clear the plot, if anything came before this call '''
        plt.clf()
        fig, ax = plt.subplots()

        ax2 = ax.twinx()
        xVals = gBin.sliceSequence
        xMask = [ x for x in range(1, len(xVals)+1) ]

        ax.plot(xMask, gBin.mccValues, color='g')
        ax2.plot(xMask, [ x / y**2 for (x, y) in zip(gBin.simplexPerimeter, gBin.simplexArea) ], color='r')

        ax.set_xlabel('Contigs binned from centroid')
        ax.set_ylabel('MCC')
        ax2.set_ylabel('Contig density')

        ''' Overwrite the xMask with the actual values '''
        fig.canvas.draw()
        ax.set_xticklabels( [ '{}%'.format( round(x * 100, 1) ) for x in xVals ] )

        plt.savefig('{}.trace.png'.format(gBin._outputPath), bbox_inches='tight')

    @staticmethod
    def PlotContours(gBin):

        ''' Clear the plot, if anything came before this call '''
        plt.clf()

        optionLength = len( gBin.mccValues )
        lastQHull = gBin._qHulls[ optionLength-1 ]
        pointsArray = gBin._sliceArray[ optionLength-1 ]
        plt.fill( pointsArray[lastQHull.vertices,0], pointsArray[lastQHull.vertices,1], c='y', alpha=0.1 )

        bestQHull = gBin._qHulls[ gBin.bestSlice ]
        pointsArray = gBin._sliceArray[ gBin.bestSlice ]
        plt.fill( pointsArray[bestQHull.vertices,0], pointsArray[bestQHull.vertices,1], c='y', alpha=0.5 )

        ''' Overlay the contigs of the bin, and any contaminating points. '''
        plt.scatter(gBin.binPoints.V1, gBin.binPoints.V2, c='g')
        plt.scatter(gBin._contaminatingPoints.V1, gBin._contaminatingPoints.V2, c='r')

        plt.savefig('{}.contours.png'.format(gBin._outputPath), bbox_inches='tight')

    @staticmethod
    def SaveMccTable(gBin):

        ''' Plot the salient data in a way that can be extracted easily for creating new contig lists.
            Working idea for now - tab-delimited file with Distance - MCC - ContigNames on each line '''
        outputWriter = open(gBin._outputPath + '.mcc.txt', 'w')
        for i, s in enumerate(gBin.sliceSequence):

            binDfSlice = gBin.SliceDataFrame(gBin.binPoints, s)
            contigsNames = list( set( binDfSlice.ContigBase ) )
            contigsNames = ','.join(contigsNames)
            outputWriter.write( '{}\t{}\t{}\n'.format(s, gBin._mccValues[i], contigsNames) )

        outputWriter.close()

    @staticmethod
    def SaveCoreContigs(gBin):

        if gBin.coreContigs:

            outputWriter = open(gBin._outputPath + '.contigs.txt', 'w')
            contigString = '\n'.join(gBin.coreContigs) + '\n'
            outputWriter.write(contigString)
            outputWriter.close()
        
        else:
            print( '\tNo contigs remain in {}, skipping...'.format(gBin.binIdentifier) )

    #endregion

class ContaminationRecordManager():

    ''' This is coded in two phases;
            1) A list of dicts are used to store the record detected during phase 1 processing.
            2) The dicts are cast into a DataFrame for querying and results returning in phase 2. '''

    def __init__(self):
        self._recordFrame = None
        self._preframeRecords = []
        self._addedContigs = set()
        self._contigDistributionRecords = []

    def AddRecord(self, contamRecord):
        self._preframeRecords.append({ 'Contig': contamRecord.contigName,
                                       'OriginalBin': contamRecord.originalBin,
                                       'Displacement': contamRecord.centroidDisplacementBand,
                                       'ContigFragment': contamRecord.contigFragmentName,
                                       'ContaminationBin': contamRecord.contaminationBin })

    def IndexRecords(self):
        self._recordFrame = pd.DataFrame(self._preframeRecords)

    def CalculateContigDistributions(self, esomTablePath):

        esomTable = pd.read_csv(esomTablePath, sep='\t')
        contaminationSummary = namedtuple('contaminationSummary', 'originalBin totalFragments contamBin contamAbund carrierBins contigName')

        for contamContig in self._recordFrame.Contig.unique():

            ''' Find where the contig currently sits, and get the total number of fragments '''
            globalSlice = esomTable[ esomTable.ContigBase == contamContig ]
            currentBin = globalSlice.BinID.unique()[0]
            totalFragments = globalSlice.shape[0]

            ''' Find all instances of bins carrying this contig, and how many fragments are in each '''
            contextSlice = self._recordFrame[ self._recordFrame.Contig == contamContig ]
            fragmentDistribution = contextSlice.ContaminationBin.value_counts()

            carrierBins = fragmentDistribution.keys()
            topBin = carrierBins[0]
            topBinAbund = float(fragmentDistribution[0])
            altBins = carrierBins[1:] if len(carrierBins) > 1 else []

            cS = contaminationSummary(originalBin=currentBin,
                                      totalFragments=totalFragments,
                                      contamBin=topBin,
                                      contamAbund=topBinAbund,
                                      carrierBins=altBins,
                                      contigName=contamContig)
            self._contigDistributionRecords.append(cS)

    def ResolveContaminationByAbundance(self, binInstanceDict, biasThreshold):

        for contamEvent in self._contigDistributionRecords:

                print(contamEvent)
                if float(contamEvent.contamAbund) / contamEvent.totalFragments > biasThreshold:
                    print( binInstanceDict[ contamEvent.originalBin ].binIdentifier )
                    binInstanceDict[ contamEvent.originalBin ].DropContig(contamEvent.contigName)
                else:
                    print( binInstanceDict[ contamEvent.contamBin ].binIdentifier )
                    binInstanceDict[ contamEvent.contamBin ].DropContig(contamEvent.contigName)

                for carrierBin in contamEvent.carrierBins:
                    binInstanceDict[ carrierBin ].DropContig(contamEvent.contigName)

        return binInstanceDict

    @staticmethod
    def ExtractContigName(contigFragmentName):
        return '|'.join( contigFragmentName.split('|')[0:-1] )

class ContaminationRecord():

    def __init__(self, binName, contigFragmentName, centroidDisplacementBand, contaminationBinName):
        self.contigName = ContaminationRecordManager.ExtractContigName(contigFragmentName)
        self.originalBin = binName
        self.contigFragmentName = contigFragmentName
        self.centroidDisplacementBand = centroidDisplacementBand
        self.contaminationBin = contaminationBinName

    def to_string(self):
        print( 'Bin: {}, contig ({}) from bin {}'.format(self.originalBin, self.contigName, self.contaminationBin) )