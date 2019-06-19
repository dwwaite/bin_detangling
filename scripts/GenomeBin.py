import os, math, uuid, operator
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from sklearn.metrics import matthews_corrcoef

class GenomeBin:

    def __init__(self, bin_name, esom_path, number_of_slices, output_path):

        self.bin_name = bin_name
        self.output_path = output_path
        self.number_of_slices = number_of_slices

        ''' Prepare the internal DataFrame of ESOM coordinates.
        
            1. Import the ESOM table
            2. Identify the centroid of the bin
            3. Measure the distance from each point to the centroid, then sort ascending
            4. Slice the table to the last binned contig
        '''
        self.esom_table = pd.read_csv(esom_path, sep='\t')

        cenX, cenY = self.centroid
        self.esom_table['Distance'] = [ self._calcDist(cenX, cenY, x, y) for x, y in zip(self.esom_table.V1, self.esom_table.V2) ]
        self.esom_table = self.esom_table.sort_values('Distance', ascending=True)

        last_contigfragment = max(idx for idx, val in enumerate( self.esom_table.BinID ) if val == self.bin_name) + 1
        self.esom_table = self.esom_table.head(last_contigfragment)
        self._expectedContigs = [ 1 if contig_bin == self.bin_name else 0 for contig_bin in self.esom_table.BinID ]

        ''' Add variables for internal use:
            iteration_scores creates a pd.DataFrame of numeric metrics
            slice, and contam_contigs is a NoSQL-like storage of complex objects '''
        self._iteration_scores = []
        self._slice_contigs = {}
        self._contam_contigs = {}

    #region Externally exposed functions

    def ComputeCloudPurity(self, qManager):

        for frame_slice in self._getNextSlice():

            ''' Create a UUID for storing contig records '''
            slice_key = uuid.uuid4()

            ''' Create hulls for later use'''
            arr = frame_slice[ ['V1', 'V2'] ].values
            dHull = Delaunay(arr)
            q_hull = ConvexHull(arr)

            ''' Identify contaminant contigs '''
            contam_record = self._identifyContaminatingContigs(dHull, frame_slice)

            ''' Calculate the MCC '''
            obsContigs = self._resolveObservedArray(frame_slice.ContigName, contam_record.ContigName)
            slice_mcc = matthews_corrcoef(self._expectedContigs, obsContigs)

            ''' Record the perimeter and area of the point slice. Note that these are special cases of
                the QHull object for a 2D shape. If we project to more dimensions, this will no longer be valid '''
            slice_area = q_hull.volume
            slice_perimeter = q_hull.area

            self._storeQualityValues(slice_mcc, slice_key, slice_area, slice_perimeter, frame_slice, contam_record)

        ''' Have two data types to store in the qManager - the modified bin object, and contamination records
            Store a tuple, with the first value as a binary switch for bin vs contam '''        
        qManager.put( (True, self) )

        top_key = self._returnTopKey()
        contam_contigs = self._contam_contigs[ top_key ]

        for new_bin, contig_base, contig_fragment in zip(contam_contigs.BinID, contam_contigs.ContigBase, contam_contigs.ContigName):

            if not new_bin == self.bin_name:

                cRecord = ContaminationRecord(self.bin_name, new_bin, contig_fragment, contig_base)
                qManager.put( (False, cRecord) )

    def DropContig(self, contigName):

        self.esom_table = self.esom_table[ self.esom_table.ContigBase != contigName ]

    """
    def to_string(self):
        return 'Name: {}, Contigs: {}, Best MCC: {}, '.format(self.bin_name, len(self.binPoints.ContigBase.unique()), self._mccValues[ self._bestSlice ])
    """
    #endregion

    #region Internal manipulation functions

    def _calcDist(self, xCen, yCen, xPos, yPos):
        dX = np.array(xCen - xPos)
        dY = np.array(yCen - yPos)    
        return np.sqrt( dX ** 2 + dY ** 2 )

    def _getNextSlice(self):

        n_contigs = self.esom_table.shape[0]
        for x in range(1, self.number_of_slices+1):

            hits = np.sin( x / self.number_of_slices * np.pi/2 ) * n_contigs
            yield self.esom_table.head( int(hits) )

    def _identifyContaminatingContigs(self, delaunay_hull, frame_slice):

        bMask = delaunay_hull.find_simplex( self.esom_table[ ['V1', 'V2'] ].values ) >= 0

        if True in bMask:
            return frame_slice.iloc[ bMask , : ]
        else:
            return None

    def _resolveObservedArray(self, targetSlice, nontargetSlice):

        binnedContigs = set(targetSlice)

        ''' Need a bit of flow control here, to account for no contaminating contigs '''
        if set(nontargetSlice):
            binnedContigs = binnedContigs | set(nontargetSlice)

        return [ 1 if x in binnedContigs else 0 for x in self.esom_table.ContigName  ]

    def _storeQualityValues(self, mcc, slice_key, area, perimeter, frame_slice, contam_contigs):

        self._iteration_scores.append( { 'MCC': mcc, 'Key': slice_key, 'Area': area, 'Perimeter': perimeter })
        self._slice_contigs[ slice_key ] = frame_slice
        self._contam_contigs[slice_key] = contam_contigs

    def _returnTopKey(self):

        top_slice = pd.DataFrame( self._iteration_scores ).nlargest(1, 'MCC')
        return top_slice.Key.values[0]

    #endregion

    #region Properties

    @property
    def bin_contigs(self):
        return list( self.esom_table[ self.esom_table.BinID == self.bin_name ].BinID )

    @property
    def nonbin_contigs(self):
        return list( self.esom_table[ self.esom_table.BinID != self.bin_name ].BinID )

    @property
    def centroid(self):

        df = self.esom_table[ self.esom_table.BinID == self.bin_name ]
        x = np.median(df.V1)
        y = np.median(df.V2)
        return (x, y)

    @property
    def mccSequence(self):
        return pd.DataFrame( self._iteration_scores )['MCC']

    @property
    def polsbyPopperScores(self):

        '''
            This is a metric that was originally described by Cox (1927; Journal of Paleontology 1(3): 179-183),
            but has more recently been re-discovered in poltical science by Polsbsy an Popper.
            
            It is simply a measure of difference between a shape and a circle of similar size, giving a measure
            on how compact the shape is (0 = no compactness, 1 = ideal).
        '''
        df = pd.DataFrame( self._iteration_scores )
        return [ 4 * np.pi * area / perimeter ** 2 for perimeter, area in zip(df.Perimeter, df.Area) ]

    """
    @property
    def coreContigs(self):

        ''' If there is no optimal MCC, just return a fail '''
        if self._bestSlice == -1: return None

        ''' Otherwise, work out the list of contigs within the MCC-maximising zone '''
        bestSliceValue = self.sliceSequence[ self._bestSlice ]
        contigs = list (self.binPoints[ self.binPoints.SliceBand <= bestSliceValue ].ContigBase.unique() )
        if contigs:
            return contigs
        else:
            return None
    """
    #endregion

    #region Output functions

    def _split_contig_sets(self):

        top_key = self._returnTopKey()

        ''' Get sets off all contigs in the bin space '''
        all_contigs = set( self.bin_contigs )
        all_contam = set( self.nonbin_contigs )

        ''' Find the contigs and contamination fragments inside the core '''
        core_contigs = set( self._slice_contigs[top_key].ContigName )
        contam_contigs = set (self._contam_contigs[top_key].ContigName )

        ''' Identify those outside the core '''
        outsider_contigs = all_contigs - core_contigs
        outsider_contams = all_contam - contam_contigs

        return core_contigs, contam_contigs, outsider_contigs, outsider_contams

    def PlotContours(self):

        ''' Define a colour vector for plotting:
                1. Core contigs = #1f78b4
                2. Contigs outside core = #a6cee3
                3. Contamination contigs (core) = #e31a1c
                4. Contamination contigs (outside) = #fb9a99
        '''
        core_contigs, contam_contigs, outsider_contigs, outsider_contams = self._split_contig_sets()

        ''' Clear the plot, if anything came before this call '''
        plt.clf()

        for contig_set, colour in zip([core_contigs, contam_contigs, outsider_contigs, outsider_contams], ['#1f78b4', '#a6cee3', '#e31a1c', '#fb9a99']):

            df = self.esom_table[ self.esom_table.ContigName.isin(contig_set) ]
            plt.scatter(df.V1, df.V2, c=colour)

        plt.savefig('{}.scatter.png'.format(self.output_path), bbox_inches='tight')

    def PlotTrace(self):

        ''' Clear the plot, if anything came before this call '''
        plt.clf()
        fig, ax = plt.subplots()

        x_vals = [ x + 1 for x in range(0, len(self._iteration_scores) ) ]

        ''' Plot the MCC against the left x-axis '''
        ax.plot(x_vals, self.mccSequence, color='g', label='MCC')
        ax.set_xlabel('Contig slice')
        ax.set_ylabel('MCC')
        ax.set_ylim([0.0, 1.0])

        ''' Plot the PPP against the right x-axis '''
        ax2 = ax.twinx()
        ax2.plot(x_vals, self.polsbyPopperScores, color='r', label='PP value')
        ax2.set_ylabel('Polsby-Popper value')
        ax2.set_ylim([0.0, 1.0])

        fig.legend(loc='upper right')
        plt.savefig('{}.trace.png'.format(self.output_path), bbox_inches='tight')

    #endregion

    #region Static functions

    @staticmethod
    def ParseStartingVariables(esom_table_path, number_of_slices, bin_names, output_prefix = None):

        ''' Just a QoL method for pre-formating the input data for constructing GenomeBin objects.
            Allows for a nice way to terminate the script early if there are path issues, before getting to the actual grunt work '''

        return_tuples = []
        esom_table_bins = set( pd.read_csv(esom_table_path, sep='\t').BinID.unique() )

        for bin_name in bin_names:

            if bin_name in esom_table_bins:

                outputPath = output_prefix + bin_name + '.refined' if output_prefix else bin_name + '.refined'
                return_tuples.append( (bin_name, esom_table_path, number_of_slices, outputPath) )

        return return_tuples

    """
    @staticmethod
    def SaveMccTable(gBin):

        ''' Plot the salient data in a way that can be extracted easily for creating new contig lists.
            Working idea for now - tab-delimited file with Distance - MCC - ContigNames on each line '''
        outputWriter = open(gBin.output_path + '.mcc.txt', 'w')
        for i, s in enumerate(gBin.sliceSequence):

            binDfSlice = gBin.SliceDataFrame(gBin.binPoints, s)
            contigsNames = list( set( binDfSlice.ContigBase ) )
            contigsNames = ','.join(contigsNames)
            outputWriter.write( '{}\t{}\t{}\n'.format(s, gBin._mccValues[i], contigsNames) )

        outputWriter.close()

    @staticmethod
    def SaveCoreContigs(gBin):

        if gBin.coreContigs:

            outputWriter = open(gBin.output_path + '.contigs.txt', 'w')
            contigString = '\n'.join(gBin.coreContigs) + '\n'
            outputWriter.write(contigString)
            outputWriter.close()

        else:
            print( 'No contigs remain in {}, skipping...'.format(gBin.bin_name) )

    @staticmethod
    def CreateCoreTable(gBinList):

        binMap = []

        for gBin in gBinList:
            if gBin.coreContigs:

                binMap.extend( [ { 'Bin': gBin.bin_name, 'ContigBase': c } for c in gBin.coreContigs ] )

            else:
                print( 'No contigs remain in {}, skipping...'.format(gBin.bin_name) )
                continue

        return pd.DataFrame(binMap)
    """
    #endregion

class ContaminationRecord():

    def __init__(self, current_bin, new_bin, contig_fragment, contig_base):

        self.current_bin = current_bin
        self.new_bin = new_bin
        self.contig_fragment = contig_fragment
        self.contig_base = contig_base

    @staticmethod
    def BuildContaminationFrame(contam_record_list):

        preframe_list = [ { 'ContigBase': cr.contig_base, 'CurrentBin': cr.current_bin, 'ContigFragment': cr.contig_fragment, 'ContaminationBin': cr.new_bin } for cr in contam_record_list ]
        return pd.DataFrame(preframe_list)

    @staticmethod
    def CountContigFragments(esom_table):

        return { contig: df.shape[0] for contig, df in pd.read_csv(esom_table, sep='\t').groupby('ContigBase') }

    @staticmethod
    def ResolveContaminationByAbundance(bin_instance_dict, contam_table, contam_counter, bias_threshold):

        for contig, fragment_df in contam_table.groupby('ContigBase'):

            ''' Count the number of contamination fragments, then find the bin holding the most '''
            contam_bin_dist = dict( Counter( fragment_df.ContaminationBin ) )
            top_contam_bin = max(contam_bin_dist.items(), key=operator.itemgetter(1))[0]

            ''' Create a list of all contamination bins
                If a contamination bin passes the threshold, drop it from the original bin and assign it to the contamination one '''
            current_bin = fragment_df.CurrentBin.values[0]
            contam_bins = list( fragment_df.ContaminationBin.unique() )

            if float( contam_bin_dist[top_contam_bin] ) / contam_counter[contig] >= bias_threshold:

                bin_instance_dict[ current_bin ].DropContig(contig)
                contam_bins = contam_bins.remove(top_contam_bin)
            
            if contam_bins:

                for contam_bin in contam_bins:
                    bin_instance_dict[ contam_bin ].DropContig(contig)

        return bin_instance_dict