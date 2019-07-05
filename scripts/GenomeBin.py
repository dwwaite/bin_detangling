import os, math, uuid, operator
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
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
        self.mcc_expectation = [ 1.0 if contig_bin == self.bin_name else 0.0 for contig_bin in self.esom_table.BinID ]

        ''' Add variables for internal use:
            iteration_scores creates a pd.DataFrame of numeric metrics
            slice, and contam_contigs is a NoSQL-like storage of complex objects '''
        self._iteration_scores = []
        self._slice_df_lookup = {}

        self._contig_core = None

    #region Externally exposed functions

    def ComputeCloudPurity(self, qManager):

        for frame_slice in self._getNextSlice():

            ''' Create a UUID for storing contig records '''
            slice_key = uuid.uuid4()          

            ''' Calculate the MCC '''
            slice_mcc = self._compute_mcc(frame_slice)

            ''' Record the perimeter and area of the point slice.
                Note that these are special cases of the ConvexHull for a 2D shape. If we project to more dimensions, this will no longer be valid '''
            q_hull = ConvexHull( frame_slice[ ['V1', 'V2'] ].values )
            slice_area = q_hull.volume
            slice_perimeter = q_hull.area

            self._storeQualityValues(slice_mcc, slice_key, slice_area, slice_perimeter, frame_slice )

        ''' Have two data types to store in the qManager - the modified bin object, and contamination records
            Store a tuple, with the first value as a binary switch for bin vs contam '''        
        qManager.put( (True, self) )

        top_key = self.top_key
        contam_contigs = self.slice_contigs_nonbin(top_key)
 
        for new_bin, contig_base, contig_fragment in zip(contam_contigs.BinID, contam_contigs.ContigBase, contam_contigs.ContigName):

            cRecord = ContaminationRecord(self.bin_name, new_bin, contig_fragment, contig_base)
            qManager.put( (False, cRecord) )

    def DropContig(self, contigName):

        self.esom_table = self.esom_table[ self.esom_table.ContigBase != contigName ]

    def build_contig_base_set(self):

        df = self._slice_df_lookup[ self.top_key ]

        contig_vector = df[ df.BinID == self.bin_name ].ContigBase
        self._contig_core = set( contig_vector )

    def add_contig_to_base_set(self, contig):

        if self._contig_core:
            self._contig_core.add(contig)

    def remove_contig_from_base_set(self, contig):

        if self._contig_core:

            if contig in self._contig_core:

                self._contig_core.remove(contig)

    def count_contig_fragments(self, contig_name):

        df = self.esom_table[ self.esom_table.ContigBase == contig_name ]
        return df.shape[0]

    #endregion

    #region Internal manipulation functions

    def _calcDist(self, xCen, yCen, xPos, yPos):
        dX = np.array(xCen - xPos)
        dY = np.array(yCen - yPos)    
        return np.sqrt( dX ** 2 + dY ** 2 )

    def _getNextSlice(self):

        '''
            This function returns slices along the full bin space divide amongst all points within it.
            Alternately, could map the values across only the bin contigs?
        '''
        n_contigs = self.esom_table.shape[0]
        for x in range(1, self.number_of_slices+1):

            hits = np.sin( x / self.number_of_slices * np.pi/2 ) * n_contigs
            yield self.esom_table.head( int(hits) )

    def _compute_mcc(self, slice_df):

        ''' Added some handling to suppress matthews_corrcoef warnings.
            There is an edge case where if there are no false contigs in a bin,
                the MCC encounters a divide by zero.
            Where this is suspected, a result is faked.
        '''
        obs_vector = [0.0] * len( self.mcc_expectation )
        for i in range(0, slice_df.shape[0]): obs_vector[i] = 1.0

        '''
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                mcc = 
            except:
                return -1.0
        '''
        return matthews_corrcoef(self.mcc_expectation, obs_vector)

    def _storeQualityValues(self, mcc, slice_key, area, perimeter, frame_slice):

        self._iteration_scores.append( { 'MCC': mcc, 'Key': slice_key, 'Area': area, 'Perimeter': perimeter })
        self._slice_df_lookup[ slice_key ] = frame_slice

    def slice_contigs_bin(self, slice_key):

        df = self._slice_df_lookup[ slice_key ]
        return df[ df.BinID == self.bin_name ]

    def slice_contigs_nonbin(self, slice_key):

        df = self._slice_df_lookup[ slice_key ]
        return df[ df.BinID != self.bin_name ]

    #endregion

    #region Properties

    @property
    def bin_contigs(self):
        return list( self.esom_table[ self.esom_table.BinID == self.bin_name ].ContigName )

    @property
    def nonbin_contigs(self):
        return list( self.esom_table[ self.esom_table.BinID != self.bin_name ].ContigName )

    @property
    def centroid(self):

        df = self.esom_table[ self.esom_table.BinID == self.bin_name ]
        x = np.median(df.V1)
        y = np.median(df.V2)
        return (x, y)

    @property
    def top_key(self):

        top_slice = pd.DataFrame( self._iteration_scores ).nlargest(1, 'MCC', keep='all')

        if top_slice.shape[0] == 1:
            return top_slice.Key.values[0]

        else:
            return top_slice.sort_index(ascending=False).Key.values[0]

    @property
    def mcc_sequence(self):
        return pd.DataFrame( self._iteration_scores )['MCC']

    @property
    def polsby_popper_sequence(self):

        '''
            This is a metric that was originally described by Cox (1927; Journal of Paleontology 1(3): 179-183),
            but has more recently been re-discovered in poltical science by Polsbsy an Popper.
            
            It is simply a measure of difference between a shape and a circle of similar size, giving a measure
            on how compact the shape is (0 = no compactness, 1 = ideal).
        '''
        df = pd.DataFrame( self._iteration_scores )
        return [ 4 * np.pi * area / perimeter ** 2 for perimeter, area in zip(df.Perimeter, df.Area) ]

    @property
    def contig_core(self):

        if self._contig_core:
            return list( self._contig_core )
        
        else:
            return []

    #endregion

    #region Output functions

    def _split_contig_sets(self):

        top_key = self.top_key

        ''' Find the contigs and contamination fragments inside the core '''
        core_contigs = set( self.slice_contigs_bin(top_key).ContigName )
        contam_contigs = set (self.slice_contigs_nonbin(top_key).ContigName )

        ''' Find the contigs and contamination fragments outside the core '''
        outsider_contigs = set( self.bin_contigs ) - core_contigs
        outsider_contams = set( self.nonbin_contigs ) - contam_contigs

        return core_contigs, contam_contigs, outsider_contigs, outsider_contams

    def PlotScatter(self):

        ''' Define a colour vector for plotting:
                1. Core contigs = #1f78b4
                2. Contigs outside core = #a6cee3
                3. Contamination contigs (core) = #e31a1c
                4. Contamination contigs (outside) = #fb9a99
        '''
        core_contigs, contam_contigs, outsider_contigs, outsider_contams = self._split_contig_sets()

        ''' Clear the plot, if anything came before this call '''
        plt.clf()

        for contig_set, colour, set_name in zip([outsider_contams, outsider_contigs, core_contigs, contam_contigs],
                                                        ['#fb9a99', '#a6cee3', '#1f78b4', '#e31a1c'],
                                                        ['Contamination (outside core)', 'Remainder', 'Core', 'Contamination (inside core)']):
            df = self.esom_table[ self.esom_table.ContigName.isin(contig_set) ]
            plt.scatter(df.V1, df.V2, c=colour, label=set_name)

        plt.legend()
        plt.savefig('{}.scatter.png'.format(self.output_path), bbox_inches='tight')

    def PlotTrace(self):

        ''' Clear the plot, if anything came before this call '''
        plt.clf()
        fig, ax = plt.subplots()

        x_vals = [ x + 1 for x in range(0, len(self._iteration_scores) ) ]

        ''' Plot the MCC against the left x-axis '''
        ax.plot(x_vals, self.mcc_sequence, color='g', label='MCC')
        ax.set_xlabel('Contig slice')
        ax.set_ylabel('MCC')
        ax.set_ylim([0.0, 1.0])

        ''' Plot the PPP against the right x-axis '''
        ax2 = ax.twinx()
        ax2.plot(x_vals, self.polsby_popper_sequence, color='r', label='PP value')
        ax2.set_ylabel('Polsby-Popper value')
        ax2.set_ylim([0.0, 1.0])

        fig.legend(loc='upper right')
        plt.savefig('{}.trace.png'.format(self.output_path), bbox_inches='tight')

    def SaveMccTable(self):

        ''' Write out the iteration table, in case users are interested '''
        output_name = '{}.mcc.txt'.format(self.output_path)

        out_df = pd.DataFrame( self._iteration_scores ).drop('Key', axis=1)
        out_df['Slice'] = [ i + 1 for i in range( 0, out_df.shape[0] ) ]
        out_df.to_csv(output_name, index=False, sep='\t')

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

    @staticmethod
    def CreateCoreTable(esom_table_name, genome_bin_list):

        binMap = []

        for genome_bin in genome_bin_list:
            if genome_bin.contig_core:

                binMap.extend( [ { 'Bin': genome_bin.bin_name, 'ContigBase': c } for c in genome_bin.contig_core ] )

            else:
                print( 'No contigs remain in {}, skipping...'.format(genome_bin.bin_name) )
                continue

        ''' Set the output file '''
        output_name = '{}.core_table.txt'.format( os.path.splitext(esom_table_name)[0] )
        pd.DataFrame(binMap).to_csv(output_name, sep='\t', index=False)

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
    #def ResolveContaminationByAbundance(bin_instance_dict, contam_table, contam_counter, bias_threshold):
    def ResolveContaminationByAbundance(arg_tuple):

        '''
            Analyse a single contig and adds a 3-tuple of return information to the manager queue
                1. Name of the contig
                2. Bin to assign the contig to, None if no contig passes the threshold
                3. Bins that the contig is seen in, that it must be removed from
        '''
        (contig, contig_fragments, bin_instance_dict, contam_table, bias_threshold, q) = arg_tuple
        contam_df = contam_table[ contam_table.ContigBase == contig ]

        ''' Count the number of fragments in the original bin, and the total number of fragments '''
        current_bin = contam_df.CurrentBin.values[0]
        fragments_in_main_bin = bin_instance_dict[ current_bin ].count_contig_fragments(contig)

        ''' Count the number of contamination fragments in the bin holding the most fragments '''
        contam_bin_dict = dict( Counter( contam_df.ContaminationBin ) )
        top_contam_bin = max(contam_bin_dict.items(), key=operator.itemgetter(1))[0]

        bin_to_keep = None
        bins_to_drop = list( contam_bin_dict.keys() )

        if contig == 'ContigA_88':
            print(contig)
            print(current_bin)
            print(contam_bin_dict)
            print( 'In: {}, Total: {}, Freq: {}'.format(fragments_in_main_bin, contig_fragments, bias_threshold) )
            print(contam_df)
            import sys; sys.exit()

        ''' Flow control here:

            1. If the original bin possesses most of the fragments, pass through, otherwise append it to the drop list
            2. If the top 'contamination' bin has most of the fragments, pop it from the drop list and append the original bin into the contamination list
            3. Drop the contig pieces from the contam_bins
        '''

        if float( fragments_in_main_bin ) / contig_fragments >= bias_threshold:

            bin_to_keep = current_bin
            print( 'Keep: {}'.format(bin_to_keep) )

        elif float( contam_bin_dict[top_contam_bin] ) / contig_fragments >= bias_threshold:

            bin_to_keep = top_contam_bin
            print( 'Contam: {}'.format(bin_to_keep) )

            bins_to_drop.remove(top_contam_bin)
            bins_to_drop.append(current_bin)
        
        else:

            bins_to_drop.append( current_bin )
        
        ''' Store the results into the Manager.Queue'''
        print(bins_to_drop)
        print('')

        q.put( (contig, bin_to_keep, bins_to_drop) )