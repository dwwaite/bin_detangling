import os, uuid, operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.metrics import matthews_corrcoef

class GenomeBin:

    def __init__(self, bin_name, esom_path, bias_threshold, number_of_slices, output_path):

        self.bin_name = bin_name
        self.output_path = output_path
        self.bias_threshold = bias_threshold
        self.number_of_slices = number_of_slices 

        self._prepare_esom_df(esom_path)
        
        self.mcc_expectation = [ 1.0 if contig_bin == self.bin_name else 0.0 for contig_bin in self.esom_table.BinID ]

        ''' Add variables for internal use:
            iteration_scores creates a pd.DataFrame of numeric metrics
            slice, and contam_contigs is a NoSQL-like storage of complex objects '''
        self._iteration_scores = []
        self._slice_df_lookup = {}

        self._core_contigs = None

    #region Externally exposed functions

    def ComputeCloudPurity(self):

        for frame_slice in self._get_next_slice():

            ''' Create a UUID for storing contig records '''
            slice_key = uuid.uuid4()          

            ''' Calculate the MCC '''
            slice_mcc = self._compute_mcc(frame_slice)

            ''' Record the perimeter and area of the point slice.
                Note that these are special cases of the ConvexHull for a 2D shape. If we project to more dimensions, this will no longer be valid '''
            q_hull = ConvexHull( frame_slice[ ['V1', 'V2'] ].values )
            slice_area = q_hull.volume
            slice_perimeter = q_hull.area

            self._store_quality_values(slice_mcc, slice_key, slice_area, slice_perimeter, frame_slice )

    def ResolveUnstableContigs(self, fragment_count_dict, qManager):

        ''' Find the key that corresponds to the top MCC, then cast out a list of the ContigBase names within this '''
        top_df = self._slice_df_lookup[ self.top_key ]

        top_contigs = top_df[ top_df.BinID == self.bin_name ].ContigBase.unique()

        ''' For each fo these contigs, remove it if it does not pass the bias_threshold '''
        core_contigs = set(top_contigs)
        for contig in top_contigs:

            contig_fragment_bin_dist = top_df[ top_df.ContigBase == contig ].BinID.value_counts()

            self_fragments = contig_fragment_bin_dist[ self.bin_name ]
            total_fragments = fragment_count_dict[contig]

            if float(self_fragments) / total_fragments < self.bias_threshold:
                core_contigs.remove(contig)

        ''' Store dicts of bin/contig, for return to the main process '''
        for c in core_contigs:
            qManager.put( { 'Bin': self.bin_name, 'Contig': c } )

    #endregion

    #region Internal manipulation functions

    def _prepare_esom_df(self, esom_path):

        ''' Prepare the internal DataFrame of ESOM coordinates...
            Import the ESOM table'''
        self.esom_table = pd.read_csv(esom_path, sep='\t')

        ''' Identify the centroid of the bin and measure the distance from each point to the centroid.
            Finally, sort ascending '''
        cenX, cenY = self.centroid

        self.esom_table['Distance'] = [ self._calc_dist(cenX, cenY, x, y) for x, y in zip(self.esom_table.V1, self.esom_table.V2) ]
        self.esom_table = self.esom_table.sort_values('Distance', ascending=True)

        ''' Slice the table to the last binned contig
            Reset the index '''
        last_contigfragment = max(idx for idx, val in enumerate( self.esom_table.BinID ) if val == self.bin_name) + 1
        self.esom_table = self.esom_table.head(last_contigfragment)
        self.esom_table.reset_index(drop=True, inplace=True)

    def _calc_dist(self, xCen, yCen, xPos, yPos):
        return np.sqrt( (xCen - xPos) ** 2 + (yCen - yPos) ** 2 )

    def _get_next_slice(self):

        ''' Pull the indices for the contig fragments in the bin '''
        index_list = list( self.esom_table[ self.esom_table.BinID == self.bin_name ].index )
        n_contigs = len(index_list)

        ''' Divide the index list into N slices, following a sine function '''
        for x in range(1, self.number_of_slices + 1):

            slice_index = np.sin( x / self.number_of_slices * np.pi/2 ) * n_contigs
            slice_index = int(slice_index)

            ''' Break the loop if we've hit the end of the curve early.
                This can happen for the last entry due to int rounding of the index '''
            if slice_index >= n_contigs:
                yield self.esom_table
                return

            curr_slice = index_list[ slice_index ]
            yield self.esom_table.iloc[ 0:curr_slice, ]

        yield self.esom_table

    def _compute_mcc(self, slice_df):

        obs_vector = [0.0] * len( self.mcc_expectation )
        obs_vector[ 0 : slice_df.shape[0] ] = [1.0] * slice_df.shape[0]

        return 0.0 if np.prod(obs_vector) == 1 else matthews_corrcoef(self.mcc_expectation, obs_vector)

    def _store_quality_values(self, mcc, slice_key, area, perimeter, frame_slice):

        self._iteration_scores.append( { 'MCC': mcc, 'Key': slice_key, 'Area': area, 'Perimeter': perimeter })
        self._slice_df_lookup[ slice_key ] = frame_slice

    def _slice_contigs_bin(self, slice_key):

        df = self._slice_df_lookup[ slice_key ]
        return df[ df.BinID == self.bin_name ]

    def _slice_contigs_nonbin(self, slice_key):

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

    #endregion

    #region Output functions

    def _split_contig_sets(self):

        top_key = self.top_key

        ''' Find the contigs and contamination fragments inside the core '''
        core_contigs = set( self._slice_contigs_bin(top_key).ContigName )
        contam_contigs = set (self._slice_contigs_nonbin(top_key).ContigName )

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
    def ParseStartingVariables(bin_names, esom_table_path, number_of_slices, bias_threshold, output_prefix = None):

        ''' Just a QoL method for pre-formating the input data for constructing GenomeBin objects.
            Allows for a nice way to terminate the script early if there are path issues, before getting to the actual grunt work '''

        return_tuples = []
        esom_table_bins = set( pd.read_csv(esom_table_path, sep='\t').BinID.unique() )

        for bin_name in bin_names:

            if bin_name in esom_table_bins:

                outputPath = output_prefix + bin_name + '.refined' if output_prefix else bin_name + '.refined'
                return_tuples.append( (bin_name, esom_table_path, bias_threshold, number_of_slices, outputPath) )

        return return_tuples

    @staticmethod
    def CreateCoreTable(esom_table_name, results_buffer_list):

        ''' Set the output file '''
        output_name = '{}.core_table.txt'.format( os.path.splitext(esom_table_name)[0] )

        ''' Save it '''
        bin_map = pd.DataFrame(results_buffer_list)
        bin_map.to_csv(output_name, sep='\t', index=False)

    #endregion