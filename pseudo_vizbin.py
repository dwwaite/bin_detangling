'''
    Creates a tSNE ordination of MAG data, using composition and coverage data.
    
    Boradly speaking, this is similar to the functionality of VizBin, and creates VizBin-compatible output files. Key differences:
    
        1. Allows user to use more than a single coverage value
        2. Allows user to vary the weighting on coverage
        3. Does not prime the solution with a PCA
'''
import sys
import pandas as pd
from optparse import OptionParser
from sklearn import preprocessing
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

# My functions and classes
from scripts.OptionValidator import ValidateFile, ValidateFloat, ValidateDataFrameColumns

def main():

    # Parse options
    usageString = "usage: %prog [options] [feature profile table]"
    parser = OptionParser(usage=usageString)

    parser.add_option('-o', '--output', help='An output prefix for all generated files (Default: Inferred from kmer table)', dest='output', default=None)
    parser.add_option('-c', '--coverage', help='Prefix for coverage columns in the feature table (Default: Coverage)', dest='coverage', default='Coverage')
    parser.add_option('-s', '--scale', help='Unit scale features (Default: False)', dest='scale', default=False, action='store_true')
    parser.add_option('-w', '--weighting', help='Weight to distribute across coverage features, between 0 and 1 (Default: uniform weight across all features)', dest='weighting', default=None)

    options, args = parser.parse_args()
    feature_table_name = args[0]

    '''
        Pre-workflow overhead: Validation of user choices.
    '''
    ValidateFile(inFile=feature_table_name, fileTypeWarning='feature table', behaviour='abort')
    feature_table = read_and_validate_table(feature_table_name, options.coverage)

    if options.weighting:
        options.weighting = parse_and_validate_weighting(options.weighting)

    '''
        Real workflow

            1. Pop contigs column from the feature table, scale if required
            2. Calculate a distance matrix from the remaining features
            3. Solve a tSNE ESOM for the data
            4. Create output files
    '''

    contig_sequence = feature_table.pop('Contig')
    if options.scale:
        feature_table = preprocessing.scale(feature_table, axis=0)

    dist_matrix = compute_dist(feature_table, options.coverage, options.weighting)

    tsne_coords = TSNE(n_components=2, metric='precomputed').fit_transform(dist_matrix)

    write_output(feature_table_name, tsne_coords, options.output)

###############################################################################

# region Pre-workflow overhead

def read_and_validate_table(ftable_name, cov_prefix):

    ftable = pd.read_csv(ftable_name, sep='\t')

    # Assume there must be able least one coverage column
    ValidateDataFrameColumns(ftable, ['Contig', '{}1'.format(cov_prefix) ])

    return ftable

def parse_and_validate_weighting(weight_value):

    weight_value = ValidateFloat(userChoice=weight_value, parameterNameWarning='coverage weighting', behaviour='abort')

    if weight_value > 1.0:
        print('Error: Trying to weight coverage for more than 100% of data.')
        sys.exit()

    elif weight_value < 0:
        print('Error: Trying to weight coverage for less than 0% of data.')
        sys.exit()

    return weight_value

# endregion

# region Clustering

def compute_dist(ftable, cov_prefix, cov_weight):

    if cov_weight:

        weight_mask = build_weighting_mask(ftable.columns, cov_prefix, cov_weight)

        d = pdist(comp_table_unit, 'euclidean', w=weight_mask)

    else:
        d = pdist(feature_table, 'euclidean')

    return squareform(d)

def build_weighting_mask(col_names, cov_prefix, cov_weight):

    # Find the indices of the coverage columns
    mapping_locs = [ cov_prefix in x for x in col_names ]

    # Count the number of composition and coverage columns
    n_comp = mapping_locs.count(False)
    n_cov = mapping_locs.count(True)

    comp_weight = 1.0 - cov_weight

    return [ cov_weight / n_cov if x else comp_weight / n_comp for x in mapping_locs ]

# endregion

def write_output(ftable_name, tsne, output_name=None):

    if not output_name:

        output_name = os.path.splitext(ftable_name)[0] + '.vb_points.txt'

    pd.to_csv(tsne, sep=',', index=False, header=False)

###############################################################################
if __name__ == '__main__':
    main()