import argparse
from sklearn import preprocessing
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from typing import List
import polars as pl

def main():

    # Parse options
    parser = argparse.ArgumentParser()

    parser.add_argument('-k', '--kmer', help='K-mer count table as input file')
    parser.add_argument('-c', '--coverage', help='Contig coverage table as input file')
    parser.add_argument('--store_features', help='File path to store feature table (optional)')
    parser.add_argument(
        '--load_features', action='store_true',
        help='Load a pre-computed feature table as input file (optional, skips --kmer and --coverage)'
    )

    parser.add_argument(
        '-n', '--normalise', choices=['unit', 'yeojohnson', 'none'], default='unit',
        help=(
            'Method for normalising per-column values (Options: unit variance (\'unit\'), '
            'Yeo-Johnson (\'yeojohnson\'), None (\'none\'). Default: unit)'
        ),
    )
    parser.add_argument(
        '-w', '--weighting', type=float, default=None,
        help='Assign over- or under-representation to the depth columns (Default: uniform weighting)'
    )
    parser.add_argument('-o', '--output', help='File path to store projected coordinates of the contigs')

    options = parser.parse_args()

    # Either produce a kmer frequency table from the count data, or read a previously saved version
    if options.load_features:
        freq_matrix = pl.read_csv(options.freq, separator='\t')

    else:

        # Load and normalise the k-mer count table
        kmer_df = pl.read_parquet(options.kmer)
        freq_df = counts_to_frequencies(kmer_df)

        # Import the coverage information if required, and append to the k-mer frequency table
        if options.coverage:
            coverage_df = map_coverage_to_fragments(pl.scan_parquet(options.coverage), freq_df)
            freq_df = pl.concat([freq_df, coverage_df], how='vertical')

        # Project the frequency table into wide format
        freq_matrix = project_to_matrix(freq_df)

    # Save, if requested
    if options.store_features:
        freq_matrix.write_csv(options.store_features, separator='\t')

    # Separate out the labels from the count data, then normalise
    label_df = freq_matrix.select('Source', 'Contig', 'Fragment')
    features_df = freq_matrix.drop('Source', 'Contig', 'Fragment')

    # Normalise, if requested
    if options.normalise:
        features_df = normalise_matrix(features_df, options.normalise)

    # Compute a (weighted) distance matrix, then project the TSNE coodinates and save the result
    dist_matrix = compute_dist(features_df, options.weighting)

    tsne_df = project_tsne(dist_matrix, label_df)
    tsne_df.write_parquet(options.output)

# region Normalisation and transformation

def map_coverage_to_fragments(coverage_frame: pl.LazyFrame, kmer_df: pl.DataFrame) -> pl.DataFrame:
    """ Manipulate a coverage file and join according to the contigs and fragments of
        the k-mer frequency plot.

        Arguments:
        coverage_frame -- a pl.LazyFrame representation of mapping data to be incorporated into the
                          ordination
        kmer_df        -- a pl.DataFrame from which the contigs of interest can be extracted
    """

    coverage_df = (
        coverage_frame
        .filter(
            pl.col('Contig').is_in(kmer_df.get_column('Contig'))
        )
        .collect()
    )

    return (
        kmer_df
        .select('Source', 'Contig', 'Fragment')
        .join(coverage_df, how='left', on='Contig')
        .unique()
        .rename({'Label': 'Feature', 'Coverage': 'Value'})
        .select('Source', 'Contig', 'Fragment', 'Feature', 'Value')
    )

def counts_to_frequencies(df: pl.DataFrame) -> pl.DataFrame:
    """ Scale the k-mer counts within each fragment to frequency values and relabel the columns
        to the form ['Source', 'Contig', 'Fragment', 'Feature', 'Value'].

        Arguments:
        df -- a pl.DataFrame of k-mer counts to be normalised as frequencies along their respective
              fragment
    """

    return (
        df
        .with_columns(
            (pl.col('Count') / pl.col('Count').sum()).over('Fragment').alias('Value'),
            pl.col('Kmer').map_elements(lambda x: f"Freq_{x}", return_dtype=pl.Utf8).alias('Feature')
        )
        .select('Source', 'Contig', 'Fragment', 'Feature', 'Value')
    )

def project_to_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """ Pivot a data frame of form ['Source', 'Contig', 'Fragment', 'Feature', 'Value'], projecting
        the Feature column to columns with Value as the cells values. Fills any gaps with zeroes.

        Arguments:
        df -- a pl.DataFrame to be pivoted into wide form
    """

    feature_order = (
        df
        .get_column('Feature')
        .unique()
        .sort()
        .to_list()
    )

    return (
        df
        .pivot(
            index=['Source', 'Contig', 'Fragment'],
            on='Feature',
            values='Value'
        )
        .fill_null(0)
        .select(['Source', 'Contig', 'Fragment'] + feature_order)
    )

def normalise_matrix(df: pl.DataFrame, method: str) -> pl.DataFrame:
    """ Take a data frame of numberic values and scale according to the method provided ('unit' or
        'yeojohnson').

        Arguments:
        df     -- the pl.DataFrame of k-mer frequencies to be normalised
        method -- the normalisation method to be applied to the data
    """

    if method == 'unit':
        norm_data = preprocessing.scale(df, axis=0)

    elif method == 'yeojohnson':
        pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
        norm_data = pt.fit_transform(df)

    norm_df = pl.DataFrame(norm_data)
    norm_df.columns = df.columns
    return norm_df

# endregion

# region Clustering

def build_weighting_mask(col_names: List[str], cov_weight: float) -> List[float]:
    """ Create a weighting mask for the data frame columns, splitting the specified weight over
        the Depth_* columns, and the remainder of the weight over the remaining (frequency) columns.

        Arguments:
        col_names  -- a list of all column names in the data
        cov_weight -- the weighting to be split between the coverage (Depth_*) columns in the data
                      frame. Residual weighting is shared equally between all k-mer frequencies columns
    """

    # Find the indices of the coverage columns
    mapping_locs = [x.startswith("Depth_") for x in col_names]

    # Count the number of composition and coverage columns
    n_comp = mapping_locs.count(False)
    n_cov = mapping_locs.count(True)

    # Convert the boolean location vector into weightings according to whether the column starts with
    # 'Depth' or not.
    comp_weight = 1.0 - cov_weight
    return [cov_weight / n_cov if x else comp_weight / n_comp for x in mapping_locs]

def compute_dist(feature_df: pl.DataFrame, weight_coverage: float=None) -> pl.DataFrame:
    """ Compute the Euclidean distances between the rows of the input data, optionally weighting according
        to the user specification. Distances are transformed to a square matrix and converted to pl.DataFrame.

        Arguments:
        feature_df      -- the pl.DataFrame of features to be used in the ordination distance calculation
        weight_coverage -- (optional) the weighting to be assigned to coverage, if an even division between
                           all (frequency and depth) columns is not desired
    """

    if weight_coverage:

        weight_mask = build_weighting_mask(feature_df.columns, weight_coverage)
        d = pdist(feature_df, 'euclidean', w=weight_mask)

    else:
        d = pdist(feature_df, 'euclidean')

    return pl.DataFrame(squareform(d))

def project_tsne(dist_matrix: pl.DataFrame, label_df: pl.DataFrame, perplexity: int=None, seed: int=None) -> pl.DataFrame:
    """ Project the distance matrix into a randomly-initialised TSNE, then bind in the original columns to
        create the final output. Accepts optional parameters which can be passed for controlling behaviour during
        unit testing.

        Arguments:
        dist_matrix -- a numeric matrix between points for the initial data projection
        label_df    -- a pl.DataFrame of text values matching the row order of dist_matrix
        perplexity  -- (optional) a user-specified perplexity value for the TSNE transformer
        seed        -- (optional) a user-specified random_state for the TSNE transformer
    """

    tsne_kwargs = {
        'n_components': 2,
         'metric': 'precomputed',
         'init': 'random'
    }

    if perplexity:
        tsne_kwargs['perplexity'] = perplexity

    if seed:
        tsne_kwargs['random_state'] = seed

    tsne_df = pl.DataFrame(
        TSNE(**tsne_kwargs)
        .fit_transform(dist_matrix)
    )

    return (
        pl
        .concat([label_df, tsne_df], how='horizontal')
        .rename({'column_0': 'TSNE_1', 'column_1': 'TSNE_2'})
    )

#endregion

if __name__ == '__main__':
    main()
