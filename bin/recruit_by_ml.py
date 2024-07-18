import os
from typing import Any, List, Tuple
from optparse import OptionParser
from dataclasses import dataclass, field
from sklearn import clone, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
import numpy as np
import polars as pl

@dataclass
class ValidationSet:

    identity_mask: list[bool] = field(default_factory=list)

    def __init__(self, training: np.ndarray, testing: np.ndarray) -> 'ValidationSet':

        self.identity_mask = [x in training for x in range(len(training) + len(testing))]

    def __eq__(self, other: 'ValidationSet') -> bool:

        return self.identity_mask == other.identity_mask

    def extract_dataframe_training(self, df: pl.DataFrame) -> Tuple[List[str], List[str],pl.DataFrame]:

        filt_df = df.filter(self.identity_mask)
        return (
            filt_df.get_column('Source').to_list(),
            filt_df.get_column('Fragment').to_list(),
            filt_df.select('TSNE_1', 'TSNE_2')
        )

    def extract_dataframe_testing(self, df: pl.DataFrame) -> Tuple[List[str], List[str], pl.DataFrame]:

        filt_df = df.filter([not x for x in self.identity_mask])
        return (
            filt_df.get_column('Source').to_list(),
            filt_df.get_column('Fragment').to_list(),
            filt_df.select('TSNE_1', 'TSNE_2')
        )

def main():

    # Set up options and models to load/create
    parser = OptionParser()

    parser.add_option('-i', '--input', help='Input table produced by theproject_ordination.py')
    parser.add_option('-o', '--output', help='The output folder for models, training data, and classification')

    # Machine-learning parameters
    parser.add_option('-s', '--seed', help='Random seed for data permutation and model building', dest='seed', default=None)
    parser.add_option('-t', '--threads', help='Number of threads to use for model training and classification (Default: 1)', dest='threads', default=1)
    parser.add_option('--cross_validate', help='Perform X-fold cross validation in model building (Default: 10)', default=10)
    parser.add_option('--random_forest', help='Create and apply a random forest model')
    parser.add_option('--neural_network', help='Create and apply a neural network model')
    parser.add_option('--svm_linear', help='Create and apply a SVM (linear kernal) model')
    parser.add_option('--svm_radial', help='Create and apply a SVM (radial basis function kernal) model')
    
    ''' Machine-learning parameters - classifier-specific '''
    #parser.add_option('--rf-trees', help='Number of decision trees for random forest classification (Default: 1000)', dest='rf_trees', default=1000)
    #parser.add_option('--nn-nodes', help='Comma-separated list of the number of neurons in the input, hidden, and output layers (Default: Input = number of features + 1, Output = number of classification outcomes, Hidden = mean of Input and Output)', dest='nn_nodes', default=None)

    options, _ = parser.parse_args()

    # Prepare the data - extract the core contigs then separate into training and testing inputs.
    core_df, recruit_df = extract_core_contigs(options.input)
    validation_data = create_training_sets(core_df.get_column('Source'), options.cross_validate, seed=options.seed)

    """
import polars as pl
from sklearn import clone, svm
from bin.recruit_by_ml import ValidationSet
from bin.recruit_by_ml import extract_core_contigs
from bin.recruit_by_ml import create_training_sets
core_df, recruit_df = extract_core_contigs('results/raw_bins.tsne_core.parquet')
validation_data = create_training_sets(core_df.get_column('Source'), 10, 12345)

for training_set in validation_data:
    break

training_labels, training_fragments, training_data = training_set.extract_dataframe_training(core_df)
testing_labels, testing_fragments, testing_data = training_set.extract_dataframe_testing(core_df)
    """

    # Test for existance of previous models, or load the queue of models to produce
    new_models = []

def train_models(core_df: pl.DataFrame, training_sets: List[ValidationSet], model_instances: List[Any], **kwargs) -> Any:
    """ Test for the existance of the specified model file. If it does not exist, train a new model and
        save the results to the specified path.
    """

    model_buffer = []
    score_buffer = []
    for training_set, model_instance in zip(training_sets, model_instances):

        training_labels, training_fragments, training_data = training_set.extract_dataframe_training(core_df)
        testing_labels, testing_fragments, testing_data = training_set.extract_dataframe_testing(core_df)

        model_instance.fit(training_data, training_labels)

        x = model_instance.predict_proba(testing_data)
        proba_df = package_predictions(x, model_instance.classes_, testing_labels, testing_fragments)
        prediction_df = proba_to_decisions(proba_df)

        model_buffer.append(model_instance)
        score_buffer.append(matthews_corrcoef(prediction_df['Bin'], prediction_df['Prediction']))

    # Find the top-scoring model and return
    i = score_buffer.index(max(score_buffer))
    return model_buffer[i]

def package_predictions(proba: 'ndarray', model_classes: List[str], labels: List[str], fragments: List[str]) -> pl.DataFrame:
    """ Take the results of a models `predict_proba()` function and apply column names, as well as bin and fragment
        names to each entry.
    """

    proba_df = pl.DataFrame(proba)
    proba_df.columns = model_classes
    return (
        proba_df.with_columns(
            pl.Series('Bin', labels),
            pl.Series('Fragments', fragments)
        )
    )

def proba_to_decisions(proba_df: pl.DataFrame) -> pl.DataFrame:
    """ Take the output of the `package_predictions()` function and covert to a format with the top
        score for each fragment.
    """

    return (
        proba_df
        .melt(id_vars=['Fragments', 'Bin'], variable_name='Prediction', value_name='Score')
        .group_by(['Fragments'])
        .agg(pl.all().sort_by('Score', descending=True).first())
        .with_columns(
            Matched=(pl.col('Bin') == pl.col('Prediction'))
        )
    )

    #MLPClassifier(hidden_layer_sizes=layerSizes, max_iter=1000, activation='relu', solver='adam', random_state=seed)

    """
    def instantiate_svm_linear(self, seed = None):
        self._model_base['SVML'] = self._instantiate_svm('linear', seed)

    def instantiate_svm_rbf(self, seed = None):
        self._model_base['SVMR'] = self._instantiate_svm('rbf', seed)

    def instantiate_svm_polynomial(self, seed = None):
        self._model_base['SVMP'] = self._instantiate_svm('poly', seed)
    """

def extract_core_contigs(file_path: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """ Read the input file and separate into the core contigs, and the contigs to be classified.
    """
    return (
        pl.scan_parquet(file_path).filter(pl.col('Core')).collect(),
        pl.scan_parquet(file_path).filter(~pl.col('Core')).collect()
    )

def create_training_sets(source_data: List[str], n_splits: int, seed=None) -> List[ValidationSet]:
    """ Apply the sklearn StratifiedShuffleSplit to produce a user-supplied number of
        training/test data sets. Does not consider the features themselves, only the
        classification state (y in the [X, y] sklearn notation).
        Returns lists of the training/test masks to apply to the input data.
    """

    test_size = 0.1
    dummy_X = np.zeros(len(source_data))

    stratifier = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    return [ValidationSet(train, test) for (train, test) in stratifier.split(dummy_X, source_data)]

    """
    def _parse_layer_choices(neuronString, esom_core):
    if neuronString:
        try:
            assert(',' in neuronString), 'Unable to split neuron string'
            neuronVals = neuronString.split(',')
            assert(len(neuronVals) == 3), 'Incorrect number of layers described'
            neuronVals = tuple( map(int, neuronVals) )
            return neuronVals
        except AssertionError as ae:
            print( '{}. Aborting...'.format(ae) )
            sys.exit()
        except:
            print( 'Unable to convert values to integer. Aborting...' )
            sys.exit()

    ''' Otherwise, infer from the data '''
    input_layer = esom_core.scaled_features.shape[1]
    output_layer = len( esom_core.original_bin.unique() )
    hidden_layer = (input_layer + output_layer) / 2

    return input_layer, int(hidden_layer), output_layer
    """


    #machineModelController = MachineController(options.models, output_file_stub, options.reload, options.nn_nodes, options.rf_trees, options.threads, options.seed)

    #if not options.reload:
    #    validation_confidence_list = machineModelController.train_models(options.cross_validate, esom_core, options.seed)
    #    report_classification(validation_confidence_list, 'core_validation', output_file_stub)

    #if not options.reload:
    #    machineModelController.save_models()
    #    machineModelController.report_training()
    #    if options.evaluate_only: sys.exit()

    #classification_result = machineModelController.classify_by_ensemble(esom_cloud)
    #classification_result['ContigBase'] = esom_cloud.contig_base
    #classification_result['ContigName'] = esom_cloud.contig_fragments

    #report_classification(classification_result, 'contig_classification', output_file_stub)

if __name__ == '__main__':
    main()