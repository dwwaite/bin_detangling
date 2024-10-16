import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import polars as pl
import plotly.express as px
from joblib import dump
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

@dataclass
class ValidationSet:

    identity_mask: List[bool] = field(default_factory=list)

    def __init__(self, training: List[int], testing: List[int]) -> 'ValidationSet':
        self.identity_mask = [x in training for x in range(len(training) + len(testing))]

    def __eq__(self, other: 'ValidationSet') -> bool:
        return self.identity_mask == other.identity_mask

@dataclass
class ValidationCollection:

    core_data: pl.DataFrame = None
    vs_collection: List[ValidationSet] = field(default_factory=list)

    def __init__(self, core_data: pl.DataFrame, n_validations: int, test_size: float=0.1, random_seed: int=None) -> 'ValidationCollection':

        self.core_data = core_data

        source_data = self.core_data.get_column('Source')
        if n_validations > 0:
            self.vs_collection = ValidationCollection.create_training_sets(
                source_data,
                n_validations,
                test_size=test_size,
                random_seed=random_seed
            )
        else:
            self.vs_collection = []

    def __eq__(self, other_vc: 'ValidationCollection') -> bool:

        df_match = self.core_data.equals(other_vc.core_data)
        sets_match = self.vs_collection == other_vc.vs_collection

        return df_match and sets_match

    def extract_training_set(self, i: int) -> Tuple[pl.DataFrame, pl.Series]:
        """ Return the training data and labels encoded in position i of the validation collection.

            Arguments:
            i -- the index of the training data to be returned
        """
        data = (
            self
            .core_data
            .filter(self.vs_collection[i].identity_mask)
            .select('TSNE_1', 'TSNE_2')
        )
        labels = (
            self
            .core_data
            .filter(self.vs_collection[i].identity_mask)
            .get_column('Source')
        )

        return (data, labels)

    def extract_testing_set(self, i: int) -> Tuple[pl.DataFrame, pl.Series]:
        """ Return the testing data and labels encoded in position i of the validation collection.

            Arguments:
            i -- the index of the training data to be returned
        """
        mask = [not x for x in self.vs_collection[i].identity_mask]

        data = (
            self
            .core_data
            .filter(mask)
            .select('TSNE_1', 'TSNE_2')
        )
        labels = (
            self
            .core_data
            .filter(mask)
            .get_column('Source')
        )

        return (data, labels)

    @staticmethod
    def import_core_data(data_path: str) -> pl.DataFrame:
        """ Reads and filters a data frame produced by `identify_bin_cores.py` for use in model training.
        """

        return (
            pl
            .scan_parquet(data_path)
            .filter(pl.col('Core'))
            .collect()
        )

    @staticmethod
    def create_training_sets(source_data: pl.Series, n_splits: int, test_size: float=0.1, random_seed: int=None) -> List[ValidationSet]:
        """ Applies the sklearn StratifiedShuffleSplit over the input data sequence to produce
            training/test data splits of the input data, according to the user input.
            Does not consider the features themselves, only the classification state (y in the [X, y]
            sklearn notation).

            Arguments:
            source_data -- a pl.Series of the classes to which the data belongs
            n_splits    -- the number of splits of the data to perform
            test_size   -- (optional) the fraction of test data (0 - 1)
            random_seed -- (optional) the random seed for splitting data
        """

        dummy_X = np.zeros(len(source_data))
        stratifier = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_seed)
        return [ValidationSet(train, test) for (train, test) in stratifier.split(dummy_X, source_data)]

def main():

    # Capture the user input and route accordingly
    parser = parse_user_arguments()

    arguments = parser.parse_args()
    arguments.func(arguments)

#region Input handler and subroutine parsers

def parse_user_arguments() -> Any:
    """ Parses the user input and assigns the functions that each subparser will call.

        Arguments:
        None
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Operations for train and applying recruitment models.')

    # Subparser for the training workflow
    training_parser = subparsers.add_parser('train', help='Train a new set of models for contig recruitment.')
    training_parser.set_defaults(func=training_workflow)

    training_parser.add_argument('-i', '--input', help='Input table produced by project_ordination.py')
    training_parser.add_argument('-o', '--output', help='The output folder for top performing models')

    training_parser.add_argument('-s', '--seed', default=None, help='Random seed for data permutation and model building')
    training_parser.add_argument('-t', '--threads', default=1, help='Number of threads to use for model training and classification (Default: 1)')
    training_parser.add_argument('--cross_validate', default=10, help='Perform X-fold cross validation in model building (Default: 10)')
    training_parser.add_argument(
        '--test_fraction', default=0.1,
        help=(
            'The fraction of data to be used for testing during cross-fold validation. '
            'Default: 0.1 which may be unstable with smaller data sets.'
        )
    )
    training_parser.add_argument('--neural_network', action='store_true', help='Create and apply a neural network model')
    training_parser.add_argument('--random_forest', action='store_true', help='Create and apply a random forest model')
    training_parser.add_argument('--svm_linear', action='store_true', help='Create and apply a SVM (linear kernal) model')
    training_parser.add_argument('--svm_radial', action='store_true', help='Create and apply a SVM (radial basis function kernal) model')

    # TO DO - model-specific arguments
    #parser.add_option('--rf-trees', help='Number of decision trees for random forest classification (Default: 1000)', dest='rf_trees', default=1000)
    #parser.add_option('--nn-nodes', help='Comma-separated sequence setting number of neurons in the input, hidden, and output layers', default=None)

    # Subparser for the recruitment workflow
    recruitment_parser = subparsers.add_parser('recruit', help='Apply a set of recruitment models to the data.')
    recruitment_parser.set_defaults(func=train_models)

    return parser

def training_workflow(args):
    """ Applies the workflow for the training routine.

        Arguments:
        args               -- arguments namespace passed from the argument parser
            input          -- the path for the input contig file
            output         -- the output folder path for all files to be created
            cross_validate -- number of data splits to perform fir model training
            test_fraction  -- the fraction of data to be used for model testing
            seed           -- random seed for training models
            neural_network -- perform neural network training
            random_forest  -- perform random forest training
            svm_linear     -- perform SVM training for linear function
            svm_radial     -- perform SVM training with radial basis function
    """

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Prepare the data - extract the core contigs then separate into training and testing inputs.
    input_data = ValidationCollection.import_core_data(args.input)
    model_data = ValidationCollection(input_data, args.cross_validate, test_size=args.test_fraction, random_seed=args.seed)

    if args.neural_network:
        clf_label = 'neural_network'
        clf_dict = instantiate_classifiers(
            MLPClassifier, clf_label, args.cross_validate,
            max_iter=1_000, activation='relu', solver='adam', random_state=args.seed
        )
        build_report_models(clf_dict, model_data, args.output, clf_label)

    if args.random_forest:
        clf_label = 'random_forest'
        clf_dict = instantiate_classifiers(
            RandomForestClassifier, clf_label, args.cross_validate,
            n_jobs=args.threads, random_state=args.seed
        )
        build_report_models(clf_dict, model_data, args.output, clf_label)

    if args.svm_linear:
        clf_label = 'svm_linear'
        clf_dict = instantiate_classifiers(
            SVC, clf_label, args.cross_validate,
            kernel='linear', gamma=0.001, C=100.0, probability=True, random_state=args.seed
        )
        build_report_models(clf_dict, model_data, args.output, clf_label)

    if args.svm_radial:
        clf_label = 'svm_rbf'
        clf_dict = instantiate_classifiers(
            SVC, 'svm_rbf', args.cross_validate,
            kernel='rbf', gamma=0.001, C=100.0, probability=True, random_state=args.seed
        )
        build_report_models(clf_dict, model_data, args.output, clf_label)

def recruit_models(args):
    print("NOT IMPLEMENTED YET!")

#endregion

#region I/O functions

def plot_scoring(data: pl.DataFrame, output_path: str):
    """ Plot the score data as grouped boxplots displaying the per-model scores.

        Arguments:
        data        -- the score DataFrame to be plotted
        output_path -- the location for data to be saved
    """
    melt_df = data.unpivot(
        index='Model',
        on=['Score_F1', 'Score_MCC', 'Score_ROC'],
        variable_name='Metric',
        value_name='Score'
    )

    fig = px.bar(
        x=melt_df['Model'], y=melt_df['Score'], color=melt_df['Metric'], barmode='group',
        title=f"Model training scores for classifiers",
        labels={
            'x': 'Model tested',
            'y': 'Score',
            'color': 'Metric'
        },
    )
    fig.write_html(output_path)

def save_scoring(data: pl.DataFrame, output_path: str):
    """ Save the score data as a tab-delimited table at the specified location.

        Arguments:
        data        -- the score DataFrame to be saved
        output_path -- the location for data to be saved
    """
    data.write_csv(output_path, separator='\t')

def save_models(models: Dict[str, Any], output_dest: str):
    """ Iterate the models in the selection and write out each one using joblib.dump.

        Arguments:
        models      -- the collection of models to be saved
        output_dest -- the location for data to be saved
    """
    for lbl, clf in models.items():
        dump(clf, f"{output_dest}/{lbl}.pkl")

#endregion

#region Model creation and training functions

def instantiate_classifiers(clf_func: Callable, clf_label: str, n_classifiers: int, **kwargs) -> Dict[str, Any]:
    """ Accepts a callable function and keyword arguments for instantiating a new sklearn classifier model.
        Creates a dictionary of models for competitive training/testing.

        Arguments:
        clf_func      -- any callable function which can be called using the values of **kwargs
        clf_label     -- a text label for forming the prefix of the dictionary keys
        n_classifiers -- the number of independent classifiers to create, forms the suffix of the dictionary keys
        **kwargs      -- keyword arguments specific to the classifier function
    """

    return {f"{clf_label}_{i}": clf_func(**kwargs) for i in range(0, n_classifiers)}

def score_models(model_label: str, lbl_true: List[str], lbl_pred: List[str], proba_matrix: np.ndarray) -> Dict[str, Any]:
    """ Calculates performance scores (F1, MCC, and ROC) for a model of interest and returns a dictionary capturing the statistics.

        Arguments:
        model_label  -- the name for the model being tested
        lbl_true     -- the sequence of correct classes for the test data
        lbl_pred     -- the sequence of predicted classes for the test data
        proba_matrix -- the probability matrix for each prediction
    """

    return {
        'Model': model_label,
        'Score_F1': f1_score(lbl_true, lbl_pred, average='weighted'),
        'Score_MCC': matthews_corrcoef(lbl_true, lbl_pred),
        'Score_ROC': roc_auc_score(lbl_true, proba_matrix, multi_class='ovr'),
    }

def train_models(model_instances: Dict[str, Any], model_data: ValidationCollection) -> pl.DataFrame:
    """ Iterates through the collection of model instances and trains each one of the provided training/testing data.

        Arguments:
        model_instances -- the collection of classification models to be trained
        model_data      -- the representation of the training/testing data splits
    """

    score_buffer = []
    for i, (model_label, model_instance) in enumerate(model_instances.items()):

        train_data, train_lbls = model_data.extract_training_set(i)
        test_data, test_lbls = model_data.extract_testing_set(i)

        model_instance.fit(train_data, train_lbls)
        predictions = model_instance.predict(test_data)
        proba_matrix = model_instance.predict_proba(test_data)

        score_buffer.append(score_models(model_label, test_lbls, predictions, proba_matrix))

    return pl.DataFrame(score_buffer)

def build_report_models(model_instances: Dict[str, Any], model_data: ValidationCollection, output_path: str, output_lbl: str):
    """ Collects the training and reporting for a sequence of classification models according to the user-provided inputs.
        Performance scores for each model are tabulated and reported as a data table and plot of the results. All trained
        models are written to the user specified path as pickle files.

        Arguments:
        model_instances -- the collection of classification models to be trained
        model_data      -- the representation of the training/testing data splits
        output_path     -- output path for all files to be created
        output_lbl      -- the file name (no extension) for the summary table and plot being created
    """

    score_df = train_models(model_instances, model_data)

    save_scoring(score_df, f"{output_path}/{output_lbl}.tsv")
    plot_scoring(score_df, f"{output_path}/{output_lbl}.html")
    save_models(model_instances, output_path)

#endregion

if __name__ == '__main__':
    main()
