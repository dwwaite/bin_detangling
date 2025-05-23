import unittest
from dataclasses import dataclass, field
from itertools import chain, product
from typing import Any, List, Self

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

# Imported in batches refering to classes, training, recruitment
from bin.recruit_by_ml import ValidationCollection, ValidationSet
from bin.recruit_by_ml import instantiate_classifiers, train_models, score_models
from bin.recruit_by_ml import proba_to_dataframe, recruit_by_model, aggregate_scores
from bin.recruit_by_ml import extract_top_score, build_assignment_table, get_top_bin, finalise_recruitment

@dataclass
class MockModel:

    classes_: List[str] = field(default_factory=list)

    def __init__(self, value_a: Any=None, value_b: Any=None):
        self.value_a=value_a
        self.value_b=value_b
        self.classes_ = ['a', 'b', 'c']

    def __eq__(self, other):
        return self.value_a == other.value_a and self.value_b == other.value_b

    def fit(self, x, y):
        return

    def predict(self, x):
        # Matched to the expected data layout of the `test_train_models()` test.
        # First 2 are expected a, observed 1 x a, 1 x b
        # Second 2 are expected b, observed 1 x b, 1 x a
        # Third 2 are expected c, allboth c
        return np.array(['a', 'b', 'b', 'a', 'c', 'c'])

    def predict_proba(self, x):
        # Matched to the expected output of the `MockModel.predict()` function.
        return np.array([
            [1.0, 0.0, 0.0], # bin_a, True
            [0.2, 0.8, 0.0], # bin_a, False
            [0.2, 0.8, 0.0], # bin_b, True
            [0.9, 0.1, 0.0], # bin_b, False
            [0.0, 0.0, 1.0], # bin c, True
            [0.0, 0.0, 1.0], # bin c, True
        ])

class ValidationCollectionBuilder:
    """ Simple builder pattern for more clear construction of mock ValidationCollection
        instances during unit testing. In the real script these are calculated from
        so the builder is unneeded but when unit testing certain code paths it's easier
        to explicitly set variables.
    """

    vc: ValidationCollection = None

    def __init__(self) -> 'ValidationCollectionBuilder':
        self.vc = ValidationCollection(pl.DataFrame([pl.Series('Source', [])]), 0)

    def add_data_core(self, df: pl.DataFrame) -> Self:
        self.vc.core_data = df
        return self

    def add_validation_set(self, identity_mask: List[bool]) -> Self:
        self.vc.vs_collection.append(TestProjectOrdination.spawn_validation_set(identity_mask))
        return self

    def build(self) -> ValidationCollection:
        return self.vc

class TestProjectOrdination(unittest.TestCase):

    @staticmethod
    def spawn_validation_set(mask_sequence: List[bool]) -> ValidationSet:
        """ Helper function to return a ValidationSet with a manually-defined identity
            mask.
        """
        vs = ValidationSet([], [])
        vs.identity_mask = mask_sequence
        return vs

#region ValidationCollection and ValidationSet classes

    def test_ValidationSet_init(self):
        """ Tests the bevahiour of the ValidationSet constructor. """

        exp_vs = TestProjectOrdination.spawn_validation_set([True, True, True, False, False])
        obs_vs = ValidationSet([0, 1, 2], [3, 4])
        self.assertEqual(exp_vs, obs_vs)

    def test_ValidationSet_init_shuffled(self):
        """ Tests the bevahiour of the ValidationSet constructor when the partitions are not
            in linear order.
        """

        exp_vs = TestProjectOrdination.spawn_validation_set([True, True, False, False, True])
        obs_vs = ValidationSet([0, 1, 4], [2, 3])
        self.assertEqual(exp_vs, obs_vs)

    def test_ValidationCollection_create_training_sets(self):
        """ Tests the behaviour of the ValidationCollection.create_training_sets() static function.
            This is called by the class constructor, so needs to be validated before instantiated
            object tests are performed.
        """

        source = pl.Series('Source', ['bin_a', 'bin_a', 'bin_a', 'bin_a', 'bin_a', 'bin_b', 'bin_b', 'bin_b', 'bin_b', 'bin_b'])

        vs_a = TestProjectOrdination.spawn_validation_set([False, True, True, False, True, True, False, True, False, False])
        vs_b = TestProjectOrdination.spawn_validation_set([False, True, False, False, True, True, False, True, True, False])

        exp_sets = [vs_a, vs_b]
        obs_sets = ValidationCollection.create_training_sets(source, 2, test_size=0.5, random_seed=1)
        self.assertEqual(exp_sets, obs_sets)

    def test_ValidationCollection_import_core_data(self):
        """ Tests the behaviour of the ValidationCollection.import_core_data() static function.
        """

        exp_df = pl.DataFrame([
            pl.Series('Source', ['bin_a', 'bin_a', 'bin_a', 'bin_a', 'bin_b', 'bin_b', 'bin_b', 'bin_b', 'bin_c', 'bin_c', 'bin_c', 'bin_c', 'bin_c']),
            pl.Series('Contig', ['a_1', 'a_1', 'a_1', 'a_2', 'b_1', 'b_1', 'b_2', 'b_3', 'c_1', 'c_1', 'c_1', 'c_1', 'c_1']),
            pl.Series('Fragment', ['a_10', 'a_11', 'a_12', 'a_20', 'b_10', 'b_11', 'b_20', 'b_30', 'c_10', 'c_11', 'c_12', 'c_13', 'c_14']),
            pl.Series('TSNE_1', [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15]),
            pl.Series('TSNE_2', [15, 14, 13, 12, 10, 9, 8, 6, 5, 4, 3, 2, 1]),
            pl.Series('Core', [True, True, True, True, True, True, True, True, True, True, True, True, True]),
        ])

        obs_df = ValidationCollection.import_core_data('tests/test_recruit_by_ml.parquet')
        assert_frame_equal(exp_df, obs_df)

    @staticmethod
    def spawn_core_frame(classes: List[str], n_samples: int) -> pl.DataFrame:
        """ Helper function to return a data frame mocking a the results of a filtered core data set
            sufficienctly sampled for the splitting/training operations. Only the required columns for
            testing and verifying results are captured.
        """
        
        class_chain = [
            [c] * n_samples for c in classes
        ]
        sources = [s for s in chain.from_iterable(class_chain)]

        sample_iter = range(0, n_samples)
        fragments = [f"{c}_{s}" for c, s in product(classes, sample_iter)]

        n_datapoints = len(sources)
        return pl.DataFrame([
            pl.Series('Source', sources),
            pl.Series('Fragment', fragments),
            pl.Series('TSNE_1', [i for i in range(0, n_datapoints)]),
            pl.Series('TSNE_2', [i for i in range(n_datapoints, 0, -1)]),
            pl.Series('Core', [True] * n_datapoints),
        ])

    def test_ValidationCollection_init(self):
        """ Tests the behaviour of the ValidationCollection constructor. """

        input_df = TestProjectOrdination.spawn_core_frame(['a', 'b', 'c'], 10)
        exp_vc = (
            ValidationCollectionBuilder()
                .add_data_core(input_df)
                .add_validation_set([
                    True, False, True, False, True, False, True, False, False, True,
                    True, False, False, True, False, True, False, False, True, True,
                    True, False, False, True, False, True, False, False, True, True
                ])
                .add_validation_set([
                    False, False, False, False, True, True, True, True, True, False,
                    True, False, True, False, True, True, False, False, False, True,
                    False, True, False, False, True, True, False, False, True, True
                ])
                .build()
        )

        obs_vc = ValidationCollection(input_df, 2, test_size=0.5, random_seed=1)
        self.assertEqual(exp_vc, obs_vc)

    def test_ValidationCollection_extract_training_set(self):
        """ Tests the behaviour of the ValidationCollection.extract_training_set() function.
        """

        df = TestProjectOrdination.spawn_core_frame(['a', 'b'], 6)
        vc = (
            ValidationCollectionBuilder()
                .add_data_core(df)
                .add_validation_set([True, False, True, False, True, False, True, False, True, False, True, False])
                .add_validation_set([False, True, False, True, False, True, False, True, False, True, False, True])
                .build()
        )

        exp_dfs = [
            pl.DataFrame([pl.Series('TSNE_1', [0, 2, 4, 6, 8, 10]), pl.Series('TSNE_2', [12, 10, 8, 6, 4, 2])]),
            pl.DataFrame([pl.Series('TSNE_1', [1, 3, 5, 7, 9, 11]), pl.Series('TSNE_2', [11, 9, 7, 5, 3, 1])]),
        ]

        exp_labels = pl.Series('Source', ['a', 'a', 'a', 'b', 'b', 'b'])

        for i, exp_df in enumerate(exp_dfs):
            obs_df, obs_labels = vc.extract_training_set(i)
            assert_frame_equal(exp_df, obs_df)
            assert_series_equal(exp_labels, obs_labels)

    def test_ValidationCollection_extract_testing_set(self):
        """ Tests the behaviour of the ValidationCollection.extract_testing_set() function using the same mask
            sets as the training test above.
        """

        df = TestProjectOrdination.spawn_core_frame(['a', 'b'], 6)
        vc = (
            ValidationCollectionBuilder()
                .add_data_core(df)
                .add_validation_set([True, False, True, False, True, False, True, False, True, False, True, False])
                .add_validation_set([False, True, False, True, False, True, False, True, False, True, False, True])
                .build()
        )

        exp_dfs = [
            pl.DataFrame([pl.Series('TSNE_1', [1, 3, 5, 7, 9, 11]), pl.Series('TSNE_2', [11, 9, 7, 5, 3, 1])]),
            pl.DataFrame([pl.Series('TSNE_1', [0, 2, 4, 6, 8, 10]), pl.Series('TSNE_2', [12, 10, 8, 6, 4, 2])])
        ]

        exp_labels = pl.Series('Source', ['a', 'a', 'a', 'b', 'b', 'b'])

        for i, exp_df in enumerate(exp_dfs):
            obs_df, obs_labels = vc.extract_testing_set(i)
            assert_frame_equal(exp_df, obs_df)
            assert_series_equal(exp_labels, obs_labels)

    def test_ValidationCollection_extract_nonoverlap(self):
        """ Verify that the ValidationCollection.extract_training_set() and
            ValidationCollection.extract_testing_set() produce non-overlapping selections over a number
            of iterations with a random seed.
        """

        def column_to_set(df, col_name):
            return set(
                df
                .get_column(col_name)
                .to_list()
            )

        def contains_overlap(set_a, set_b):
            len(set_a & set_b) > 0

        n_splits = 100
        core_df = TestProjectOrdination.spawn_core_frame(['a', 'b', 'c'], 10)
        vc = ValidationCollection(core_df, n_splits, test_size=0.5)

        for i in range(0, n_splits):
            train_df, _ = vc.extract_training_set(i)
            test_df, _ = vc.extract_testing_set(i)

            for column in ['TSNE_1', 'TSNE_2']:
                self.assertFalse(
                    contains_overlap(
                        column_to_set(train_df, column),
                        column_to_set(test_df, column)
                    )
                )

#endregion

#region Model creation and training functions

    def test_instantiate_classifiers_model(self):
        """ Tests the behaviour of the instantiate_classifiers() function with a mock callable/kwarg
            combination and a single iteration.
        """

        exp_model = MockModel(value_a=1, value_b='a')
        obs_models = instantiate_classifiers(MockModel, 'a', 1, value_a=1, value_b='a')
        self.assertEqual(exp_model, obs_models['a_0'])

    def test_instantiate_classifiers_iteration(self):
        """ Tests the behaviour of the instantiate_classifiers() function for multiple instantiations.
        """

        n_models = 5
        exp_model = MockModel(value_a=2, value_b='c')

        obs_models = instantiate_classifiers(MockModel, 'a', n_models, value_a=2, value_b='c')

        self.assertEqual(len(obs_models), n_models)
        for i in range(0, n_models):
            k = f"a_{i}"
            self.assertEqual(exp_model, obs_models[k])

    def test_score_models(self):
        """ Tests the behaviour of the score_models() function for a known input sequence.
        """

        exp_dict = {
            'Model': 'name',
            'Score_F1': 0.8222222222222223,
            'Score_MCC': 0.7833494518006403,
            'Score_ROC': 1.0,
        }

        obs_dict = score_models(
            'name',
            ['1', '1', '2', '2', '3', '3'],
            ['1', '1', '1', '2', '3', '3'],
            np.array([
                [0.9, 0.1, 0.0],
                [0.8, 0.1, 0.1],
                [0.6, 0.2, 0.2],
                [0.1, 0.9, 0.0],
                [0.1, 0.1, 0.8],
                [0.0, 0.1, 0.9],
            ])
        )

        self.assertDictEqual(exp_dict, obs_dict)

    def test_train_models(self):
        """ Tests the behaviour of the train_models() function for a single, known validation set.
        """

        exp_df = pl.DataFrame([{
            'Model': 'test',
            'Score_F1': 0.666667,
            'Score_MCC': 0.5,
            'Score_ROC': 0.875,
        }])

        df = TestProjectOrdination.spawn_core_frame(['a', 'b', 'c'], 5)
        vc = (
            ValidationCollectionBuilder()
                .add_data_core(df)
                .add_validation_set([True, True, True, False, False] * 3)
                .build()
        )

        # Take the second two of each Source as the testing set.
        mock_model = {'test': MockModel()}

        obs_df = train_models(mock_model, vc)
        assert_frame_equal(exp_df, obs_df)

#endregion

#region Model recruitment functions

    def test_proba_to_dataframe(self):
        """ Tests the behaviour of the proba_to_dataframe() function with a small data matrix
            and appropriate column names and fragment labels.
        """

        fragment_order = ['a', 'b', 'c']
        bin_order = ['first_col', 'second_col', 'third_col']

        exp_df = pl.DataFrame([
            pl.Series('Fragment', sorted(fragment_order * 3)),
            pl.Series('Bin', bin_order * 3),
            pl.Series('Score', [1., 2., 3., 4., 5., 6., 7., 8., 9.])
        ])

        input_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])

        obs_df = proba_to_dataframe(input_matrix, bin_order, fragment_order)
        assert_frame_equal(exp_df, obs_df.sort('Score', descending=False))

    def test_recruit_by_model(self):
        """ Tests the behaviour of the recruit_by_model() function with a mock clf object
            to fake the predict_proba() call and classes_ parameter.
        """

        exp_df = pl.DataFrame([
            pl.Series('Fragment', ['a1', 'a2', 'b1', 'b2', 'c1', 'c2'] * 3),
            pl.Series('Bin', ['a'] * 6 + ['b'] * 6 + ['c'] * 6),
            pl.Series('Score', [1., .2, .2, .9, 0., 0., 0., .8, .8, .1, 0., 0., 0., 0., 0., 0., 1., 1.]),
            pl.Series('Model', ['test'] * 18),
        ])

        input_df = pl.DataFrame([
            pl.Series('Fragment', ['a1', 'a2', 'b1', 'b2', 'c1', 'c2']),
            pl.Series('TSNE_1', [1., 2., 3., 4., 5., 6.]),
            pl.Series('TSNE_2', [6., 5., 4., 3., 2., 1.]),
        ])

        obs_df = recruit_by_model(MockModel(), input_df, 'test')
        assert_frame_equal(exp_df, obs_df.sort(['Bin', 'Fragment'], descending=False))

    def test_aggregate_scores(self):
        """ Tests the behaviour of the aggregate_scores() function with a default pseudocount.
        """

        exp_df = pl.DataFrame([
            pl.Series('Fragment', ['a', 'b', 'b']),
            pl.Series('Bin', ['a', 'b', 'c']),
            pl.Series('Score', [6.005, 20.009, 6.001]),
        ])

        input_df = pl.DataFrame([
            pl.Series('Fragment', ['a', 'a', 'b', 'b', 'b']),
            pl.Series('Bin', ['a', 'a', 'b', 'b', 'c']),
            pl.Series('Score', [2., 3., 4., 5., 6.]),
        ])

        obs_df = aggregate_scores(input_df)
        assert_frame_equal(exp_df, obs_df.sort('Bin', descending=False), check_exact=False, atol=.001)

    def test_aggregate_scores_pseudocount(self):
        """ Tests the behaviour of the aggregate_scores() function with an altered pseudocount.
        """

        exp_df = pl.DataFrame([
            pl.Series('Fragment', ['a', 'b', 'b']),
            pl.Series('Bin', ['a', 'b', 'c']),
            pl.Series('Score', [12.0, 30., 7.]),
        ])

        input_df = pl.DataFrame([
            pl.Series('Fragment', ['a', 'a', 'b', 'b', 'b']),
            pl.Series('Bin', ['a', 'a', 'b', 'b', 'c']),
            pl.Series('Score', [2., 3., 4., 5., 6.]),
        ])

        obs_df = aggregate_scores(input_df, pseudocount=1)
        assert_frame_equal(exp_df, obs_df.sort('Bin', descending=False))

    def test_extract_top_score(self):
        """ Tests the behaviour of the extract_top_score() function with a default reporting
            threshold (this is effectively no filtering).
        """

        exp_df = pl.DataFrame([
            pl.Series('Fragment', ['a', 'b', 'c']),
            pl.Series('Bin', ['a', 'c', 'c']),
            pl.Series('Score', [1., 1., .9]),
        ])

        input_df = pl.DataFrame([
            pl.Series('Fragment', ['a', 'a', 'b', 'b', 'c']),
            pl.Series('Bin', ['a', 'b', 'b', 'c', 'c']),
            pl.Series('Score', [1., .8, .8, 1., .9]),
        ])

        obs_df = extract_top_score(input_df)
        assert_frame_equal(exp_df, obs_df.sort('Fragment', descending=False))

    def test_extract_top_score_threshold(self):
        """ Tests the behaviour of the extract_top_score() function with a modified reporting
            threshold to exclude values.
        """

        exp_df = pl.DataFrame([
            pl.Series('Fragment', ['a', 'b', 'c']),
            pl.Series('Bin', ['a', 'c', 'unassigned']),
            pl.Series('Score', [1., 1., .9]),
        ])

        input_df = pl.DataFrame([
            pl.Series('Fragment', ['a', 'a', 'b', 'b', 'c']),
            pl.Series('Bin', ['a', 'b', 'b', 'c', 'c']),
            pl.Series('Score', [1., .8, .8, 1., .9]),
        ])

        obs_df = extract_top_score(input_df, threshold=0.95)
        assert_frame_equal(exp_df, obs_df.sort('Fragment', descending=False))

    def test_build_assignment_table(self):
        """ Tests the behaviour of the build_assignment_table() function with a mix of core
            and non-core fragments.
        """

        core_lf = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'b', 'b']),
            pl.Series('Contig', ['a1', 'a2', 'b1', 'b2']),
            pl.Series('Fragment', ['a_1', 'a_2', 'b_1', 'b_2']),
            pl.Series('Core', [True, False, True, False]),

        ]).lazy()

        recruit_lf = pl.DataFrame([
            pl.Series('Fragment', ['a_2', 'b_2']),
            pl.Series('Bin', ['a', 'c']),
        ]).lazy()

        exp_df = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'b', 'b']),
            pl.Series('Contig', ['a1', 'a2', 'b1', 'b2']),
            pl.Series('Fragment', ['a_1', 'a_2', 'b_1', 'b_2']),
            pl.Series('Bin', ['a', 'a', 'b', 'c']),
        ])

        obs_df = build_assignment_table(core_lf, recruit_lf)
        assert_frame_equal(exp_df, obs_df)

    def test_build_assignment_table_core(self):
        """ Tests the behaviour of the build_assignment_table() function with only core
            fragments.
        """

        core_lf = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'b', 'b']),
            pl.Series('Contig', ['a1', 'a2', 'b1', 'b2']),
            pl.Series('Fragment', ['a_1', 'a_2', 'b_1', 'b_2']),
            pl.Series('Core', [True, True, True, True]),

        ]).lazy()

        recruit_lf = pl.DataFrame([
            pl.Series('Fragment', ['a_1', 'a_2', 'b_1', 'b_2']),
            pl.Series('Bin', ['x', 'x', 'x', 'x']),
        ]).lazy()

        exp_df = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'b', 'b']),
            pl.Series('Contig', ['a1', 'a2', 'b1', 'b2']),
            pl.Series('Fragment', ['a_1', 'a_2', 'b_1', 'b_2']),
            pl.Series('Bin', ['a', 'a', 'b', 'b']),
        ])

        obs_df = build_assignment_table(core_lf, recruit_lf)
        assert_frame_equal(exp_df, obs_df)

    def test_build_assignment_table_noncore(self):
        """ Tests the behaviour of the build_assignment_table() function with only non-core
            fragments.
        """

        core_lf = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'b', 'b']),
            pl.Series('Contig', ['a1', 'a2', 'b1', 'b2']),
            pl.Series('Fragment', ['a_1', 'a_2', 'b_1', 'b_2']),
            pl.Series('Core', [False, False, False, False]),

        ]).lazy()

        recruit_lf = pl.DataFrame([
            pl.Series('Fragment', ['a_1', 'a_2', 'b_1', 'b_2']),
            pl.Series('Bin', ['w', 'x', 'y', 'z']),
        ]).lazy()

        exp_df = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'b', 'b']),
            pl.Series('Contig', ['a1', 'a2', 'b1', 'b2']),
            pl.Series('Fragment', ['a_1', 'a_2', 'b_1', 'b_2']),
            pl.Series('Bin', ['w', 'x', 'y', 'z']),
        ])

        obs_df = build_assignment_table(core_lf, recruit_lf)
        assert_frame_equal(exp_df, obs_df)

    def test_get_top_bin(self):
        """ Test the behaviour for the get_top_bin() function when there is a clear
            winning assignment.
        """

        input_series = pl.Series('Bin', ['a', 'a', 'b', 'a', 'c'])
        exp_dict = {'Bin': 'a', 'Support': 0.6}

        obs_dict = get_top_bin(input_series)
        self.assertDictEqual(exp_dict, obs_dict)

    def test_get_top_bin_draw(self):
        """ Test the behaviour for the get_top_bin() function when there are are multiple competing
            options. Expectation is the first entry encountered will be returned.
        """

        input_series = pl.Series('Bin', ['a', 'a', 'b', 'b', 'c'])
        exp_dict = {'Bin': 'a', 'Support': 0.4}

        obs_dict = get_top_bin(input_series)
        self.assertDictEqual(exp_dict, obs_dict)

    def test_finalise_recruitment(self):
        """ Test the behaviour for the finalise_recruitment() function.
        """

        input_df = pl.DataFrame([
            pl.Series('Source', ['a', 'a', 'a', 'a', 'b', 'b', 'b']),
            pl.Series('Contig', ['a1', 'a1', 'a1', 'a1', 'b1', 'b2', 'b2']),
            pl.Series('Bin', ['a', 'a', 'a', 'b', 'b', 'b', 'c']),
        ])

        exp_df = pl.DataFrame([
            pl.Series('Source', ['a', 'b', 'b']),
            pl.Series('Contig', ['a1', 'b1', 'b2']),
            pl.Series('Bin', ['a', 'b', 'b']),
            pl.Series('Support', [0.75, 1.0, 0.5])
        ])

        obs_df = finalise_recruitment(input_df)
        assert_frame_equal(exp_df, obs_df)

#endregion

if __name__ == '__main__':

    unittest.main()
