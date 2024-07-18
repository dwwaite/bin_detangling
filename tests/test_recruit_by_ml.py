import sys
import unittest
from dataclasses import dataclass, field
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from bin.recruit_by_ml import ValidationSet
from bin.recruit_by_ml import extract_core_contigs
from bin.recruit_by_ml import create_training_sets
from bin.recruit_by_ml import package_predictions
from bin.recruit_by_ml import proba_to_decisions
from bin.recruit_by_ml import train_models

@dataclass
class MockModel:

    classes_: list[bool] = field(default_factory=list)

    def __init__(self, classes):
        self.classes_ = classes

    def fit(self, x, y):
        return

    def predict_proba(self, x):
        return np.array([
            [1, 2],
            [3, 4]
        ])

class TestProjectOrdination(unittest.TestCase):

#region Data import and preparation

    def test_extract_core_contigs(self):

        exp_core_df = pl.DataFrame([pl.Series('Core', [True, True, True]), pl.Series('Results', [1, 2, 5])])
        exp_ncore_df = pl.DataFrame([pl.Series('Core', [False, False]), pl.Series('Results', [3, 4])])

        obs_core_df, obs_ncore_df = extract_core_contigs('tests/test_extract_core_contigs.parquet')

        assert_frame_equal(exp_core_df, obs_core_df)
        assert_frame_equal(exp_ncore_df, obs_ncore_df)

    def test_create_training_sets(self):

        # Create a sufficiently populated test data set
        input_data = ['A'] * 20
        input_data += ['B'] * 10
        input_data += ['C'] * 10

        # Expected output (for seed = 1)
        exp_results = [
            ValidationSet(
                [14, 39, 23, 21, 24, 37, 6, 9, 27, 30, 8, 29, 19, 0, 36, 31, 2, 25, 35, 4, 10, 38, 32, 13, 15, 20, 18, 28, 7, 12, 33, 3, 16, 26, 17, 1],
                [11, 5, 22, 34]
            ),
            ValidationSet(
                [0, 29, 15, 31, 16, 26, 33, 12, 1, 39, 18, 19, 14, 27, 21, 38, 25, 20, 3, 5, 28, 10, 22, 7, 37, 35, 30, 34, 2, 24, 32, 4, 11, 17, 13, 6],
                [36, 8, 9, 23]
            )
        ]

        obs_results = create_training_sets(input_data, 2, seed=1)

        for exp_data, obs_data in zip(exp_results, obs_results):
            self.assertEqual(exp_data, obs_data)

#endregion

#region Model training

    def test_package_predictions(self):

        column_classes = ['class_a', 'class_b']
        exp_df = pl.DataFrame([
            pl.Series('class_a', [1, 3]),
            pl.Series('class_b', [2, 4]),
            pl.Series('Bin', ['class_a', 'class_b']),
            pl.Series('Fragments', ['a', 'b'])
        ])

        obs_df = package_predictions(np.array([[1, 2], [3, 4]]), column_classes, ['class_a', 'class_b'], ['a', 'b'])
        assert_frame_equal(exp_df, obs_df)

    def test_proba_to_decisions(self):

        input_data = pl.DataFrame([
            pl.Series('class_a', [0.9, 0.3, 0.6]),
            pl.Series('class_b', [0.1, 0.7, 0.4]),
            pl.Series('Bin', ['class_a', 'class_b', 'class_b']),
            pl.Series('Fragments', ['a', 'b', 'c'])
        ])

        exp_df = pl.DataFrame([
            pl.Series('Fragments', ['a', 'b', 'c']),
            pl.Series('Bin', ['class_a', 'class_b', 'class_b']),
            pl.Series('Prediction', ['class_a', 'class_b', 'class_a']),
            pl.Series('Score', [0.9, 0.7, 0.6]),
            pl.Series('Matched', [True, True, False])
        ])

        obs_df = proba_to_decisions(input_data)
        assert_frame_equal(exp_df, obs_df.sort('Fragments', descending=False))

    def test_train_models(self):

        input_df = pl.DataFrame([
            pl.Series('Source', ['bin_1', 'bin_1', 'bin_1', 'bin_1', 'bin_2', 'bin_2', 'bin_2', 'bin_2']),
            pl.Series('Contig', ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']),
            pl.Series('Fragment', ['A_1', 'A_2', 'A_3', 'A_4', 'B_1', 'B_2', 'B_3', 'B_4']),
            pl.Series('TSNE_1', [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4]),
            pl.Series('TSNE_2', [10.1, 10.2, 10.3, 10.4, 20.1, 20.2, 20.3, 20.4])
        ])

        # Take the second two of each Source as the testing set.
        input_validation = ValidationSet([0, 1, 4, 5], [2, 3, 6, 7])

        mock_model = MockModel(['bin_1', 'bin_2'])

        x = train_models(input_df, [input_validation], [mock_model])
        print(x)

#endregion

    """
    def train_models(core_df: pl.DataFrame, training_sets: List[ValidationSet], model_base: Any, **kwargs) -> Any:
        model_buffer = []
        score_buffer = []
        for training_set in training_sets:

            training_labels, training_fragments, training_data = training_set.extract_dataframe_training(core_df)
            testing_labels, testing_fragments, testing_data = training_set.extract_dataframe_testing(core_df)

            model_instance.fit(training_data, training_labels)

            x = model_instance.predict_proba(testing_data)
            proba_df = package_predictions(x, model_instance.classes_, testing_labels, testing_fragments)
            prediction_df = proba_to_decisions(proba_df)

            model_buffer.append(model_instance)
            score_buffer.append(
                matthews_corrcoef(prediction_df['Bin'], prediction_df['Prediction'])
            )

        # Find the top-scoring model and return
        i = score_buffer.index(max(score_buffer))
        return model_buffer[i]
    """

if __name__ == '__main__':

    unittest.main()
