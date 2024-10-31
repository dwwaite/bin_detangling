import unittest

import polars as pl
from polars.testing import assert_frame_equal

from bin.compute_depth_profile import depth_label_generator
from bin.compute_depth_profile import parse_depth_file

class TestComputeDepthProfile(unittest.TestCase):

    def test_depth_label_generator(self):

        exp_result = [
            ('abc.txt', 'Depth_1'),
            ('def.txt', 'Depth_2'),
            ('ghi.txt', 'Depth_3')
        ]

        obs_result = [x for x in depth_label_generator(['abc.txt', 'def.txt', 'ghi.txt'])]
        self.assertListEqual(exp_result, obs_result)

    def test_parse_depth_file(self):

        exp_df = pl.DataFrame([
            pl.Series('Contig', ['a', 'b', 'c']),
            pl.Series('Coverage', [5.0, 4.0, 10.5]),
            pl.Series('Label', ['Depth_1', 'Depth_1', 'Depth_1'])
        ])

        obs_df = parse_depth_file('tests/mock_depth.txt', 'Depth_1')
        sorted_df = obs_df.sort('Contig', descending=False)

        assert_frame_equal(exp_df, sorted_df)

if __name__ == '__main__':

    unittest.main()
