import sys
import unittest
from Bio.Seq import Seq

import polars as pl
from polars.testing import assert_frame_equal

from bin.compute_kmer_profile import Fragment
from bin.compute_kmer_profile import slice_fasta_file
from bin.compute_kmer_profile import sequence_to_kmers
from bin.compute_kmer_profile import compute_kmer_profile
from bin.compute_kmer_profile import deploy_multithreaded_function
from bin.compute_kmer_profile import combine_data_frames

class TestComputeKmerProfile(unittest.TestCase):

    def test_fragment_create_line(self):

        exp_result = ">A\nATGC\n"

        obs_result = Fragment('', '', 'A', Seq('ATGC'), 0).create_line()
        self.assertEqual(exp_result, obs_result)

    def test_slice_fasta_file(self):

        # First case - sequence splits into 4 pieces, but final is too short
        # Second case - sequence splits into 2 pieces, but final is too short
        exp_results = [
            Fragment('tests/mock_sequence.fna', 'seq_a', 'seq_a__0', Seq('ATCGA'), 4),
            Fragment('tests/mock_sequence.fna', 'seq_a', 'seq_a__5', Seq('TCGAT'), 4),
            Fragment('tests/mock_sequence.fna', 'seq_a', 'seq_a__10', Seq('CGATCG'), 4),
            Fragment('tests/mock_sequence.fna', 'seq_b', 'seq_b__0', Seq('GCTAGCTA'), 4),
        ]
        obs_results = slice_fasta_file('tests/mock_sequence.fna', 5, 4)

        self.assertListEqual(exp_results, obs_results)

    def test_slice_fasta_file_short(self):

        # Test for when the sequences are shorter than the window size
        exp_results = [
            Fragment('tests/mock_sequence.fna', 'seq_a', 'seq_a__0', Seq('ATCGATCGATCGATCG'), 4),
            Fragment('tests/mock_sequence.fna', 'seq_b', 'seq_b__0', Seq('GCTAGCTA'), 4),
        ]
        obs_results = slice_fasta_file('tests/mock_sequence.fna', 100, 4)

        self.assertListEqual(exp_results, obs_results)

    def test_sequence_to_kmers(self):

        exp_results = ['AC', 'CG', 'GA']
        obs_results = [x for x in sequence_to_kmers('ACGA', 2)]

        self.assertListEqual(exp_results, obs_results)

    def test_sequence_to_kmers__rc(self):

        # First kmer  = AGG
        # Second kmer = CCC (rev comp of GGG)
        # Third kmer  = GGA
        exp_results = ['AGG', 'CCC', 'GGA']
        obs_results = [x for x in sequence_to_kmers('AGGGA', 3)]

        self.assertListEqual(exp_results, obs_results)

    def test_sequence_to_kmers__skip_n(self):

        exp_results = ['AA', 'CC']
        obs_results = [x for x in sequence_to_kmers('AANCC', 2)]

        self.assertListEqual(exp_results, obs_results)

    def test_compute_kmer_profile(self):

        exp_results = pl.DataFrame([
            {'Source': 'source', 'Contig': 'contig', 'Fragment': 'fragment', 'Kmer': 'AT', 'Count': 2},
            {'Source': 'source', 'Contig': 'contig','Fragment': 'fragment', 'Kmer': 'TA', 'Count': 1},
            {'Source': 'source', 'Contig': 'contig', 'Fragment': 'fragment', 'Kmer': 'GA', 'Count': 1}
        ])
        obs_results = compute_kmer_profile(Fragment('source', 'contig', 'fragment', 'ATATC', 2))

        assert_frame_equal(exp_results, obs_results)

    @staticmethod
    def mock_function(i):
        return i+1

    def test_deploy_multithreaded_function(self):

        input_data = [i for i in range(0, 100000)]
        exp_results = [i+1 for i in input_data]

        obs_results = deploy_multithreaded_function(input_data, TestComputeKmerProfile.mock_function, 2)
        self.assertListEqual(exp_results, obs_results)

    def test_deploy_multithreaded_kmers(self):

        exp_df = pl.DataFrame([
            {'Source': 'source', 'Contig': 'contig', 'Fragment': 'fragment', 'Kmer': 'AT', 'Count': 2},
            {'Source': 'source', 'Contig': 'contig','Fragment': 'fragment', 'Kmer': 'TA', 'Count': 1},
            {'Source': 'source', 'Contig': 'contig', 'Fragment': 'fragment', 'Kmer': 'GA', 'Count': 1}
        ])

        obs_results = deploy_multithreaded_function(
            [Fragment('source', 'contig', 'fragment', 'ATATC', 2)],
            compute_kmer_profile,
            1
        )

        obs_df = obs_results[0]
        assert_frame_equal(exp_df, obs_df)

    def test_combine_data_frames(self):

        sources = ['A', 'A', 'B', 'C', 'C']
        contigs = ['a_1', 'a_2', 'b_1', 'c_1', 'c_2']
        fragments = ['1', '2', '1', '2', '3']
        kmers = ['AC', 'TC', 'AA', 'GG', 'CC']
        counts = [1, 1, 1, 1, 1]

        exp_df = pl.DataFrame([
            pl.Series('Source', sources),
            pl.Series('Contig', contigs),
            pl.Series('Fragment', fragments),
            pl.Series('Kmer', kmers),
            pl.Series('Count', counts),
        ])

        # Shuffle in order B, C, A
        input_dfs = []
        for (start, stop) in [(3, 5), (2, 3), (0, 2)]:
            input_dfs.append(
                pl.DataFrame([
                    pl.Series('Source', sources[start: stop]),
                    pl.Series('Contig', contigs[start: stop]),
                    pl.Series('Fragment', fragments[start: stop]),
                    pl.Series('Kmer', kmers[start: stop]),
                    pl.Series('Count', counts[start: stop]),
                ])
            )

        obs_df = combine_data_frames(input_dfs)
        assert_frame_equal(exp_df, obs_df)

if __name__ == '__main__':

    unittest.main()
