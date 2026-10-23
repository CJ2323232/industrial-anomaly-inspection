"""Compare blocked search with hand arithmetic and direct float64 subtraction."""
import unittest
import torch

from scripts.nearest_neighbors import exact_nearest_neighbors


class NearestNeighborTests(unittest.TestCase):
    def test_hand_computed_distances(self):
        queries = torch.tensor([[0., 4.], [3., 0.]], dtype=torch.float64)
        bank = torch.tensor([[0., 0.], [3., 4.], [9., 9.]], dtype=torch.float64)
        distances, indices = exact_nearest_neighbors(queries, bank, 1, 2)
        torch.testing.assert_close(distances, torch.tensor([3., 3.], dtype=torch.float64))
        self.assertEqual(indices.tolist(), [1, 0])

    def test_uneven_chunks_match_independent_dense_oracle_without_mutation(self):
        generator = torch.Generator().manual_seed(19)
        queries = torch.randn(7, 10, generator=generator, dtype=torch.float64)[:, ::2]
        bank = torch.randn(11, 10, generator=generator, dtype=torch.float64)[:, ::2]
        original_q, original_b = queries.clone(), bank.clone()
        expected, locations = torch.linalg.vector_norm(queries[:, None] - bank[None], dim=2).min(dim=1)
        for chunks in ((1, 1), (3, 4), (20, 30)):
            distances, indices = exact_nearest_neighbors(queries, bank, *chunks)
            torch.testing.assert_close(distances, expected, atol=1e-12, rtol=1e-12)
            self.assertEqual(indices.tolist(), locations.tolist())
        torch.testing.assert_close(queries, original_q, atol=0, rtol=0)
        torch.testing.assert_close(bank, original_b, atol=0, rtol=0)

    def test_ties_across_blocks_choose_lowest_index(self):
        query = torch.tensor([[0., 0.]])
        bank = torch.tensor([[3., 4.], [1., 0.], [-1., 0.], [1., 0.]])
        for chunk in (1, 2, 4):
            distances, indices = exact_nearest_neighbors(query, bank, 1, chunk)
            self.assertEqual(distances.item(), 1.)
            self.assertEqual(indices.item(), 1)

    def test_close_large_vectors_and_exact_matches(self):
        queries = torch.tensor([[10000., 10000.], [10000., 10001.]])
        bank = torch.tensor([[10000., 10000.], [10000., 10003.]])
        distances, indices = exact_nearest_neighbors(queries, bank, 1, 1)
        self.assertEqual(distances.tolist(), [0., 1.])
        self.assertEqual(indices.tolist(), [0, 0])

    def test_reject_invalid_inputs(self):
        valid = torch.ones(2, 3)
        for queries, bank in ((torch.ones(0, 3), valid), (valid, torch.ones(2, 4)),
                              (valid, valid.double()), (torch.full((2, 3), float("nan")), valid),
                              (valid, torch.full((2, 3), float("inf"))), (valid.long(), valid.long())):
            with self.assertRaises(ValueError):
                exact_nearest_neighbors(queries, bank)
        for chunk in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                exact_nearest_neighbors(valid, valid, bank_chunk=chunk)

    def test_reference_formula_on_384d_synthetic_features(self):
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        generator = torch.Generator().manual_seed(21)
        queries = torch.randn(5, 384, generator=generator, dtype=torch.float64)
        bank = torch.randn(9, 384, generator=generator, dtype=torch.float64)
        queries[0] = bank[4]  # Include the delicate zero-distance case.
        reference = PatchcoreModel.euclidean_dist(queries, bank)
        expected, indices = reference.min(dim=1)
        actual, locations = exact_nearest_neighbors(queries, bank, 2, 4)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-7)
        torch.testing.assert_close(reference.gather(1, locations[:, None]).squeeze(1), expected,
                                   atol=1e-6, rtol=1e-7)
        self.assertEqual(locations.tolist(), indices.tolist())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_direct_cpu_oracle(self):
        generator = torch.Generator().manual_seed(20)
        for dtype, tolerance in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
            queries = torch.randn(7, 8, generator=generator, dtype=dtype)
            bank = torch.randn(11, 8, generator=generator, dtype=dtype)
            expected, locations = torch.linalg.vector_norm(queries[:, None] - bank[None], dim=2).min(dim=1)
            distances, indices = exact_nearest_neighbors(queries.cuda(), bank.cuda(), 3, 4)
            torch.testing.assert_close(distances.cpu(), expected, atol=tolerance, rtol=tolerance)
            self.assertEqual(indices.cpu().tolist(), locations.tolist())


if __name__ == "__main__":
    unittest.main()
