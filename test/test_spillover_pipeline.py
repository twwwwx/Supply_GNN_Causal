import unittest

import numpy as np

from src.gen_data import sample_data_spillover


class TestSpilloverDGP(unittest.TestCase):
    def test_exposure_shape_and_state_index(self):
        draw = sample_data_spillover(
            sample_size=120,
            seed=11,
            graph_model="rgg",
            tau_dir=2.0,
            tau_in=1.0,
            tau_out=1.0,
        )
        t = np.asarray(draw["T"], dtype=int)
        state_idx = np.asarray(draw["state_index"], dtype=int)

        self.assertEqual(t.shape, (120, 3))
        self.assertTrue(np.all((t == 0) | (t == 1)))
        self.assertTrue(np.all((state_idx >= 0) & (state_idx <= 7)))

        recomputed = 4 * t[:, 0] + 2 * t[:, 1] + t[:, 2]
        self.assertTrue(np.array_equal(recomputed, state_idx))

    def test_true_taus_consistency(self):
        draw = sample_data_spillover(
            sample_size=80,
            seed=22,
            graph_model="er",
            tau_dir=2.5,
            tau_in=0.8,
            tau_out=-0.3,
        )
        true_taus = draw["true_taus"]
        self.assertAlmostEqual(true_taus["tau_tot"], true_taus["tau_dir"] + true_taus["tau_in"] + true_taus["tau_out"])


class TestSpilloverEstimator(unittest.TestCase):
    def test_estimator_contract_if_torch_available(self):
        try:
            from src.ate import tau_vector_and_se_from_gnn
        except Exception:
            self.skipTest("Torch/PyG stack not available in this environment.")

        draw = sample_data_spillover(sample_size=40, seed=7, graph_model="er")
        fit = tau_vector_and_se_from_gnn(
            draw,
            feature_key="node_features",
            directed=False,
            use_gpu=False,
            variance_type="iid",
            num_layers=1,
            output_dim=4,
            seed=7,
        )
        keys = ("tau_dir", "tau_in", "tau_out", "tau_tot")
        self.assertEqual(set(fit["tau_hat"].keys()), set(keys))
        self.assertEqual(set(fit["se_hat"].keys()), set(keys))
        self.assertEqual(set(fit["sigma2_hat"].keys()), set(keys))
        self.assertEqual(set(fit["bandwidth"].keys()), set(keys))

        propensity = np.asarray(fit["propensity"], dtype=float)
        self.assertEqual(propensity.shape[1], 8)
        row_sums = np.sum(propensity, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-5))
        self.assertTrue(np.all(np.asarray(list(fit["se_hat"].values()), dtype=float) >= 0.0))


if __name__ == "__main__":
    unittest.main()
