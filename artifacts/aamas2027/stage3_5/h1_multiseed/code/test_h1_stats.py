#!/usr/bin/env python3
"""Offline unit tests for the H1 multi-seed statistics helpers.

Pure numerical tests (no frozen artifacts needed): the Student-t CDF/PPF are
checked against textbook critical values, and sample_stats / one_sample_t
against hand-computed references.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_h1_multi as m  # noqa: E402


class TestTDistribution(unittest.TestCase):
    def test_cdf_symmetry(self):
        for df in (1, 4, 9, 30):
            self.assertAlmostEqual(m.t_cdf(0.0, df), 0.5, places=12)
            self.assertAlmostEqual(
                m.t_cdf(1.3, df) + m.t_cdf(-1.3, df), 1.0, places=12
            )

    def test_known_critical_values(self):
        # textbook two-sided 95% critical values
        known = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 9: 2.2622, 29: 2.0452}
        for df, tcrit in known.items():
            self.assertAlmostEqual(m.t_cdf(tcrit, df), 0.975, places=4)
            self.assertAlmostEqual(m.t_ppf(0.975, df), tcrit, places=3)

    def test_two_sided_p(self):
        self.assertAlmostEqual(m.t_two_sided_p(0.0, 4), 1.0, places=12)
        # t=2.7764 df=4 -> two-sided p = 0.05
        self.assertAlmostEqual(m.t_two_sided_p(2.7764, 4), 0.05, places=3)
        self.assertEqual(m.t_two_sided_p(math.inf, 4), 0.0)

    def test_betai_special_cases(self):
        self.assertAlmostEqual(m._betai(1.0, 1.0, 0.3), 0.3, places=12)  # uniform
        self.assertAlmostEqual(m._betai(2.0, 1.0, 0.5), 0.25, places=12)  # x^2
        self.assertEqual(m._betai(0.5, 0.5, 0.0), 0.0)
        self.assertEqual(m._betai(0.5, 0.5, 1.0), 1.0)


class TestSampleStats(unittest.TestCase):
    def test_known_sample(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = m.sample_stats(vals)
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["mean"], 3.0, places=12)
        self.assertAlmostEqual(s["sd"], math.sqrt(2.5), places=12)
        self.assertAlmostEqual(s["se"], math.sqrt(2.5) / math.sqrt(5), places=12)
        self.assertAlmostEqual(s["tcrit"], 2.7764, places=3)
        self.assertEqual(s["min"], 1.0)
        self.assertEqual(s["max"], 5.0)
        self.assertLess(s["ci95_lo"], 3.0)
        self.assertGreater(s["ci95_hi"], 3.0)

    def test_one_sample_t(self):
        deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = m.one_sample_t(deltas)
        self.assertAlmostEqual(r["t"], 3.0 / (math.sqrt(2.5) / math.sqrt(5)), places=10)
        self.assertAlmostEqual(r["t"], 4.2426, places=3)
        self.assertEqual(r["df"], 4)
        # hand reference: t=4.2426, df=4 -> two-sided p ~= 0.0132
        self.assertTrue(0.010 < r["p_two_sided_raw"] < 0.016)

    def test_zero_variance(self):
        r = m.one_sample_t([2.0, 2.0, 2.0])
        self.assertEqual(r["t"], math.inf)
        self.assertEqual(r["p_two_sided_raw"], 0.0)
        r0 = m.one_sample_t([0.0, 0.0, 0.0])
        self.assertEqual(r0["t"], 0.0)
        self.assertAlmostEqual(r0["p_two_sided_raw"], 1.0, places=12)


class TestBindings(unittest.TestCase):
    def test_job_ids(self):
        self.assertEqual(m.frozen_sc_job_id(800, 323), "A-EVAL-SC-C800-S323")
        self.assertEqual(m.h1_job_id(1200, 999), "H1MS-H1-C1200-S999")
        self.assertEqual(m.SEEDS, [123, 323, 532, 777, 999])
        self.assertEqual(m.NEW_SEEDS, [323, 532, 777, 999])
        self.assertEqual(m.N_SEEDS, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
