#!/usr/bin/env python3
"""Focused validation tests for the Stage 3.5 B0 hybrid diagnostic.

Run with the frozen-approved Mac runtime:
    /Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python \
        artifacts/aamas2027/stage3_5/b0_diagnostic/tests/test_b0.py

Coverage:
  1. BF settlement helper == frozen BF settle rule (feasibility, ties, no-op).
  2. BF flush helper == frozen BF flush rule (strict threshold, ties,
     settlement exclusion, no-op, oversize, usability, no-feasible-settle).
  3. Factorized BF helpers recompose to frozen adapter.bf_action everywhere.
  4. H1 truly feeds the BF settle index into the SC conditional flush head.
  5. H2 uses the unmodified SC settle decode; SC/SC path == agent.act.
  6. Replayed SC/SC and BF/BF episode metrics equal frozen Stage 2 outputs.
  7. Money = settled - 10*flushes, count identities, finite/valid actions.
  8. Frozen Stage 3 reference tables agree with the frozen raw episodes.
"""
from __future__ import annotations

import math
import os
import random
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import b0_lib as b0  # noqa: E402

ADAPTER_DIR = b0.FROZEN_REPO / "tools" / "aamas_stage2"
sys.path.insert(0, str(ADAPTER_DIR))
import adapter  # noqa: E402

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


class FakeEnv:
    """Minimal object exposing exactly the attributes the BF rule reads."""

    def __init__(self, k, C, balances, usable, current_tx):
        self.k = k
        self.C = float(C)
        self.wallets = [float(b) for b in balances]
        self._usable_flags = [bool(u) for u in usable]
        self.current_tx = float(current_tx)

    def _usable(self, i):
        return self._usable_flags[i]


# ---------------------------------------------------------------------------
# 1-3. Rule-level tests (no torch)
# ---------------------------------------------------------------------------

class TestBFRule(unittest.TestCase):
    def test_settle_smallest_feasible_with_index_tie(self):
        env = FakeEnv(3, 120, [10, 5, 5], [1, 1, 1], 5)
        self.assertEqual(b0.bf_settle(env), 1)
        env = FakeEnv(3, 120, [10, 10, 10], [1, 1, 1], 10)
        self.assertEqual(b0.bf_settle(env), 0)

    def test_settle_respects_balance_feasibility(self):
        # Smallest *feasible* balance: 11 < 50, the infeasible 4 is ignored.
        env = FakeEnv(3, 120, [4, 50, 11], [1, 1, 1], 10)
        self.assertEqual(b0.bf_settle(env), 2)

    def test_settle_noop_when_no_feasible(self):
        env = FakeEnv(3, 120, [1, 2, 3], [1, 1, 1], 100)
        self.assertEqual(b0.bf_settle(env), 3)

    def test_settle_ignores_frozen_wallets(self):
        # Only wallet 2 is usable and feasible; frozen rich wallet is skipped.
        env = FakeEnv(3, 120, [1000, 1000, 10], [0, 0, 1], 10)
        self.assertEqual(b0.bf_settle(env), 2)
        # Feasible wallets exist, but all are frozen -> no-op.
        env = FakeEnv(3, 120, [1000, 1000, 1000], [0, 0, 0], 10)
        self.assertEqual(b0.bf_settle(env), 3)

    def test_flush_strict_threshold_boundary(self):
        # C/k = 40, threshold = 20; balance == 20 is NOT eligible.
        env = FakeEnv(3, 120, [20, 20, 19.999], [1, 1, 1], 10)
        self.assertEqual(b0.bf_flush(env, settle=3), 2)
        env = FakeEnv(3, 120, [20, 20, 20], [1, 1, 1], 10)
        self.assertEqual(b0.bf_flush(env, settle=3), 3)

    def test_flush_tie_break_index(self):
        env = FakeEnv(4, 160, [5, 5, 5, 5], [1, 1, 1, 1], 10)
        self.assertEqual(b0.bf_flush(env, settle=4), 0)

    def test_flush_excludes_settlement_wallet(self):
        # Wallet 0 has the smallest balance but is the settlement wallet.
        env = FakeEnv(4, 160, [1, 5, 6, 7], [1, 1, 1, 1], 1)
        self.assertEqual(b0.bf_settle(env), 0)
        self.assertEqual(b0.bf_flush(env, settle=0), 1)

    def test_flush_allowed_when_settlement_noop(self):
        env = FakeEnv(3, 120, [1, 2, 30], [1, 1, 1], 100)
        self.assertEqual(b0.bf_settle(env), 3)
        self.assertEqual(b0.bf_flush(env, settle=3), 0)

    def test_flush_independent_of_oversize_tx(self):
        # Huge transaction (oversize) must not suppress a valid flush.
        env = FakeEnv(3, 120, [1, 2, 30], [1, 1, 1], 9999)
        self.assertEqual(b0.bf_settle(env), 3)
        self.assertEqual(b0.bf_flush(env, settle=3), 0)

    def test_flush_noop_when_none_eligible(self):
        env = FakeEnv(3, 120, [19, 19.5, 20], [1, 1, 1], 10)
        # settle=2 leaves balances 19/19.5 < 20 eligible -> pick 0; then
        # settle=0 excluding it leaves 19.5 -> pick 1; excluding all low
        # wallets leaves only 20 -> no-op.
        self.assertEqual(b0.bf_flush(env, settle=2), 0)
        self.assertEqual(b0.bf_flush(env, settle=1), 0)
        self.assertEqual(b0.bf_flush(env, settle=0), 1)

    def test_flush_skips_frozen_wallets(self):
        env = FakeEnv(3, 120, [1, 2, 3], [0, 1, 1], 100)
        self.assertEqual(b0.bf_flush(env, settle=3), 1)

    def test_flush_threshold_uses_C_over_k(self):
        # C=800, k=24 -> B=33.333..., threshold 16.666...
        balances = [16.7] * 24
        env = FakeEnv(24, 800, balances, [1] * 24, 10)
        self.assertEqual(b0.bf_flush(env, settle=24), 24)
        balances[7] = 16.6
        env = FakeEnv(24, 800, balances, [1] * 24, 10)
        self.assertEqual(b0.bf_flush(env, settle=24), 7)

    def test_recomposition_matches_frozen_adapter_on_random_envs(self):
        rng = random.Random(20261002)
        for trial in range(2000):
            k = rng.choice([1, 2, 3, 5, 24])
            C = rng.choice([800, 1200])
            B = C / k
            balances = [rng.uniform(-0.0, 1.5 * B) for _ in range(k)]
            usable = [rng.random() < 0.8 for _ in range(k)]
            tx = rng.choice([1, 2, int(B * 0.4), int(B * 0.5), int(B), int(B * 2), int(B * 5)])
            env = FakeEnv(k, C, balances, usable, tx)
            s = b0.bf_settle(env)
            f = b0.bf_flush(env, s)
            self.assertEqual(s * (k + 1) + f, adapter.bf_action(env))

    def test_real_threshold_constant(self):
        # The frozen adapter RULE string pins the same semantics we implement.
        self.assertIn("balance<0.5*C/k", adapter.RULE)
        self.assertIn("noop=k", adapter.RULE)


# ---------------------------------------------------------------------------
# Torch / frozen-runtime integration tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_TORCH, "approved runtime with torch required")
class TestHybridIntegration(unittest.TestCase):
    sess = None

    @classmethod
    def setUpClass(cls):
        b0.configure_caches()
        adapter.verify_sources(b0.FROZEN_REPO)
        b0.register_adapter(adapter)
        # Reuse the B0 session machinery: imports + verified NEW12 pools.
        sys.path.insert(0, str(CODE_DIR))
        import run_b0  # noqa: WPS433

        cls.run_b0 = run_b0
        cls.sess = run_b0.Session()
        cls.pool_US = cls.sess.pool("US")["pool"]

    def _env_agent(self, C):
        cfg = self.sess.cfg(C)
        env = self.sess.sc.make_env(cfg, max_steps=b0.T)
        return env, cfg, self.sess.agent(C)

    def test_sc_sc_hybrid_matches_agent_act_deterministic(self):
        import torch

        for C in b0.CAPACITIES:
            env, _cfg, agent = self._env_agent(C)
            state = env.reset(tx_stream=self.pool_US[0])
            for t in range(150):
                s, f, a, _, _ = b0.hybrid_action(env, agent, state, "SC", "SC")
                a_ref, s_ref, f_ref, _lp, _v, _g = agent.act(state, deterministic=True)
                self.assertEqual((s, f, a), (s_ref, f_ref, a_ref), f"C{C} t{t}")
                state = env.step(a)[0]

    def test_h1_feeds_bf_settle_into_flush_head(self):
        for C in b0.CAPACITIES:
            env, _cfg, agent = self._env_agent(C)
            recorded = []
            orig = agent.model.forward_flush_given_settle

            def rec(state_t, settle_t):
                recorded.append(int(settle_t.item()))
                return orig(state_t, settle_t)

            agent.model.forward_flush_given_settle = rec
            try:
                state = env.reset(tx_stream=self.pool_US[1])
                for t in range(120):
                    s_bf = b0.bf_settle(env)
                    s, f, a, _scs, _scf = b0.hybrid_action(
                        env, agent, state, "BF", "SC", diagnostics=False
                    )
                    self.assertEqual(s, s_bf)
                    self.assertEqual(recorded[-1], s_bf)
                    state = env.step(a)[0]
            finally:
                agent.model.forward_flush_given_settle = orig

    def test_h1_flush_is_argmax_conditioned_on_bf_settle(self):
        import torch

        for C in b0.CAPACITIES:
            env, _cfg, agent = self._env_agent(C)
            state = env.reset(tx_stream=self.pool_US[2])
            for t in range(120):
                s_bf = b0.bf_settle(env)
                _s, f, _a, _scs, _scf = b0.hybrid_action(
                    env, agent, state, "BF", "SC", diagnostics=False
                )
                state_t = torch.tensor(
                    state, dtype=torch.float32, device=agent.device
                ).unsqueeze(0)
                with torch.no_grad():
                    expected = int(
                        torch.argmax(
                            agent.model.forward_flush_given_settle(
                                state_t, torch.tensor([s_bf], dtype=torch.long)
                            ),
                            dim=-1,
                        ).item()
                    )
                self.assertEqual(f, expected)
                state = env.step(s_bf * (env.k + 1) + f)[0]

    def test_h2_uses_sc_settle_and_bf_flush_exclusion(self):
        import torch

        for C in b0.CAPACITIES:
            env, _cfg, agent = self._env_agent(C)
            state = env.reset(tx_stream=self.pool_US[3])
            for t in range(120):
                state_t = torch.tensor(
                    state, dtype=torch.float32, device=agent.device
                ).unsqueeze(0)
                with torch.no_grad():
                    sl, _ = agent.model.forward_settle_value(state_t)
                    s_sc = int(torch.argmax(sl, dim=-1).item())
                s, f, a, _scs, _scf = b0.hybrid_action(
                    env, agent, state, "SC", "BF", diagnostics=False
                )
                self.assertEqual(s, s_sc)
                self.assertEqual(f, b0.bf_flush(env, s_sc))
                # Settlement exclusion: if f < k then f != s.
                self.assertTrue(f == env.k or f != s)
                state = env.step(a)[0]

    def test_sc_sc_parity_on_frozen_episodes(self):
        for C in b0.CAPACITIES:
            cap = self.run_b0.run_parity_for_capacity(
                self.sess, C, regimes=["US", "UB"], episodes=3
            )
            for reg, v in cap["regimes"].items():
                self.assertTrue(v["sc_sc_exact"], (C, reg, v["sc_sc_first_error"]))

    def test_bf_bf_parity_on_frozen_episodes(self):
        for C in b0.CAPACITIES:
            cap = self.run_b0.run_parity_for_capacity(
                self.sess, C, regimes=["US", "PLB"], episodes=3
            )
            for reg, v in cap["regimes"].items():
                self.assertTrue(v["bf_bf_ja_env_exact"], (C, reg, v["bf_ja_first_error"]))
                self.assertTrue(v["bf_bf_sc_env_exact"], (C, reg, v["bf_sc_env_first_error"]))

    def test_hybrid_episode_accounting_and_finite_actions(self):
        for C in b0.CAPACITIES:
            cfg = self.sess.cfg(C)
            agent = self.sess.agent(C)
            for hybrid, sm, fm in [(b0.H1, "BF", "SC"), (b0.H2, "SC", "BF")]:
                res = b0.evaluate_hybrid_pool(
                    self.sess.sc,
                    cfg,
                    agent,
                    self.pool_US[:4],
                    sm,
                    fm,
                    record=False,
                    rule_fidelity_check=True,
                )
                self.assertEqual(res["num_episodes"], 4)
                for r in res["raw_results"]:
                    self.assertTrue(math.isfinite(r["eval_money"]))
                    self.assertAlmostEqual(
                        r["eval_money"], r["settled"] - 10 * r["flushes"], places=9
                    )
                    self.assertEqual(
                        int(r["drops"]) + int(r["accepted_count"]), b0.T
                    )
                    self.assertEqual(
                        int(r["oversize_drops"]) + int(r["insufficient_drops"]),
                        int(r["drops"]),
                    )
                    self.assertEqual(int(r["total_tx_count"]), b0.T)
                    for key in ("settled", "flushes", "drops", "accepted_count"):
                        self.assertTrue(math.isfinite(float(r[key])))


class TestFrozenReferences(unittest.TestCase):
    def test_stage3_tables_equal_raw_episodes_for_reference_cells(self):
        refs = b0.load_stage3_reference()
        for label, method in [("SC", "SC-FAC"), ("BF", "BF-T0.5")]:
            for C in b0.CAPACITIES:
                vec = b0.episodes_to_vectors(
                    b0.load_frozen_episodes(b0.FROZEN_JOB_IDS[(label, C)])
                )
                tab = refs[(method, C)]
                self.assertTrue(
                    math.isclose(b0.macro_from_vectors(vec), tab["macro"], abs_tol=1e-9)
                )
                for reg in b0.REGIMES:
                    self.assertTrue(
                        math.isclose(
                            np.mean(vec[reg]["money"]),
                            tab["per_regime_money"][reg],
                            abs_tol=1e-9,
                        )
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
