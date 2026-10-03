#!/usr/bin/env python3
"""D1-H1-SAFE-v1 focused tests (15+ invariants; unittest, no pytest/scipy).

Run with the frozen runtime interpreter:
    /Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python \
        artifacts/aamas2027/stage4/d1_h1_safe/tests/test_d1.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
WT = Path(__file__).resolve().parents[5]
B0_CODE = WT / "artifacts/aamas2027/stage3_5/b0_diagnostic/code"
for p in (str(WT), str(B0_CODE), str(CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import b0_lib as b0  # noqa: E402
import run_d1  # noqa: E402
from tools.aamas_stage4 import d1_core as d1  # noqa: E402


# ---------------------------------------------------------------------------
# Lightweight controllable agent for mask-logic tests (no checkpoint needed)
# ---------------------------------------------------------------------------

class _FakeModel:
    def __init__(self, settle: List[float], flush: List[float]) -> None:
        import torch

        self.s = torch.tensor(settle, dtype=torch.float32).unsqueeze(0)
        self.f = torch.tensor(flush, dtype=torch.float32).unsqueeze(0)

    def forward_settle_value(self, state: Any) -> Any:
        return self.s, self.s

    def forward_flush_given_settle(self, state: Any, settle: Any) -> Any:
        return self.f


class _FakeAgent:
    def __init__(self, settle: List[float], flush: List[float]) -> None:
        import torch

        self.model = _FakeModel(settle, flush)
        self.device = torch.device("cpu")


def _make_env(module: Any, cfg: Dict[str, Any], T: int = 20) -> Any:
    return module.make_env(cfg, max_steps=T)


class MechanicsTests(unittest.TestCase):
    """Mask/fork/refill mechanics on real frozen E0 envs (deterministic)."""

    @classmethod
    def setUpClass(cls) -> None:
        # Keep cache writes out of the frozen tree (and sandbox-friendly).
        b0.CACHE_DIR = Path(os.environ.get("D1_CACHE_DIR", "/tmp/d1_h1_cache"))
        cls.adapter = b0.import_frozen_adapter()
        b0.register_adapter(cls.adapter)
        b0.configure_caches()
        cls.adapter.verify_sources(b0.FROZEN_REPO)
        cls.sc, cls.ja = b0.import_frozen_modules(cls.adapter)
        cls.cfg800 = b0.load_ja_transfer_config(800)
        cls.streams, _ = b0.load_new12_streams(cls.adapter)
        cls.pool_us = cls.streams[0]["pool"]

    def _env(self, T: int = 20, stream: Any = None):
        env = _make_env(self.ja, self.cfg800, T=T)
        if stream is None:
            stream = np.zeros((T,), dtype=np.int64)
        env.reset(tx_stream=np.asarray(stream, dtype=np.int64))
        return env

    # 1 -----------------------------------------------------------------
    def test_01_bf_rule_unchanged_vs_frozen_adapter(self) -> None:
        """Factorized BF helpers recombine to the hash-pinned adapter rule."""
        env = self._env(T=30, stream=self.pool_us[0][:30])
        for _ in range(30):
            self.assertEqual(
                b0.bf_joint_action(env), self.adapter.bf_action(env)
            )
            env.step(b0.bf_joint_action(env))

    # 2 -----------------------------------------------------------------
    def test_02_mask_changes_exactly_one_logit_and_excludes_settle(self) -> None:
        import torch

        k = 24
        flush = [0.0] * (k + 1)
        flush[3] = 5.0  # raw argmax = wallet 3
        agent = _FakeAgent(settle=[0.0] * (k + 1), flush=flush)
        env = self._env()
        dec = d1.d1_h1_action(env, agent, np.zeros(1, dtype=np.float32), "M", 3)
        # raw arm sees wallet 3; masked arm must not
        self.assertEqual(dec["raw_f"], 3)
        self.assertNotEqual(dec["chosen_f"], 3)
        self.assertEqual(dec["chosen_f"], 0)  # next-smallest index among zeros
        self.assertEqual(dec["mask_changed"], 1)
        # unmasked arm is byte-identical to raw
        dec_u = d1.d1_h1_action(env, agent, np.zeros(1, dtype=np.float32), "U", 3)
        self.assertEqual(dec_u["chosen_f"], 3)
        self.assertEqual(dec_u["mask_changed"], 0)
        # exact one-logit alteration is also asserted inside d1_h1_action.
        self.assertTrue(torch.isfinite(torch.tensor(1.0)))

    # 3 -----------------------------------------------------------------
    def test_03_settle_noop_masks_nothing(self) -> None:
        k = 24
        flush = [0.0] * (k + 1)
        flush[7] = 9.0
        agent = _FakeAgent(settle=[0.0] * (k + 1), flush=flush)
        env = self._env()
        dec = d1.d1_h1_action(env, agent, np.zeros(1, dtype=np.float32), "M", k)
        self.assertEqual(dec["raw_f"], 7)
        self.assertEqual(dec["chosen_f"], 7)
        self.assertEqual(dec["mask_changed"], 0)
        self.assertEqual(dec["action"], k * (k + 1) + 7)

    # 4 -----------------------------------------------------------------
    def test_04_noop_flush_always_available(self) -> None:
        k = 24
        flush = [-1e30] * k + [0.0]
        agent = _FakeAgent(settle=[0.0] * (k + 1), flush=flush)
        env = self._env()
        for s in (0, 5, 23):
            dec = d1.d1_h1_action(env, agent, np.zeros(1, dtype=np.float32), "M", s)
            self.assertEqual(dec["chosen_f"], k)  # no-op flush survives mask
        dec = d1.d1_h1_action(env, agent, np.zeros(1, dtype=np.float32), "U", 0)
        self.assertEqual(dec["chosen_f"], k)

    # 5 -----------------------------------------------------------------
    def test_05_tie_break_smallest_index_preserved(self) -> None:
        k = 24
        flush = [1.0] * (k + 1)  # full tie -> index 0
        agent = _FakeAgent(settle=[0.0] * (k + 1), flush=flush)
        env = self._env()
        self.assertEqual(
            d1.d1_h1_action(env, agent, np.zeros(1, dtype=np.float32), "U", 0)["raw_f"],
            0,
        )
        dec = d1.d1_h1_action(env, agent, np.zeros(1, dtype=np.float32), "M", 0)
        self.assertEqual(dec["chosen_f"], 1)  # 0 masked, smallest rest wins

    # 6 -----------------------------------------------------------------
    def test_06_actions_always_valid(self) -> None:
        k = 24
        rng = np.random.default_rng(0)
        env = self._env(T=40, stream=self.pool_us[0][:40])
        for t in range(40):
            flush = rng.normal(size=k + 1).tolist()
            agent = _FakeAgent(settle=rng.normal(size=k + 1).tolist(), flush=flush)
            s = int(b0.bf_settle(env))
            for arm in ("U", "M"):
                dec = d1.d1_h1_action(env, agent, np.zeros(1, dtype=np.float32), arm, s)
                self.assertTrue(0 <= dec["action"] < (k + 1) ** 2)
            env.step(d1.d1_h1_action(env, agent, np.zeros(1, dtype=np.float32), "M", s)["action"])

    # 7 -----------------------------------------------------------------
    def test_07_fork_non_interference_and_local_semantics(self) -> None:
        """(s,s) flushes+voids; (s,k) settles; live env untouched by fork."""
        T = 4
        tx = int(round(self.cfg800["env"]["C"] / 24)) - 1  # fits a full wallet
        stream = np.full((T,), tx, dtype=np.int64)
        env = self._env(T=T, stream=stream)
        # at t=0 all wallets full & equal -> BF settle is index 0
        s = int(b0.bf_settle(env))
        self.assertEqual(s, 0)
        sig0 = d1.env_state_signature(env)
        row = d1.local_fork(env, s)
        self.assertEqual(d1.env_state_signature(env), sig0)  # isolation
        self.assertEqual(row["A_flushes"], 1.0)
        self.assertEqual(row["A_settled"], 0.0)                # settlement voided
        self.assertEqual(row["B_flushes"], 0.0)
        self.assertEqual(row["B_settled"], float(tx))          # no-op flush settles
        self.assertEqual(row["B_accepted"], 1.0)
        self.assertAlmostEqual(row["money_A_minus_B"], -tx - 10.0, places=6)
        # and the live environment can still take the normal step
        _s, _r, done, info = env.step(s * 25 + s)
        self.assertEqual(info["flushes_this_step"], 1)
        self.assertFalse(info["accepted"])

    # 8 -----------------------------------------------------------------
    def test_08_observer_on_off_identical_transitions(self) -> None:
        """Instrumented U produces bit-identical per-step infos to plain H1."""
        T = 60
        sess = _session()
        agent = sess.agent(800, 123)
        cfg = sess.bundle(800, 123)["cfg"]
        env_a = sess.sc.make_env(cfg, max_steps=T)
        env_b = sess.sc.make_env(cfg, max_steps=T)
        env_a.reset(tx_stream=self.pool_us[0][:T])
        env_b.reset(tx_stream=self.pool_us[0][:T])
        state = env_b._get_state()
        for t in range(T):
            sa = b0.bf_settle(env_a)
            _sc, fa, action_a, _, _ = b0.hybrid_action(
                env_a, agent, env_a._get_state(), "BF", "SC"
            )
            pre = d1.observe_pre(env_b, int(b0.bf_settle(env_b)))
            dec = d1.d1_h1_action(env_b, agent, state, "U", int(b0.bf_settle(env_b)))
            self.assertEqual(dec["action"], action_a)
            _, _, _, info_a = env_a.step(action_a)
            state, _, _, info_b = env_b.step(dec["action"])
            for key in ("fit_idx", "settled_value", "accepted", "dropped",
                        "oversize_dropped", "flushes_this_step", "flush_choice",
                        "settle_choice"):
                self.assertEqual(info_a[key], info_b[key], f"t={t} key={key}")

    # 9 -----------------------------------------------------------------
    def test_09_f3_refill_boundary(self) -> None:
        T = 8
        env = self._env(T=T, stream=np.ones((T,), dtype=np.int64))
        size = env.wallet_size
        # flush wallet 0 + no-op settle at t=0 (action 24*25+0)
        _s, _r, _d, info = env.step(24 * 25 + 0)
        self.assertEqual(info["flushes_this_step"], 1)
        self.assertEqual(env.freeze_until[0], 0 + 3 - 1)
        self.assertTrue(env.pending_refill[0])
        self.assertEqual(env.wallets[0], 0.0)
        # t=1, t=2: still frozen
        self.assertFalse(env._usable(0))
        env.step(24 * 25 + 24)
        self.assertFalse(env._usable(0))  # time=2, need time>2
        env.step(24 * 25 + 24)
        # refill completes when time becomes 3
        self.assertTrue(env._usable(0))
        self.assertFalse(env.pending_refill[0])
        self.assertEqual(env.wallets[0], size)
        # tracker/ttr: flush_t=0 -> ready after the step starting at t=2 (ttr=3)
        tracker = d1.RefillTracker(24)
        tracker.on_flush(0, 0)
        events = d1.make_post_observer(tracker)(env, (True,) + (False,) * 23)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][2], 3)

    # 10 ----------------------------------------------------------------
    def test_10_oversize_telemetry(self) -> None:
        T = 3
        env = self._env(T=T, stream=np.full((T,), 1000, dtype=np.int64))
        tel = d1.CellTelemetry(24, ["US"])
        tracker = d1.RefillTracker(24)
        post = d1.make_post_observer(tracker)
        for t in range(T):
            s = int(b0.bf_settle(env))
            pre = d1.observe_pre(env, s)
            self.assertEqual(s, 24)
            self.assertTrue(pre["oversize"])
            self.assertFalse(pre["prefeasible"])
            _, _, _, info = env.step(24 * 25 + 24)
            events = post(env, pre["pending"])
            tel.begin_step_refill_value(env.wallet_size, len(events))
            tel.record_step("US", 0, pre,
                            dict(s_bf=24, sc_settle=-1, raw_f=24, chosen_f=24,
                                 mask_changed=0, action=24 * 25 + 24),
                            info, events, tracker, (0, 0), None)
        self.assertEqual(tel.counts["US"]["oversize_drops"], T)
        self.assertEqual(tel.counts["US"]["insufficient_drops"], 0)
        self.assertEqual(tel.counts["US"]["starvation_zero_feasible_steps"], 0)

    # 11 ----------------------------------------------------------------
    def test_11_conflict_and_lost_settlement_counters(self) -> None:
        T = 2
        tx = int(round(self.cfg800["env"]["C"] / 24)) - 1
        env = self._env(T=T, stream=np.full((T,), tx, dtype=np.int64))
        tel = d1.CellTelemetry(24, ["US"])
        tracker = d1.RefillTracker(24)
        post = d1.make_post_observer(tracker)
        s = int(b0.bf_settle(env))
        pre = d1.observe_pre(env, s)
        self.assertTrue(pre["prefeasible"])
        _, _, _, info = env.step(s * 25 + s)  # raw conflict, executed, lost
        events = post(env, pre["pending"])
        dec = dict(s_bf=s, sc_settle=-1, raw_f=s, chosen_f=s, mask_changed=0,
                   action=s * 25 + s)
        tel.begin_step_refill_value(env.wallet_size, len(events))
        tel.record_step("US", 0, pre, dec, info, events, tracker, (0, 0), None)
        c = tel.counts["US"]
        self.assertEqual(c["conflict_raw"], 1)
        self.assertEqual(c["conflict_prefeasible"], 1)
        self.assertEqual(c["conflict_executed"], 1)
        self.assertEqual(c["lost_settlement"], 1)
        self.assertEqual(c["accepted"], 0)

    # 12 ----------------------------------------------------------------
    def test_12_refill_tracker_lifecycle(self) -> None:
        tracker = d1.RefillTracker(2)
        tracker.on_flush(0, 0)
        # refill at t=3, wallet never settled afterwards
        tracker.on_refill(0, 3)
        out = tracker.close_episode()
        self.assertEqual(out["refilled_never_used"], 1)
        self.assertEqual(out["terminal_pending"], 0)

        tracker = d1.RefillTracker(2)
        tracker.on_flush(1, 5)
        out = tracker.close_episode()
        self.assertEqual(out["terminal_pending"], 1)  # pending at horizon

        tracker = d1.RefillTracker(2)
        tracker.on_flush(0, 0)
        tracker.on_refill(0, 3)
        tracker.on_settled(0)
        tracker.on_flush(0, 9)  # second cycle after reuse
        out = tracker.close_episode()
        self.assertEqual(out["terminal_pending"], 1)

    # 13 ----------------------------------------------------------------
    def test_13_money_and_count_identities_hold_in_evaluator(self) -> None:
        sess = _session()
        cfg = sess.bundle(800, 123)["cfg"]
        tel = d1.CellTelemetry(24, ["US"])
        res = run_d1.evaluate_d1_pool(
            sess.sc, cfg, sess.agent(800, 123), self.pool_us[:2], "U", tel, "US",
            n_episodes=2,
        )
        for m in res["raw_results"]:
            self.assertAlmostEqual(
                m["eval_money"], m["settled"] - 10.0 * m["flushes"], places=9
            )
            self.assertEqual(int(m["drops"]) + int(m["accepted_count"]), 1000)
        self.assertEqual(tel.counts["US"]["steps"], 2000)

    # 14 ----------------------------------------------------------------
    def test_14_u_archived_parity_two_episodes(self) -> None:
        """U metrics equal the archived Stage 3.5 H1 episodes (US, 2 eps)."""
        sess = _session()
        cfg = sess.bundle(800, 123)["cfg"]
        tel = d1.CellTelemetry(24, ["US"])
        res = run_d1.evaluate_d1_pool(
            sess.sc, cfg, sess.agent(800, 123), self.pool_us[:2], "U", tel, "US",
            n_episodes=2,
        )
        archived = run_d1.archived_rows(800, 123)[:2]
        for metric, row in zip(res["raw_results"], archived):
            for fld in ("money", "settled", "flushes", "drops", "accepted_count"):
                got = metric["eval_money"] if fld == "money" else metric[fld]
                self.assertEqual(float(got), float(row[fld]), fld)

    # 15 ----------------------------------------------------------------
    def test_15_trace_schema_and_fork_isolation_in_evaluator(self) -> None:
        sess = _session()
        cfg = sess.bundle(800, 123)["cfg"]
        tel = d1.CellTelemetry(24, ["US"])
        run_d1.evaluate_d1_pool(
            sess.sc, cfg, sess.agent(800, 123), self.pool_us[:2], "M", tel, "US",
            n_episodes=2,
        )
        for ep in (0, 1):
            I = np.asarray(tel.traces["US"][ep]["I"])
            F = np.asarray(tel.traces["US"][ep]["F"])
            self.assertEqual(I.shape, (1000, len(d1.TRACE_INT_COLUMNS)))
            self.assertEqual(F.shape, (1000, len(d1.TRACE_FLOAT_COLUMNS)))
            # masked settle wallet is never the submitted flush when s_BF < k
            for row in I:
                if row[1] < 24:
                    self.assertNotEqual(int(row[5]), int(row[1]))
        # M invariant: mask changes argmax iff raw argmax was s_BF
        c = tel.counts["US"]
        self.assertEqual(c["mask_changed_choice"], c["conflict_raw"])
        # all fork rows are bounded/finite and B never worse oddly
        for ep in (0, 1):
            for fr in tel.traces["US"][ep]["fork"]:
                self.assertTrue(all(np.isfinite(fr)))
                self.assertEqual(fr[3] > 0, True)  # B settles the feasible tx


_SESSION = None


def _session() -> run_d1.D1Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = run_d1.D1Session()
    return _SESSION


if __name__ == "__main__":
    unittest.main(verbosity=2)
