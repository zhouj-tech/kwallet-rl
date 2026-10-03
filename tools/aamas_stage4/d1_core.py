#!/usr/bin/env python3
"""D1-H1-SAFE-v1 core mechanics.

Three additive mechanisms on top of the *frozen* E0 environment and the
*frozen* SC-FAC conditional policy:

1. ``d1_decision`` — H1 action with the settlement-wallet-masked flush head.
   Arm "U" is metric-identical to archived H1; arm "M" replaces exactly one
   flush logit (the BF settle index) with -infinity when s_BF < k.

2. ``observe_pre`` / ``RefillTracker`` / ``CellTelemetry`` — bounded read-only
   step telemetry. Observers never write to the environment, so transitions
   are identical with instrumentation on/off (proven by tests + U parity).

3. ``local_fork`` — read-only one-step snapshot fork (s,s) vs (s,k) at raw
   conflict states. The live environment is never advanced by a fork.

Post-hoc exploratory diagnostic -- NOT confirmatory.
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Fixed trace schema (predeclared; traces are only episode 0 and 1 per regime)
# ---------------------------------------------------------------------------

TRACE_INT_COLUMNS = [
    "step",
    "s_bf",
    "n_feasible",
    "n_usable",
    "raw_f",
    "chosen_f",
    "requested_f",
    "executed_f",  # -1 when no flush executed
    "noop_flush",
    "unusable_flush",
    "conflict_raw",
    "conflict_executed",
    "lost_settlement",
    "sc_settle_cf",  # -1 for BF cells
    "sc_settle_differs",
    "pending_n",
    "frozen_n",
    "ready_n",
    "refill_completed",
    "starvation_zero_usable",
    "starvation_zero_feasible",
    "accepted",
]
TRACE_FLOAT_COLUMNS = [
    "tx",
    "bal_s",
    "usable_balance",
    "refilled_value",
    "settled_value",
]
# fork columns: step, tx, A_settled, B_settled, A_flushes, B_flushes,
# money_A_minus_B (immediate, LOCAL EXPOSURE ONLY)
TRACE_FORK_COLUMNS = [
    "step",
    "tx",
    "A_settled",
    "B_settled",
    "A_flushes",
    "B_flushes",
    "money_A_minus_B",
]
TRACED_EPISODES = (0, 1)

HIST_EDGES = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)


def hist_bin(value: int, edges: Tuple[int, ...] = HIST_EDGES) -> str:
    for e in edges:
        if value <= e:
            return f"<={e}"
    return f">{edges[-1]}"


# ---------------------------------------------------------------------------
# Environment snapshotting (no RNG / cache mutation occurs on stream rollouts)
# ---------------------------------------------------------------------------

_CLONED_SCALARS = (
    "C",
    "k",
    "F",
    "wallet_size",
    "time",
    "current_tx",
    "total_settled",
    "total_accepted",
    "num_flushes",
    "drops",
    "oversize_drops",
    "insufficient_drops",
)


def light_clone_env(env: Any) -> Any:
    """Cheap functional clone for the read-only local fork.

    Copies only the mutable rollout state; the immutable tx stream, the
    prebuilt context caches and the RNG are shared (never written during a
    fixed-stream rollout).
    """
    clone = copy.copy(env)
    clone.wallets = list(env.wallets)
    clone.freeze_until = list(env.freeze_until)
    clone.pending_refill = list(env.pending_refill)
    clone.tx_history = deque(env.tx_history, maxlen=env.tx_history.maxlen)
    for name in _CLONED_SCALARS:
        setattr(clone, name, getattr(env, name))
    return clone


def env_state_signature(env: Any) -> Tuple[Any, ...]:
    """Full mutable-state signature (used by tests for fork isolation)."""
    return (
        tuple(env.wallets),
        tuple(env.freeze_until),
        tuple(env.pending_refill),
        tuple(env.tx_history),
        env.time,
        env.current_tx,
        env.total_settled,
        env.total_accepted,
        env.num_flushes,
        env.drops,
        env.oversize_drops,
        env.insufficient_drops,
        env._get_state().tobytes(),
    )


# ---------------------------------------------------------------------------
# Read-only observation
# ---------------------------------------------------------------------------

def observe_pre(env: Any, s_bf: int) -> Dict[str, Any]:
    tx = env.current_tx
    usable = [i for i in range(env.k) if env._usable(i)]
    feasible = [i for i in usable if env.wallets[i] >= tx]
    oversize = tx > env.wallet_size
    prefeasible = (
        s_bf < env.k
        and not oversize
        and env._usable(s_bf)
        and env.wallets[s_bf] >= tx
    )
    return dict(
        tx=float(tx),
        s_bf=int(s_bf),
        oversize=bool(oversize),
        prefeasible=bool(prefeasible),
        bal_s=float(env.wallets[s_bf]) if s_bf < env.k else -1.0,
        n_feasible=len(feasible),
        n_usable=len(usable),
        usable_balance=float(sum(env.wallets[i] for i in usable)),
        pending_n=int(sum(env.pending_refill)),
        frozen_n=int(env.k - len(usable)),
        ready_n=len(usable),
        pending=tuple(env.pending_refill),
        time=int(env.time),
    )


def make_post_observer(tracker: "RefillTracker"):
    def post_events(env: Any, pre_pending: Tuple[bool, ...]):
        out = []
        for i in range(env.k):
            if pre_pending[i] and not env.pending_refill[i]:
                c = tracker.cycle[i]
                ttr = int(env.time) - int(c["flush_t"])
                out.append((i, int(env.time), ttr))
        return out

    return post_events


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def d1_h1_action(
    env: Any,
    agent: Any,
    state: np.ndarray,
    arm: str,
    s_bf: int,
) -> Dict[str, int]:
    """H1 decision with raw and (arm M) settlement-masked flush argmax.

    ``s_bf`` is precomputed by the frozen BF settle rule (the evaluator needs
    it before the decision for pre-feasibility telemetry). The SC settle head
    is evaluated for counterfactual telemetry only and NEVER controls the
    rollout.
    """
    import torch

    assert arm in ("U", "M")
    model, device, k = agent.model, agent.device, env.k
    s = int(s_bf)
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        settle_logits, _ = model.forward_settle_value(state_t)
    if not bool(torch.isfinite(settle_logits).all()):
        raise RuntimeError("Non-finite settle logits")
    sc_settle = int(torch.argmax(settle_logits, dim=-1).item())

    s_t = torch.tensor([s], dtype=torch.long, device=device)
    with torch.no_grad():
        flush_logits = model.forward_flush_given_settle(state_t, s_t)
    if not bool(torch.isfinite(flush_logits).all()):
        raise RuntimeError("Non-finite flush logits")
    raw_f = int(torch.argmax(flush_logits, dim=-1).item())

    if arm == "M" and s < k:
        masked_logits = flush_logits.clone()
        masked_logits[0, s] = torch.finfo(masked_logits.dtype).min
        # exactly one logit changed (local proof, no other entry touched)
        assert int((masked_logits != flush_logits).sum().item()) == 1
        chosen_f = int(torch.argmax(masked_logits, dim=-1).item())
    else:
        chosen_f = raw_f

    assert 0 <= s <= k and 0 <= raw_f <= k and 0 <= chosen_f <= k
    return dict(
        s_bf=s,
        sc_settle=sc_settle,
        raw_f=raw_f,
        chosen_f=chosen_f,
        mask_changed=int(chosen_f != raw_f),
        action=s * (k + 1) + chosen_f,
    )


def bf_decision(env: Any, s_bf: int, bf_flush_fn: Any) -> Dict[str, int]:
    s = int(s_bf)
    f = int(bf_flush_fn(env, s))
    return dict(
        s_bf=s,
        sc_settle=-1,
        raw_f=f,
        chosen_f=f,
        mask_changed=0,
        action=s * (env.k + 1) + f,
    )


# ---------------------------------------------------------------------------
# Read-only local fork
# ---------------------------------------------------------------------------

def local_fork(env: Any, s_bf: int) -> Dict[str, float]:
    """Fork (s,s) vs (s,k) from a snapshot. The live env is never touched."""
    base = env.k + 1
    env_a = light_clone_env(env)
    env_b = light_clone_env(env)
    _sa, _ra, _da, info_a = env_a.step(s_bf * base + s_bf)
    _sb, _rb, _db, info_b = env_b.step(s_bf * base + env.k)
    settled_a = float(info_a["settled_value"])
    settled_b = float(info_b["settled_value"])
    fa = float(info_a["flushes_this_step"])
    fb = float(info_b["flushes_this_step"])
    return dict(
        step=float(env.time),
        tx=float(env.current_tx),
        A_settled=settled_a,
        B_settled=settled_b,
        A_flushes=fa,
        B_flushes=fb,
        money_A_minus_B=(settled_a - settled_b) - 10.0 * (fa - fb),
        B_accepted=float(bool(info_b["accepted"])),
    )


# ---------------------------------------------------------------------------
# Refill cycle accounting + telemetry aggregation
# ---------------------------------------------------------------------------

class RefillTracker:
    """Tracks flush -> refill -> reused cycles within one episode."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.cycle: List[Optional[Dict[str, Any]]] = [None] * k

    def on_flush(self, i: int, t: int) -> None:
        prev = self.cycle[i]
        if prev is not None and prev["ready_t"] is None:
            raise RuntimeError("flush on pending wallet (environment invariant broken)")
        self.cycle[i] = dict(flush_t=t, ready_t=None, used=False)

    def on_refill(self, i: int, ready_t: int) -> None:
        c = self.cycle[i]
        if c is None:
            raise RuntimeError("refill without recorded flush")
        c["ready_t"] = ready_t

    def on_settled(self, i: int) -> None:
        c = self.cycle[i]
        if c is not None and c["ready_t"] is not None:
            c["used"] = True

    def close_episode(self) -> Dict[str, int]:
        never_used = 0
        terminal_pending = 0
        for c in self.cycle:
            if c is None:
                continue
            if c["ready_t"] is None:
                terminal_pending += 1
            elif not c["used"]:
                never_used += 1
        return dict(refilled_never_used=never_used, terminal_pending=terminal_pending)


class _Hist:
    def __init__(self) -> None:
        self.bins: Dict[str, int] = {}

    def add(self, value: int) -> None:
        key = hist_bin(int(value))
        self.bins[key] = self.bins.get(key, 0) + 1


class CellTelemetry:
    """Bounded per-cell counters/histograms plus the fixed-subset traces."""

    COUNT_KEYS = (
        "steps",
        "settle_bf_feasible",
        "settle_bf_noop",
        "conflict_raw",
        "conflict_prefeasible",
        "conflict_executed",
        "lost_settlement",
        "flush_requested_real",
        "flush_executed",
        "flush_noop",
        "flush_unusable",
        "mask_changed_choice",
        "accepted",
        "insufficient_drops",
        "oversize_drops",
        "refill_completions",
        "starvation_zero_usable_steps",
        "starvation_zero_feasible_steps",
        "episodes_refilled_never_used",
        "episodes_terminal_pending",
    )

    def __init__(self, k: int, regimes: List[str]) -> None:
        self.k = int(k)
        self.regimes = list(regimes)
        self.counts: Dict[str, Dict[str, int]] = {
            r: {key: 0 for key in self.COUNT_KEYS} for r in regimes
        }
        self.sums: Dict[str, Dict[str, float]] = {
            r: dict(refilled_value=0.0) for r in regimes
        }
        self.ttr_hist: Dict[str, _Hist] = {r: _Hist() for r in regimes}
        self.hard_starve_hist: Dict[str, _Hist] = {r: _Hist() for r in regimes}
        self.feas_starve_hist: Dict[str, _Hist] = {r: _Hist() for r in regimes}
        self.fork: Dict[str, Dict[str, float]] = {
            r: dict(
                fork_events=0.0,
                tx_sum=0.0,
                settled_A_sum=0.0,
                settled_B_sum=0.0,
                flushes_A_sum=0.0,
                flushes_B_sum=0.0,
                exposure_A_minus_B_sum=0.0,
                B_accepted_sum=0.0,
            )
            for r in regimes
        }
        self.traces: Dict[str, Dict[int, Dict[str, List[Any]]]] = {
            r: {} for r in regimes
        }

    # -- one step ---------------------------------------------------------

    def record_step(
        self,
        regime: str,
        ep: int,
        pre: Dict[str, Any],
        dec: Dict[str, int],
        info: Dict[str, Any],
        refill_events: List[Tuple[int, int, int]],
        tracker: RefillTracker,
        starve_flags: Tuple[int, int],
        fork_row: Optional[Dict[str, float]],
    ) -> None:
        c = self.counts[regime]
        k = self.k
        c["steps"] += 1
        if dec["s_bf"] < k:
            c["settle_bf_feasible"] += 1
        else:
            c["settle_bf_noop"] += 1

        raw_f, chosen_f = dec["raw_f"], dec["chosen_f"]
        executed = int(info["flushes_this_step"])
        conflict_raw = int(raw_f == dec["s_bf"] and dec["s_bf"] < k)
        conflict_exec = int(
            conflict_raw and executed == 1 and info["flush_choice"] == dec["s_bf"]
        )
        lost = int(pre["prefeasible"] and conflict_exec and not bool(info["accepted"]))

        c["conflict_raw"] += conflict_raw
        c["conflict_prefeasible"] += int(conflict_raw and pre["prefeasible"])
        c["conflict_executed"] += conflict_exec
        c["lost_settlement"] += lost
        c["flush_requested_real"] += int(chosen_f < k)
        c["flush_noop"] += int(chosen_f == k)
        c["flush_unusable"] += int(chosen_f < k and executed == 0)
        c["flush_executed"] += executed
        c["mask_changed_choice"] += int(dec.get("mask_changed", 0))
        c["accepted"] += int(bool(info["accepted"]))
        c["oversize_drops"] += int(bool(info["oversize_dropped"]))
        c["insufficient_drops"] += int(bool(info["dropped"]))
        c["starvation_zero_usable_steps"] += int(starve_flags[0])
        c["starvation_zero_feasible_steps"] += int(starve_flags[1])

        for i, ready_t, ttr in refill_events:
            tracker.on_refill(i, ready_t)
            c["refill_completions"] += 1
            self.ttr_hist[regime].add(ttr)
        if refill_events:
            # total restored value this step, supplied by the evaluator
            self.sums[regime]["refilled_value"] += self._pending_refill_value

        if bool(info["accepted"]) and info["fit_idx"] is not None:
            tracker.on_settled(int(info["fit_idx"]))

        if fork_row is not None:
            fk = self.fork[regime]
            fk["fork_events"] += 1
            fk["tx_sum"] += fork_row["tx"]
            fk["settled_A_sum"] += fork_row["A_settled"]
            fk["settled_B_sum"] += fork_row["B_settled"]
            fk["flushes_A_sum"] += fork_row["A_flushes"]
            fk["flushes_B_sum"] += fork_row["B_flushes"]
            fk["exposure_A_minus_B_sum"] += fork_row["money_A_minus_B"]
            fk["B_accepted_sum"] += fork_row["B_accepted"]

        if ep in TRACED_EPISODES:
            self._trace_row(
                regime, ep, pre, dec, info, refill_events, starve_flags,
                conflict_raw, conflict_exec, lost, executed, fork_row,
            )

    _pending_refill_value = 0.0

    def _trace_row(
        self,
        regime: str,
        ep: int,
        pre: Dict[str, Any],
        dec: Dict[str, int],
        info: Dict[str, Any],
        refill_events: List[Tuple[int, int, int]],
        starve_flags: Tuple[int, int],
        conflict_raw: int,
        conflict_exec: int,
        lost: int,
        executed_flush: int,
        fork_row: Optional[Dict[str, float]],
    ) -> None:
        bucket = self.traces[regime].setdefault(ep, {"I": [], "F": [], "fork": []})
        k = self.k
        executed_idx = int(info["flush_choice"]) if executed_flush else -1
        bucket["I"].append(
            [
                pre["time"],
                dec["s_bf"],
                pre["n_feasible"],
                pre["n_usable"],
                dec["raw_f"],
                dec["chosen_f"],
                dec["chosen_f"],
                executed_idx,
                int(dec["chosen_f"] == k),
                int(dec["chosen_f"] < k and executed_flush == 0),
                conflict_raw,
                conflict_exec,
                lost,
                dec["sc_settle"],
                int(dec["sc_settle"] != dec["s_bf"]) if dec["sc_settle"] >= 0 else 0,
                pre["pending_n"],
                pre["frozen_n"],
                pre["ready_n"],
                len(refill_events),
                starve_flags[0],
                starve_flags[1],
                int(bool(info["accepted"])),
            ]
        )
        bucket["F"].append(
            [
                pre["tx"],
                pre["bal_s"],
                pre["usable_balance"],
                self._pending_refill_value,
                float(info["settled_value"]),
            ]
        )
        if fork_row is not None:
            bucket["fork"].append(
                [
                    fork_row["step"],
                    fork_row["tx"],
                    fork_row["A_settled"],
                    fork_row["B_settled"],
                    fork_row["A_flushes"],
                    fork_row["B_flushes"],
                    fork_row["money_A_minus_B"],
                ]
            )

    # -- episode/run lifecycle -------------------------------------------

    def begin_step_refill_value(self, wallet_size: float, n_refills: int) -> None:
        self._pending_refill_value = float(wallet_size) * int(n_refills)

    def close_episode(self, regime: str, tracker: RefillTracker) -> None:
        out = tracker.close_episode()
        self.counts[regime]["episodes_refilled_never_used"] += out["refilled_never_used"]
        self.counts[regime]["episodes_terminal_pending"] += out["terminal_pending"]

    def add_starvation_run(self, regime: str, kind: str, length: int) -> None:
        (self.hard_starve_hist if kind == "hard" else self.feas_starve_hist)[
            regime
        ].add(length)

    def summary(self) -> Dict[str, Any]:
        return dict(
            counts=self.counts,
            sums=self.sums,
            ttr_hist={r: dict(h.bins) for r, h in self.ttr_hist.items()},
            hard_starvation_hist={
                r: dict(h.bins) for r, h in self.hard_starve_hist.items()
            },
            feasible_starvation_hist={
                r: dict(h.bins) for r, h in self.feas_starve_hist.items()
            },
            forks=self.fork,
        )
