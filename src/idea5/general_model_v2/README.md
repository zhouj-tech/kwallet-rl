# General Collateral Model v2: Two-Pool Extension

This package is a controlled extension between the one-pool General Collateral Model and the full K-Wallet environment.

## Purpose

The goal is to test whether conditional action factorization remains useful when the collateral problem becomes more structured but still stays simpler than K-Wallet.

## Key Design

- Two independent collateral pools: Pool A and Pool B.
- Each transaction has a value and a type.
- Type 0 transactions use Pool A.
- Type 1 transactions use Pool B.
- The policy chooses:
  - `settle_action`: discard or accept.
  - `flush_action`: no flush, flush Pool A, or flush Pool B.
- Flush cost is per flush event.
- The money objective is unchanged:
  `money = money_p * settled_value - money_tau * flushes`.

## Action Space

For `flush_levels = 17`:

- `flush_action = 0`: no flush.
- `flush_action = 1..16`: flush Pool A by fractions `1/16..1`.
- `flush_action = 17..32`: flush Pool B by fractions `1/16..1`.
- `num_flush_choices = 33`.
- Flat joint action size is `66`.
- Factorized/conditional policy output size is `35`.

The explicit indexing check is:

- `flush_action = 17` decodes to Pool B with fraction `1/16`.

## Models

The first v2 version includes:

- Grid Threshold baseline.
- Flat PPO.
- Independent Factorized AC.
- Conditional Factorized AC.

No threshold imitation warm start is included in v2 first version.

## Data

Two-pool pools are stored as `.npz` files with:

- `values`: shape `[episodes, T]`.
- `types`: shape `[episodes, T]`, values 0 or 1.

Use:

```bash
python src/idea5/general_model_v2/scripts/generate_two_pool_pools.py --T 1000 --seed 123
```

The generator loads v1 value pools and assigns deterministic transaction types with `type_prob_A = 0.5`.

## Experiment Plan

First run smoke tests for pool generation, threshold, and conditional AC. Then run a small saved comparison at `C=1000`, `seed=123` for Grid Threshold and Stable Conditional AC. Do not run formal multi-seed experiments until those checks pass.
