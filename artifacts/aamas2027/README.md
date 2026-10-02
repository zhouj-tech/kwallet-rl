# AAMAS 2027 — K-Wallet Artifacts

This directory stores frozen experimental and derived analysis artifacts for
the AAMAS 2027 K-Wallet study.

## Stage 2B Raw Freeze

Raw bundle:

`stage2/raw/KWALLET_AAMAS_STAGE2B_RAW_20261002.tar.gz`

Freeze date: 2026-10-02

SHA256:

`576c9baa00c4438543ab9f9c2063f65bb26b1a6f287974a9e8ab91b18189ace3`

Scientific code lineage:

`5573ec642f0f28c218f3e6058478f62ab6db6b2b`

Scientific status:

- 42 Stage 2B evaluation outputs available
- NEW12-v1 held-out evaluation
- Mac CPU runtime approved after exact cross-platform parity validation
- Frozen evaluator, job matrix, and transaction streams
- No principal-policy retraining during Stage 2B evaluation

The Stage 2B raw bundle is immutable.

## Stage 3

Derived statistical analyses, tables, figures, and reports belong under:

`stage3/`

Stage 3 analysis must read from the frozen Stage 2B results and must not
modify the raw freeze.
