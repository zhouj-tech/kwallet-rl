# Conditional Factorized AC Summary

This folder keeps only the useful final analysis files for Conditional Factorized AC.

Selected model:
Conditional Factorized AC, E=32, H=256.

Main setting:
k=24, C in {800, 1200}, F=3, T=1000, 10 seeds.

Files:
- main_result_table.csv: main comparison table against Basic PPO and Factorized AC.
- key_paired_evidence.csv: key paired-difference evidence for statistical comparison.
- paper_writeup.md: method, ablation, and result interpretation text.

Main conclusion:
Conditional Factorized AC preserves the 92% policy-output reduction at k=24 while improving over independent Factorized AC. Under C=800,k=24, it significantly outperforms Basic PPO. Under C=1200,k=24, it remains comparable to Basic PPO and achieves significantly higher money score.
