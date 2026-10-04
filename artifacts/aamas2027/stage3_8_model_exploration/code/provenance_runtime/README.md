# provenance_runtime — INFRASTRUCTURE ONLY

`persistent_screen_queue.py` is the recovery scheduler used to orchestrate the
24-cell training matrix and SCREEN-v3 evaluation queue after a mid-campaign
interruption. It is **infrastructure / runtime provenance**, NOT a scientific
method. It made no modeling, training, selection, or evaluation decisions; it
only retried failed/interrupted runs and validated results from disk.

Archived here solely to document how the campaign was executed and recovered.
