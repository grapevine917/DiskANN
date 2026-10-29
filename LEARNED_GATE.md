# Learned-gate research branch

This branch integrates candidate-level gates and whole-search termination into
DiskANN search. It retains the upstream MIT license.

`include/online_gate.h` supports the frozen `LRGATE01`, `STGATE01`, and
`XGGATE01` state-predictor exports used by the Gate experiments. The XGBoost
export contains a bias and a sequence of binary trees; scoring sums leaf values
and applies sigmoid. Cache hits remain protected. This does not imply a recall
guarantee when pruning is enabled.

The three-dataset benchmark drivers, frozen dataset-specific overlays, asset
manifest, and isolated server instructions live in the separate
[Gate repository](https://github.com/grapevine917/Gate), under
`experiments/server/` and `docs/deployment/SERVER_RUNBOOK.md`.

Build this branch on the target machine. Do not copy old static libraries or
executables between machines. Dataset-specific historical overlays (especially
the uint8 100M implementation) must be built and verified separately when
reproducing their results. The Gate server entry point performs this step and
checks LR/MLP/XGBoost/ET outputs against stored native-check samples.

A successful build or disabled-gate parity check does not establish recall,
latency, or I/O improvements. Those require the benchmark protocol and inputs.
