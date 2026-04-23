# Architecture

See `MASTER_PROMPT.md` §3 for the canonical diagram. This document expands it once concrete modules land.

## Components (target)

- **Orchestrator** (`automl.api`) — sklearn-compatible facade.
- **Search Engine** (`automl.search.engine`) — Optuna study, TPE sampler, ASHA pruner.
- **Pipeline Factory** (`automl.preprocessing.factory` + `automl.feature_selection.factory` + `automl.models.registry`) — assembles a sklearn `Pipeline` from a sampled config.
- **CV Evaluator** (`automl.evaluation`) — fold-aware scoring with pruner intermediate reports.
- **Meta Store** (`automl.meta`) — fingerprint dataset → retrieve top-N historical configs to seed Optuna.
- **Ensembler** (`automl.ensembling`) — top-k stacking with leakage-free OOF predictions.
- **Explainer** (`automl.explain`) — SHAP + permutation + markdown report.
- **Run Recorder** (`automl.runs`) — config + data hash + git SHA + artifacts + `run_card.md`.
