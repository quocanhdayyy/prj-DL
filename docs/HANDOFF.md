# Handoff — 21/10/2026

## Current phase

Prepare the existing DDI/MLP environment only if the user asks. The plan is written; installation and model runtime have not happened in this phase.

## Confirmed state

- Branch at snapshot: `feat/mlp`.
- MLP implementation and `--architecture mlp` already exist in `model.py`; do not build a second model.
- The installation-only plan is [reports/MLP_IMPLEMENTATION_PLAN.md](../reports/MLP_IMPLEMENTATION_PLAN.md).
- The user-provided DOCX conversion is [Bao_cao_y_tuong_Deep_Learning_DDI_hoan_chinh.md](../Bao_cao_y_tuong_Deep_Learning_DDI_hoan_chinh.md).
- No environment installation, preprocessing, tests, model execution, training, evaluation, or prediction has been done in this phase.
- Snapshot has no `data/processed/` or `runs/` artifacts. The tracked root CSV/NPZ files are raw/legacy inputs; do not treat them as newly generated splits.
- `VALIDATION.md` is historical macOS evidence dated 19/10/2026, not a check of this environment.

## Next actions and limits

1. If the user asks for installation, confirm branch/status and follow the installation-only scope in the plan.
2. Record actual package/syntax checks and remaining limits in the relevant docs and `progress.md`.
3. Stop before preprocessing, tests, model construction, training, evaluation, or prediction unless the user expands the scope.

## Read by need

- [AGENTS.md](../AGENTS.md): repo routing and collaboration preferences.
- [RULES.md](RULES.md): data, Git, validation, and documentation-update rules.
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md): source-of-truth map.
- [progress.md](../progress.md): implementation status log.
- [VALIDATION.md](../VALIDATION.md): dated validation evidence.
- [README.md](../README.md): commands and pipeline contracts.

The handoff is a current snapshot; `progress.md` is the implementation log, `VALIDATION.md` is evidence, and the plan is proposed work. Update this file only when a material status change affects next steps, blockers, or artifacts.
