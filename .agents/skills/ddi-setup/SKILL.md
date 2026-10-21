---
name: ddi-setup
description: Prepare or install the existing DDI/MLP project environment when the user asks for setup or installation. Do not use for documentation-only, model implementation, training, evaluation, or prediction tasks.
---

# DDI environment setup

Use this skill only for a user-requested environment preparation or installation task in this repository. First read the current [MLP implementation plan](../../../reports/MLP_IMPLEMENTATION_PLAN.md), then follow the repository [rules](../../../docs/RULES.md) and [README](../../../README.md).

The MLP already exists in `model.py`; reuse its existing entrypoints. Follow the latest plan's environment and scope: the current workflow completes code first, then verifies runtime and trains on Colab. Do not infer that an installation request authorizes preprocessing, model construction, tests, training, evaluation, or prediction. Report what was actually checked and distinguish package/syntax checks from runtime validation.
