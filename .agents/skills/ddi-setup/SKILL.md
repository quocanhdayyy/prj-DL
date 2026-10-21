---
name: ddi-setup
description: Prepare or install the existing DDI/MLP project environment when the user asks for setup or installation. Do not use for documentation-only, model implementation, training, evaluation, or prediction tasks.
---

# DDI environment setup

Use this skill only for a user-requested environment preparation or installation task in this repository. First read the current [MLP implementation plan](../../../reports/MLP_IMPLEMENTATION_PLAN.md), then follow the repository [rules](../../../docs/RULES.md) and [README](../../../README.md).

The MLP already exists in `model.py`; reuse its existing entrypoints. Keep plan-only, installation-only, and runtime work distinct. The current plan is installation-only: do not import/construct a model, preprocess data, run tests, train, evaluate, or predict unless the user expands the scope. Report what was actually checked and distinguish package/syntax checks from runtime validation.
