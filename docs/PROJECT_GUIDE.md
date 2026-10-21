# Project guide

Use these files as the sources of truth; do not copy their contents into this index.

- [HANDOFF.md](HANDOFF.md): concise continuity snapshot and next actions; read first when resuming work.
- [progress.md](../progress.md): chronological implementation status, completed changes, evidence, and pending work.
- [VALIDATION.md](../VALIDATION.md): validation evidence with its original date and environment; distinguish it from fresh checks.
- [reports/MLP_IMPLEMENTATION_PLAN.md](../reports/MLP_IMPLEMENTATION_PLAN.md): code-first plan, with runtime and training deferred to Colab; proposed steps are not evidence of execution.
- [MLP_GUIDE.md](MLP_GUIDE.md): reusable lessons from CNN–BiLSTM, MLP/data contracts, validation/test selection and future Colab commands.
- [README.md](../README.md): supported pipeline, commands, data contracts, and usage limits.
- [reports/bao_cao_y_tuong_DDI.md](../reports/bao_cao_y_tuong_DDI.md): original research proposal.
- [Bao_cao_y_tuong_Deep_Learning_DDI_hoan_chinh.md](../Bao_cao_y_tuong_Deep_Learning_DDI_hoan_chinh.md): faithful Markdown conversion of the supplied DOCX.

The MLP implementation and `--architecture mlp` option already exist in `model.py`. Complete and verify the shared code before Colab training; do not build another model. Use `--skip-test` during selection and `--evaluate-only` for the selected run's final test. Do not treat the DOCX's historical statements about implementation and smoke checks as fresh results.

Codex discovers repository skills under `.agents/skills`. Root `AGENTS.md` applies at the repository root; its linked files should be read only when their scope applies. See [official skill discovery documentation](https://learn.chatgpt.com/docs/build-skills) and [AGENTS.md configuration](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
