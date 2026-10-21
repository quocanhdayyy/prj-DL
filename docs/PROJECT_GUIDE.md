# Project guide

Use these files as the sources of truth; do not copy their contents into this index.

- [README.md](../README.md): supported pipeline, commands, data contracts, and usage limits.
- [progress.md](../progress.md): implementation progress and open work.
- [VALIDATION.md](../VALIDATION.md): recorded validation results and their environment/date.
- [reports/bao_cao_y_tuong_DDI.md](../reports/bao_cao_y_tuong_DDI.md): original research proposal.
- [reports/MLP_IMPLEMENTATION_PLAN.md](../reports/MLP_IMPLEMENTATION_PLAN.md): user-approved installation-only plan; it does not mean setup or runtime work has happened.
- [Bao_cao_y_tuong_Deep_Learning_DDI_hoan_chinh.md](../Bao_cao_y_tuong_Deep_Learning_DDI_hoan_chinh.md): faithful Markdown conversion of the supplied DOCX.

The MLP implementation and `--architecture mlp` option already exist in `model.py`. The current plan says the environment still needs setup; it does not call for another model implementation. Do not treat the DOCX's historical statements about implementation and smoke checks as fresh results.

Codex discovers repository skills under `.agents/skills`. Root `AGENTS.md` applies at the repository root; its linked files should be read only when their scope applies. See [official skill discovery documentation](https://learn.chatgpt.com/docs/build-skills) and [AGENTS.md configuration](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
