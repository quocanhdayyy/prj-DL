# Repository guidance

These instructions apply only to this repository.

- Use concise Vietnamese for collaboration and reports unless the user asks otherwise. Follow Ponytail full: inspect only the files needed, reuse existing helpers, and make the smallest complete change.
- Read [docs/RULES.md](docs/RULES.md) for repository-wide data, Git, and validation rules. Read [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) when you need to locate project sources of truth. For an MLP environment setup task, load the repo skill at [.agents/skills/ddi-setup/SKILL.md](.agents/skills/ddi-setup/SKILL.md); do not load it for unrelated work.
- The main session may delegate a bounded task when delegation is useful. Prefer coordination over parallel implementation: assign a clear outcome, then wait for the agent's final result or blocker instead of polling progress or rereading its intermediate work. Review the delivered changes in scope; make a small correction or restart/follow up only for a concrete defect or blocker. Do not duplicate completed research or add an audit agent for a simple task.
- Use `gpt-6-luna` with low reasoning effort by default for suitable tasks, as requested by the user. Use the fewest agents needed; parallelize only independent work. Move to a higher model only if the user asks or a reported blocker cannot be solved at the current level; do not infer permission to upgrade.
- The user may explicitly ask the main session to do work itself, including writing a full plan. Honor the latest explicit instruction; simple tasks need not be delegated.
- Distinguish plan-only, installation-only, and runtime tasks. The current MLP setup plan is installation-only and does not authorize model execution; follow the user's latest scope if it changes. A no-monitor preference does not waive necessary validation or review of the final result.
