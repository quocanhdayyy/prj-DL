# Repository rules

## Scope and data

- Keep changes within the requested task. Reuse the existing preprocessing, model, prediction, and test entrypoints before adding another path.
- Treat tracked CSV/NPZ inputs as source data. Do not overwrite, delete, move, or expose them. Generated splits, metadata, and run artifacts belong under the existing ignored output paths described in the README.
- Keep ordered drug pairs ordered. Preserve the existing single-label and label mapping contracts unless the user requests a research-scope change.
- Do not interpret the 86 class IDs as clinical advice; their meanings and source dataset remain unverified in project docs.

## Plan, installation, and runtime

- A plan describes proposed work; it is not evidence that work ran. `reports/MLP_IMPLEMENTATION_PLAN.md` now separates code/docs completion from later Colab runtime. Reuse the existing MLP; do not install heavy dependencies or train locally. Verify runtime and data on Colab before training; do not execute the future Colab steps during code preparation.
- Installation checks such as `pip check` and Python source compilation do not establish runtime behavior. Model tests, smoke runs, training, and inference are separate runtime work and require that scope to be requested.
- `VALIDATION.md` records dated evidence. Its macOS results from 2026-10-19 are historical, not fresh checks of the current checkout or environment; the local checks on 2026-10-21 cover syntax and mocked logic only.
- Match verification to the request. For documentation-only changes, check links, scope, and `git diff --check`; do not run model tests. For code changes, run only appropriate checks authorized by the task.

## Progress and continuity

- When a feature or behavior change is complete, update the relevant README, plan, or report when its usage, scope, or documented decisions change, and add a concise entry to `progress.md` so reports and later sessions have continuity. Do not copy a full plan/report into the progress log or repeat every document for a typo-only change.
- Record the concrete change, checks actually run and their evidence, and remaining work. Never present planned or historical checks as fresh results.
- When a material status change affects next steps, blockers, or available artifacts, sync `docs/HANDOFF.md`; keep it a short current snapshot, not a second progress log.

## Git

- Work on the task's existing branch; do not create a branch or merge main unless requested.
- Stage explicit paths with `git add -- <path>` and inspect the staged diff before committing. Keep commits small and coherent; preserve unrelated and user-owned changes.
- Do not amend, rebase, reset, clean, force-push, or use broad staging unless the user explicitly requests it. Never discard user files.
- A request to commit authorizes a local commit only. Push only when the user explicitly asks to push; a commit does not grant push permission.
