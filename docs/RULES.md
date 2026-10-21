# Repository rules

## Scope and data

- Keep changes within the requested task. Reuse the existing preprocessing, model, prediction, and test entrypoints before adding another path.
- Treat tracked CSV/NPZ inputs as source data. Do not overwrite, delete, move, or expose them. Generated splits, metadata, and run artifacts belong under the existing ignored output paths described in the README.
- Keep ordered drug pairs ordered. Preserve the existing single-label and label mapping contracts unless the user requests a research-scope change.
- Do not interpret the 86 class IDs as clinical advice; their meanings and source dataset remain unverified in project docs.

## Plan, installation, and runtime

- A plan describes proposed work; it is not evidence that work ran. `reports/MLP_IMPLEMENTATION_PLAN.md` is currently installation-only: use the existing MLP, prepare/check the environment only when asked, and do not run preprocessing, tests, model construction, training, evaluation, or prediction under that scope.
- Installation checks such as `pip check` and Python source compilation do not establish runtime behavior. Model tests, smoke runs, training, and inference are separate runtime work and require that scope to be requested.
- `VALIDATION.md` records historical results from macOS on 2026-10-19. Cite it as recorded history, not as a fresh check on the current checkout or environment.
- Match verification to the request. For documentation-only changes, check links, scope, and `git diff --check`; do not run model tests. For code changes, run only appropriate checks authorized by the task.

## Git

- Work on the task's existing branch; do not create a branch or merge main unless requested.
- Stage explicit paths with `git add -- <path>` and inspect the staged diff before committing. Keep commits small and coherent; preserve unrelated and user-owned changes.
- Do not amend, rebase, reset, clean, force-push, or use broad staging unless the user explicitly requests it. Never discard user files.
- A request to commit authorizes a local commit only. Push only when the user explicitly asks to push; a commit does not grant push permission.
