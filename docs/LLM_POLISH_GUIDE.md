# Repo Audit: Response to the "Deadline Polish" Checklist

This document is a repo-specific reply to the "If you're at deadline..."
checklist (pasted in chat). It's written as instructions for an LLM (or human)
editing project-facing docs: what's already present, what's missing, and what
to change with minimal scope.

## Colab: keep the runtime stable (no `torch`/`numpy` installs)

The notebook already uses the "no-install" approach: clone the repo, then add
`src/` to `sys.path` (`notebooks/colab_demo.ipynb`, `colab.md`).

- Current pip usage: `!pip -q install requests tqdm` (lightweight).
- Do **not** add `poetry install`, `pip install -r ...`, or any `torch`/`numpy`
  installs/upgrades in Colab; that can break the preloaded runtime.

Local development can still use Poetry (`AGENTS.md`, `.github/workflows/tests.yml`).

## 1) "Single landing page that routes reviewers in 10 seconds"

Status: mostly done.

Already present:
- Top-level landing page: `README.md` (pitch, Colab badge, quick links).
- Reviewer routing docs: `docs/START_HERE.md`, `docs/SHOWCASE.md`,
  `docs/PROJECT_ONEPAGER.md`.

High-leverage gaps (optional, small):
- `README.md` does not currently surface a tiny results table; the best tables
  live in `docs/sendou_tier1_eval_report.md`.
- `README.md` has an architecture figure (`neural_network_themed.gv.png`) but
  not a "wow" mech-interp artifact (e.g., screenshot/snippet of top SAE
  features + influenced tokens).

## 2) "Two resume bullets that read like impact"

Status: missing as a canonical artifact in-repo.

What you can cite today:
- Model + constrained decoding: `docs/PROJECT_ONEPAGER.md`,
  `src/splatnlp/utils/reconstruct/`, `src/splatnlp/model/models.py`.
- SAE + hooks + workflow: `docs/mechinterp_workflow.md`,
  `src/splatnlp/monosemantic_sae/`.
- Metrics/evidence: `docs/sendou_tier1_eval_report.md`.

If you add exactly one file, add a short `docs/RESUME_BULLETS.md` that contains
2-3 bullets + a single evidence line per bullet (link to Colab + eval report).

## 3) "A 'proof it runs' path that is brutally reliable"

Status: good.

Already present:
- Notebook has a top markdown cell explaining "Run all" and what it shows
  (`notebooks/colab_demo.ipynb` cell 0).
- Artifact download is explicit and fails fast on missing files
  (`notebooks/colab_demo.ipynb` cell 5).

Potential micro-upgrade:
- Add one sentence to the notebook intro clarifying "expected output" (e.g.,
  "you'll see baseline vs completion builds + top SAE features").

## 4) "Crisp explanation of context violations"

Status: measured well; explanation can be surfaced more prominently.

Already present:
- The eval report explicitly tracks edit behavior:
  `top1_observed_slot_recall` / `top1_edit_chance`
  (`docs/sendou_tier1_eval_report.md`, `src/splatnlp/eval/sendou_stats.py`).
- A context-preserving baseline exists (`conditional`) and is compared against
  the models (`docs/sendou_tier1_eval_report.md`).
- Multi-reference Tier-1 scoring is implemented to reduce "wrong-but-good"
  penalties (`docs/sendou_tier1_eval_report.md`).

High-leverage doc change:
- Add a 2-3 sentence note in `README.md` (or `docs/PROJECT_ONEPAGER.md`) that
  frames "context edits" as a deliberate "coach mode" design choice, and points
  at the edit metrics.

## 5) "One wow interpretability artifact"

Status: exists in the demo; not emphasized in the landing page.

Already present:
- Colab demo includes an Ultra + SAE readout section (`notebooks/colab_demo.ipynb`).
- Deep workflow doc exists (`docs/mechinterp_workflow.md`).

High-leverage gap:
- Add one small artifact (screenshot/snippet) to `README.md` or `docs/SHOWCASE.md`
  so a reviewer gets the "wow" without opening the full report/notebook.

## Fast polish checks (repo hygiene)

Already present:
- `LICENSE`
- CI: `.github/workflows/tests.yml` (Python 3.11 + `poetry run pytest -q`)

Missing/optional:
- `CITATION.cff` (nice credibility signal)
- README badge(s) for CI (optional)
