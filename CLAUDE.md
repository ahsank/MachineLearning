# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal MkDocs Material site of notes/summaries written while studying machine learning papers, published at `https://ahsank.github.io/MachineLearning/`. It also contains an older, unrelated `udacity/` coursework folder that predates the paper-notes purpose — leave it as-is (don't fix, run, or modernize it) unless explicitly asked.

## Site structure and gotchas

- `mkdocs.yml` sets `docs_dir: notes` — this is **not** the MkDocs default `docs/`. New content goes under `notes/`.
- Two kinds of notes live side by side: `notes/papers/` (one paper per note) and `notes/concepts/` (general ML/math concepts spanning multiple papers, e.g. SVD, backprop).
- Adding a new note requires updating **two** places or it won't appear on the site: the `nav` section in `mkdocs.yml` and the relevant `index.md` (`notes/papers/index.md` or `notes/concepts/index.md`).
- New paper notes should mirror the structure of `notes/papers/example-attention-is-all-you-need.md` (TL;DR, Problem, Key idea, Architecture, etc., with `!!! info` admonitions and `$...$`/`$$...$$` math via MathJax). Concept notes follow a looser structure (see `notes/concepts/singular-value-decomposition.md`) but use the same admonition/math conventions.

## Local preview and deploy

- Local preview uses a conda env, not pip/venv: `conda activate mkdocs` (create once with `conda create -n mkdocs -c conda-forge mkdocs-material`), then `mkdocs serve` and open `http://127.0.0.1:8000`.
- Deployment is CI-only via `.github/workflows/deploy.yml`, which runs `mkdocs gh-deploy --force` on every push to `master`. There's no separate manual deploy step.

## Linting

- Markdown is linted with markdownlint-cli2 (Node/npm, separate from the conda env): `npm install` once, then `npm run lint:md`. Config is `.markdownlint.yaml`.

## Workflow

- This is a solo repo — commit directly to `master`, no feature-branch/PR workflow.
