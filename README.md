# ML Notes

Notes and summaries written while studying machine learning papers,
published as a static site with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

Published site: <https://ahsank.github.io/MachineLearning/>

## Writing a new note

Notes live under [`notes/papers/`](notes/papers/), one Markdown file per
paper. [`notes/papers/example-attention-is-all-you-need.md`](notes/papers/example-attention-is-all-you-need.md)
is a template showing the format (summary, math via `$...$`/`$$...$$`,
admonitions, code blocks).

1. Copy the example file and rename it for your paper, e.g.
   `notes/papers/resnet.md`.
2. Add it to the `nav` section of [`mkdocs.yml`](mkdocs.yml) and to
   [`notes/papers/index.md`](notes/papers/index.md).
3. Preview locally (see below) and check formatting.
4. Commit and push to `master` — GitHub Actions builds the site and
   publishes it to GitHub Pages automatically.

## Local preview

A conda environment named `mkdocs` with `mkdocs-material` installed is set
up for this:

    conda activate mkdocs
    mkdocs serve

Then open <http://127.0.0.1:8000> and edit files under `notes/` — the
preview reloads on save.

If you don't have that environment, create it with:

    conda create -n mkdocs -c conda-forge mkdocs-material

## Older content

[`udacity/`](udacity/) holds coursework (a Jupyter notebook on the Boston
housing dataset) predating this repo's use for paper notes. It isn't part
of the published site.
