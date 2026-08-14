# ML Notes

Notes and summaries written while studying machine learning papers. Each
entry captures the problem the paper tackles, the key idea, and whatever
math or code made it click for me.

## How this is organized

- **[Papers](papers/index.md)** — one page per paper, named
  `papers/<short-title>.md`.

See [`papers/example-attention-is-all-you-need.md`](papers/example-attention-is-all-you-need.md)
for the format a new note follows.

## Writing a new note

1. Copy the example page under `notes/papers/` and rename it to match the
   paper (e.g. `notes/papers/resnet.md`).
2. Add it to the `nav` section of `mkdocs.yml`.
3. Preview locally:

       conda activate mkdocs
       mkdocs serve

   then open <http://127.0.0.1:8000>.
4. Push to `master` — GitHub Actions builds and publishes the site
   automatically.
