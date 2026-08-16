# Writing a New Note

See [`papers/example-attention-is-all-you-need.md`](papers/example-attention-is-all-you-need.md)
for the format a new paper note follows.

1. Copy the example page under `notes/papers/` (or an existing page under
   `notes/concepts/`) and rename it to match the topic (e.g.
   `notes/papers/resnet.md` or `notes/concepts/backpropagation.md`).
2. Add it to the `nav` section of `mkdocs.yml`.
3. Preview locally:

       conda activate mkdocs
       mkdocs serve

   then open <http://127.0.0.1:8000>.
4. Push to `master` — GitHub Actions builds and publishes the site
   automatically.
