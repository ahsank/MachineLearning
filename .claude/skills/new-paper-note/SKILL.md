---
name: new-paper-note
description: Scaffold a new ML paper study note in this repo — creates the note file from the example template and wires it into mkdocs.yml nav and notes/papers/index.md in one step. Use when the user asks to add/start a note or summary for a paper.
---

Given a paper (title, authors/year, and optionally an arXiv/URL link, provided via $ARGUMENTS or asked for if missing), do the following:

1. **Pick a slug**: kebab-case from the paper title (e.g. "Attention Is All You Need" → `attention-is-all-you-need`). Do not reuse the `example-` prefix.

2. **Create `notes/papers/<slug>.md`** by copying the structure of `notes/papers/example-attention-is-all-you-need.md`:
   - Title heading: `# <Paper Title> (<Authors>, <Year>)`
   - A `!!! info "Reference"` admonition with citation and link (if known).
   - Sections: `## TL;DR`, `## Problem`, `## Key idea`, `## Architecture` (or whatever sections fit the paper — these are a starting point, not a rigid requirement), using `$...$`/`$$...$$` for math and fenced code blocks where useful.
   - Do NOT include the template's "copy this file to start a new paper summary" instruction line — that's specific to the example file.
   - Leave sections as brief placeholders if the user hasn't given you enough detail about the paper yet; don't fabricate technical claims about a paper you don't have real information on.

3. **Add the note to `mkdocs.yml`**: insert a new entry under the `nav: > Papers:` list, alongside the existing "Example: Attention Is All You Need" entry, using the paper's title as the nav label and `papers/<slug>.md` as the path.

4. **Add the note to `notes/papers/index.md`**: add a link/entry consistent with how the existing example entry is listed there.

5. Report back which files were created/edited. Do not commit — leave that to the user.
