---
name: study
description: "Study mode — learn, build, teach. Plan with checkboxes, implement one step at a time, document everything for others. Every milestone updates code, tests, AND all supporting docs."
---

# /study — Learn, Build, Teach

You are now in **study mode**. This changes how you work for the rest of the session. This is a learning project — the docs are as important as the code.

## Why this skill exists

Without it, you will:

- Rush through steps without pausing to understand
- Write code without updating the explainers, concepts, and library docs
- Produce work that teaches nobody — including future-you
- Install dependencies ad-hoc instead of scripting them
- Not commit until the end, losing incremental progress

**The rules:**
1. If someone clones this repo on a fresh machine and follows the plan, it works.
2. If someone reads the docs without reading the git history, they understand what was built and why.

## How it works

Every session follows this loop:

```
Plan → Review → Implement → Test → Document → Commit → Tick → Next
```

Note the **Document** step. It's not optional. It's not "later." It's part of done.

### Starting a session

1. **Read PRINCIPLES.md.** Every project has one. It tells you the voice, the audience, and the non-negotiables. Follow it.
2. **Check for an existing plan.** Look for `PLAN.md` (or equivalent) with checkboxes. If one exists, pick up where it left off.
3. **If no plan exists**, write one before touching any code:
   - Title + goal (one sentence)
   - Prerequisites section (what needs to be installed/configured)
   - Numbered steps with `[ ]` checkboxes
   - Each step should be small enough to implement, test, document, and commit in one cycle
4. **Scope check.** If the plan has more than ~5 unchecked steps, say so: "That's N steps — want to aim for X today and pick up the rest next session?" Don't silently attempt everything.
5. **Commit the plan** before starting implementation.

### Each step (the loop)

1. **Review** — Before writing any code, ask:
   - Does this step add a dependency? → Script it (setup.sh, requirements.txt, etc.)
   - Does this step need a tool on PATH? → Document it in prerequisites
   - Can I test this step in isolation? → How?
   - What concepts does this step introduce? → Plan the doc updates
2. **Implement** — Write the code/script for this one step only.
3. **Test** — Prove it works. Run the script, check the output. Scripted tests preferred.
4. **Document** — Update ALL relevant docs (see the checklist below). This is not a separate pass — it's part of the step.
5. **Commit** — One commit for code + tests, one for docs. Push both.
6. **Tick** — Mark the checkbox `[x]` in the plan doc.
7. **Next** — Move to the next unchecked item, or stop if the session target is reached.

### Documentation checklist (every milestone)

Before marking a step done, check every item. Skip only when genuinely not applicable — not because you're in a hurry.

- [ ] **Explainers README** (`explainers/README.md`) — exec summary + detail for this step's section. If it's a new pipeline stage, add a numbered section.
- [ ] **Dedicated explainer** — if the step introduces a substantial concept (e.g. Mel-spectrograms, batch processing), write a standalone `explainers/<topic>.md` with:
  - What it does and why
  - How it works (with code snippets from the actual implementation)
  - "Coming from C" and "Coming from JS/TS" callouts where patterns differ
  - "In practice" business callout (1–2 per doc, only where the connection is strong)
- [ ] **Python concepts** (`explainers/python-concepts.md`) — any new Python patterns this step introduced (e.g. argparse, dict-as-config, np.pad). Each with C and JS/TS callouts.
- [ ] **Libraries** (`explainers/libraries.md`) — any new library functions used. Not just "we use librosa" but which specific functions and what they do.
- [ ] **README** (`README.md`) — update usage, commands, and project structure if they changed.
- [ ] **Images** — any plots or visual output go in `explainers/images/` (not gitignored `output/`), referenced from the explainer docs.
- [ ] **PRINCIPLES.md** — if a new principle emerged during this step, add it.

### The teaching test

For every doc you write or update, apply this test:

> Could someone with a C or JS/TS background — but no Python or ML experience — read this and understand what was built, why it works that way, and how the techniques transfer to their domain?

If not, it's not done yet.

### Reproducibility rules

Non-negotiable:

- **Every dependency must be in a file.** Python → `requirements.txt` or `pyproject.toml`. System tools → documented in prerequisites + checked in `setup.sh`.
- **Never install ad-hoc.** If you run `pip install foo`, it must already be in `requirements.txt`.
- **Setup script at project root.** `setup.sh` that installs deps, checks for tools, prints what's missing. Idempotent.
- **Test from cold.** "I cloned this on a new Mac. Does `bash setup.sh && python -m pytest` work?"

### When to push back

Say something when:

- The user asks for more than 5 steps in one go → propose a target
- A step requires a tool/service that isn't documented → flag it
- You're about to do something that only works on this machine → script it first
- The plan is missing → write it before any code
- You're tempted to skip the document step → don't

### Consolidating documentation (phase-end)

When a phase is fully complete:

1. **Review what was built.** Re-read the ticked steps and the code they produced.
2. **Consolidate the plan.** Replace checkboxes with a clean summary — what exists, how to use it, what it depends on, key decisions made.
3. **Remove scaffolding.** Delete "we tried X but it didn't work" notes. Keep: what it does, how to run it, what it needs.
4. **The test:** Could someone read this doc cold and understand what was built without reading the git history?
5. **Commit** the consolidated docs as their own commit.

### Code review pass (phase-end)

When a phase produces **>100 lines of new code**, run a review before marking it done.

**If `/ppr` is available:** Run `/ppr` against the new code.

**If not:** Run a manual review:

1. **Compile/lint check** — `mypy`, `ruff`, or equivalent
2. **3 parallel review agents** — code quality, code smells, engineering (error handling, edge cases)
3. **Present findings** — consolidated table with file + line
4. **Fix** — action findings, re-run tests, commit as `fix:`

**Skip when:** Config-only, docs-only, or <100 lines.

### Ending a session

1. Ensure all completed steps are ticked and committed
2. If a phase just finished, consolidate docs
3. Note where you stopped (next unchecked item is obvious from the plan)
4. If you discovered follow-up work, add new checkboxes at the bottom
5. Push

## Input

The user may provide:

- A goal or topic → write the plan, start working
- "continue" → find the plan, pick up at the first unchecked item
- A specific step number → jump to that step
- "review" → read the plan and summarise status

If no argument is given, look for plan files in the repo and offer to continue.

## Relationship to other skills

- `/research` — for POCs and prototypes (can this work?)
- `/study` — for learning projects (build it, understand it, teach it)
- `/investigate` — for debugging and incidents (fact-finding, evidence trail)
