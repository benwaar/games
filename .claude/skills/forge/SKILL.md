---
name: forge
description: "Multi-agent build orchestration — plan in NEXT.md, delegate to sonnet agents, tick checkboxes, archive to LOG.md when done."
---

# /forge — Multi-agent Build Orchestration

You are now in **forge mode**. Work is planned in `NEXT.md`, executed by sonnet subagents, and archived to `LOG.md` when complete.

## The loop

```
Plan → Delegate → Verify → Archive
```

### 1. Plan

- Check for `NEXT.md` at repo root. If it exists with unchecked boxes, pick up where it left off.
- If no plan exists (or current work is done), write one:
  - Short title + one-line goal
  - Phases with `[ ]` checkboxes — small enough that one agent can do one item
  - Each item must say what file(s) to touch and what the change is
- Commit the plan before building.

### 2. Delegate

Spawn **sonnet** agents (Agent tool, `model: "sonnet"`). Rules:

- **Directive prompts only.** "Read `X`. Edit `Y` to add `Z`. Run tests. Tick `[x]` on item N in `NEXT.md`."
- **One agent per independent item.** Parallel where items don't conflict. Sequential when they share files.
- **Every agent prompt must include:**
  - Path to `NEXT.md` and which checkbox(es) are theirs
  - Instruction to tick `[x]` when done
  - Instruction to run tests after editing
- **Never plan-mode.** Agents implement. They don't plan, propose, or ask.

### 3. Verify

After agents complete:

- `python -m pytest` (or project-specific test command) — all green
- Read `NEXT.md` — all boxes for this phase ticked
- Commit if agents didn't (one commit per logical unit)

### 3b. Review (multi-agent)

Run when a phase produces **>100 lines of changed code**. Skip for docs-only or config-only phases.

Spawn **3 parallel sonnet agents**, each given the diff (`git diff main...HEAD` or the relevant range):

1. **Code quality** — bugs, type errors, dead code, naming, missing edge cases
2. **Code smells** — duplication, god functions, premature abstraction, anti-patterns
3. **Silent failures** — swallowed errors, empty catches, fallbacks that hide bugs, missing validation at boundaries

Each agent returns a table: `| Severity | File:Line | Finding |` (Critical / Improvement / Nit).

After all three return:
- Merge findings, deduplicate
- Fix criticals and agreed improvements
- Re-run tests
- Commit fixes as `fix:` (separate from feature commits)

### 4. Archive

When a phase (or all phases) in `NEXT.md` are complete:

1. **Move completed items to `LOG.md`** — date-stamped entry with what was built, not the raw checkboxes
2. **Clear completed sections from `NEXT.md`** — leave only future work
3. **Update project docs** — README, CLAUDE.md, explainers as needed
4. Commit: `docs: archive phase N to LOG.md`

This means `NEXT.md` is always the current work. `LOG.md` is the history. A new `/forge` session starts clean.

## Input

- A goal → write the plan in `NEXT.md`, start delegating
- `continue` → read `NEXT.md`, pick up at first unchecked item
- `review` → summarise status (done / in-progress / remaining)
- `archive` → run step 4 only (move done items to LOG.md)

If no argument given, check `NEXT.md` and offer to continue or archive.

## Key differences from /research

| | /research | /forge |
|---|---|---|
| Executor | You (single agent) | Sonnet subagents |
| Plan file | `plan.md` | `NEXT.md` |
| History | Consolidated into plan | Archived to `LOG.md` |
| Docs | Every step | Phase-end |
| Best for | Learning, POCs | Building features, multi-file work |
