# M12 Plan — Documentation & Consolidation

**Goal:** Replace all Phase 2 milestone checkboxes in PLAN.md with a clean
summary of what was built. Update all explainers so the project is fully
documented from a cold read.

## Prerequisites

- M7–M11 all complete ✅
- 108 tests passing

---

## Loop: Plan → Implement → Test → Document → Commit → Tick → Next

---

### Step 1 — PLAN.md Phase 2 consolidation

- [ ] Replace M7–M12 checkboxes with a clean "What was built" summary section
  (like Phase 1 ✅ at the top of the file)
- [ ] Include: what exists, how to run it, what it depends on, key decisions made

**Test:** could someone read just the Phase 2 summary and understand what the
model is, how to use it, and what it achieved?

**Docs:** this step IS the docs.

**Commit:** `docs(signal-hunt): consolidate Phase 2 in PLAN.md`

---

### Step 2 — explainers/README.md sweep

- [ ] Check every section header links to a real file
- [ ] Check "See:" links point to files that exist
- [ ] Add any missing sections for Phase 2 concepts not yet covered
- [ ] Remove any "Coming in MX" stubs that are now built

**Docs:** this step IS the docs.

**Commit:** `docs(signal-hunt): tidy explainers README after Phase 2`

---

### Step 3 — python-concepts.md check

- [ ] Review what new patterns Phase 2 introduced that aren't yet documented:
  - `dataclass` with `field(default_factory=...)` — in config.py
  - `module.train()` / `module.eval()` — PyTorch model modes
  - `torch.no_grad()` context manager
  - `scope="module"` in pytest fixtures

**Docs:** add any missing concepts.

**Commit:** `docs(signal-hunt): python-concepts additions from Phase 2`

---

### Step 4 — final check: teaching test

- [ ] Read explainers/README.md cold. Does it tell the full story from
  raw audio to inference without needing to read the git history?
- [ ] Fix any gaps found.

**Commit:** `docs(signal-hunt): Phase 2 teaching test fixes`

---

### Step 5 — close out M12 + Phase 2

- [ ] Mark M12 checkboxes `[x]` in `PLAN.md`
- [ ] Update `NEXT.md` to point at Phase 3 (M13 — note dataset collection)
- [ ] Update memory: Phase 2 complete, next is M13

**Commit:** `docs(signal-hunt): tick M12, Phase 2 complete`

---

## Gate (from PLAN.md)

Docs pass the teaching test. Someone with C/JS/TS background can follow
the full path from raw audio to inference without reading git history.
All Phase 2 checkboxes replaced with clean summary.
