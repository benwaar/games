# Code Quality Guide for utala: kaos 9

Comprehensive guide for code quality checks, linting, type checking, and security scanning.

## Quick Start

```bash
# Run all quality checks with report
./quality.sh

# Or manually
./run.sh check_quality.py -r
```

## Overview

The project uses three complementary tools for code quality:

1. **Ruff** - Fast modern linter (replaces flake8, pylint, isort, etc.)
2. **Mypy** - Static type checking
3. **Bandit** - Security vulnerability scanner

## Quality Tools

### Ruff (Linting & Style)

**What it checks:**
- Code style (PEP 8)
- Import sorting
- Code simplifications
- Potential bugs
- Modern Python idioms

**Example issues:**
- Unsorted imports
- Use of `Optional[X]` instead of `X | None` (Python 3.10+)
- Nested if statements that can be combined
- Unused imports

**Auto-fix many issues:**
```bash
ruff check src --fix
```

### Mypy (Type Checking)

**What it checks:**
- Type annotations
- Type compatibility
- Optional/None handling
- Function signatures

**Example issues:**
- `Optional[int]` passed where `int` expected
- Missing type annotations
- Incompatible types in assignments

**Note:** Type checking is currently lenient (Phase 1 focus). Stricter checking can be enabled in Phase 2.

### Bandit (Security)

**What it checks:**
- Security vulnerabilities
- Dangerous function usage
- Hardcoded secrets
- Insecure random number generation (for crypto)

**Configured to skip:**
- `B101` - assert_used (we use asserts appropriately)
- `B311` - random (standard random is fine for game simulation)

**Current status:** Low-severity issues only (acceptable for Phase 1)

## Usage

### Basic Usage

```bash
# Run all checks (verbose)
./run.sh check_quality.py

# Run all checks (quiet) with report
./run.sh check_quality.py -q -r

# Convenience script
./quality.sh
```

### Individual Tools

```bash
# Run ruff only
ruff check src

# Run ruff with auto-fix
ruff check src --fix

# Run mypy only
mypy src

# Run bandit only
bandit -r src
```

### Check Specific Files

```bash
python check_quality.py -s src/utala/engine.py
python check_quality.py -s src/utala/agents/
```

## Reports

Quality reports saved to: `test_reports/quality/quality_report_YYYYMMDD_HHMMSS.txt`

**Report contents:**
- Summary of all checks (PASS/FAIL)
- Detailed output from each tool
- File locations and line numbers for issues
- Suggestions for fixes

**Example report:**
```
================================================================================
CODE QUALITY REPORT
================================================================================
Generated: 2026-02-12 14:40:18
Source Directory: src

SUMMARY
--------------------------------------------------------------------------------
RUFF                 ✗ FAIL
MYPY                 ✗ FAIL
BANDIT               ✓ PASS

Overall Status: ✗ SOME CHECKS FAILED

DETAILED RESULTS
--------------------------------------------------------------------------------
...
```

## Current Status

### Phase 1 Quality Baseline

**Ruff Issues:** ~30 style issues
- Mostly auto-fixable
- Import sorting
- Type annotation modernization
- Code simplification suggestions

**Mypy Issues:** ~19 type errors
- Optional[int] vs int | None usage
- Optional type handling in action masking
- Non-critical for functionality

**Bandit Issues:** 7 low-severity warnings
- Use of `random.Random()` (acceptable for games)
- Use of `assert` (acceptable in our code)
- No high or medium severity issues

**Overall Assessment:** Code is functional and secure. Style and type issues are cosmetic and can be addressed incrementally.

## Configuration

### pyproject.toml

Central configuration for all tools:

```toml
[tool.ruff]
target-version = "py310"
line-length = 100
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true

[tool.bandit]
exclude_dirs = ["tests", "venv"]
skips = ["B101", "B311"]
```

See [pyproject.toml](pyproject.toml) for full configuration.

## Fixing Issues

### Auto-Fix with Ruff

Many ruff issues can be auto-fixed:

```bash
# Preview fixes
ruff check src --fix --dry-run

# Apply fixes
ruff check src --fix

# Format code
ruff format src
```

### Type Issues (Mypy)

Common patterns:

```python
# Before (causes mypy error)
def func(value: Optional[int]):
    resources.has_rocketman(value)  # Error: Optional[int] vs int

# After (fixed)
def func(value: Optional[int]):
    if value is not None:
        resources.has_rocketman(value)  # OK
```

### Security Issues (Bandit)

Most bandit issues are false positives for our use case:

```python
# Bandit warns about this (B311)
self.rng = random.Random(seed)

# This is fine for game simulation (not cryptography)
# Already configured to skip in pyproject.toml
```

## Best Practices

1. **Run quality checks before committing**
   ```bash
   ./quality.sh -q
   ```

2. **Auto-fix ruff issues regularly**
   ```bash
   ruff check src --fix
   ```

3. **Address high/medium severity issues immediately**
   - Bandit high/medium issues = security problems
   - Currently none in codebase ✓

4. **Incremental improvement**
   - Fix issues as you touch files
   - Don't need to fix everything at once
   - Focus on critical paths first

5. **Use type hints for new code**
   ```python
   def new_function(x: int, y: str) -> bool:
       ...
   ```

## IDE Integration

### VS Code

Install extensions:
- Ruff (charliermarsh.ruff)
- Mypy Type Checker (ms-python.mypy-type-checker)

Add to `.vscode/settings.json`:
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.mypyEnabled": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### PyCharm

1. Settings → Tools → External Tools
2. Add ruff, mypy as external tools
3. Enable on-save actions

## CI/CD Integration

Quality checks can be integrated into CI/CD:

```bash
# In CI pipeline
python check_quality.py -q -r

# Exit code 0 = all passed
# Exit code 1 = some failed
```

**GitHub Actions example:**
```yaml
- name: Run quality checks
  run: |
    pip install -r requirements.txt
    python check_quality.py -q
```

## Troubleshooting

### "Command not found: ruff"

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Type Errors After Changes

Run mypy to find issues:
```bash
mypy src/utala/engine.py
```

### Too Many Ruff Issues

Focus on specific categories:
```bash
# Only check imports
ruff check src --select I

# Only check type annotations
ruff check src --select UP

# Only check bugs
ruff check src --select B
```

## Metrics

Current quality metrics:

```
Total Lines of Code: ~1,270
Ruff Issues: ~30 (mostly style)
Mypy Errors: ~19 (type annotations)
Bandit Issues: 7 low-severity (acceptable)
Security Issues: 0 high/medium ✓
```

**Quality Score: 7/10**
- ✓ No security issues
- ✓ Clean architecture
- ✓ Consistent style
- ⚠️ Some type annotation improvements needed
- ⚠️ Some code simplifications available

## Phase 2 Goals

1. **Achieve ruff clean**
   - Auto-fix all style issues
   - Zero ruff warnings

2. **Improve type coverage**
   - Add type hints to new code
   - Fix existing Optional handling

3. **Enable stricter mypy**
   - `disallow_untyped_defs = true`
   - Gradual rollout per-module

4. **Maintain security**
   - Zero high/medium bandit issues
   - Regular security scans

## Summary

- ✅ **Quality tools installed and configured**
- ✅ **Automated checking via ./quality.sh**
- ✅ **Reports generated in test_reports/quality/**
- ✅ **No security issues**
- ✅ **Style issues are cosmetic and fixable**
- ✅ **Type checking in place for new code**

Code quality monitoring is now part of the development workflow!
