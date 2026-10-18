# Git Hooks Guide

Automated code quality and commit message validation for utala: kaos 9.

## Overview

This project uses Git hooks to enforce code quality standards and conventional commit messages. All hooks run automatically when you commit code.

## Installed Hooks

### 1. Pre-Commit Hook

Runs **before** the commit is created to validate code quality.

**Checks:**
- ✅ **Ruff Linting**: Enforces code style (PEP 8, import sorting, etc.)
- ✅ **Mypy Type Checking**: Validates static type annotations
- ✅ **Unit Tests**: Ensures all 79 tests pass

**When it runs:** Before every `git commit`

**What happens:**
- All three checks must pass for the commit to succeed
- If any check fails, the commit is blocked
- You'll see detailed error output to help you fix issues

**Example output (success):**
```
======================================================================
Running pre-commit checks...
======================================================================

🔍 Ruff linting...
✅ Ruff linting passed

🔍 Mypy type checking...
✅ Mypy type checking passed

🔍 Unit tests...
✅ Unit tests passed

======================================================================
✅ All pre-commit checks passed!
======================================================================
```

**Example output (failure):**
```
======================================================================
Running pre-commit checks...
======================================================================

🔍 Ruff linting...
❌ Ruff linting failed

src/utala/engine.py:42:1: F401 'random' imported but unused
Found 1 error.

...

❌ Some pre-commit checks failed
======================================================================

To fix:
  • Ruff:  ruff check src --fix
  • Mypy:  Add type hints or assertions
  • Tests: Fix failing tests

To skip these checks (NOT RECOMMENDED):
  git commit --no-verify
```

### 2. Commit Message Hook

Validates commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) format.

**Format:**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Valid types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system changes
- `ci`: CI configuration changes
- `chore`: Other changes
- `revert`: Revert a previous commit

**Examples:**
```bash
# Good
git commit -m "feat: add Monte Carlo agent"
git commit -m "fix(engine): correct action masking logic"
git commit -m "docs: update README with setup instructions"

# Bad (will be rejected)
git commit -m "Added stuff"
git commit -m "WIP"
git commit -m "fixed it"
```

### 3. Prepare Commit Message Hook

Provides a commit message template to help you write properly formatted messages.

## Installation

Run the installation script:

```bash
./install-hooks.sh
```

This copies hooks from `hooks/` to `.git/hooks/` and makes them executable.

**Note:** The hooks are installed locally in your `.git/hooks/` directory and are **not** tracked by git. Each developer must run `./install-hooks.sh` after cloning the repository.

## Common Workflows

### Making a Commit

Normal workflow - hooks run automatically:

```bash
git add <files>
git commit -m "feat: add new feature"
```

The pre-commit hook runs first, then the commit message is validated.

### Skipping Pre-Commit Checks

**⚠️ NOT RECOMMENDED** - Only use when absolutely necessary:

```bash
git commit --no-verify -m "feat: emergency hotfix"
```

This skips the pre-commit hook but still validates the commit message.

### Fixing Quality Issues

If pre-commit checks fail:

**Ruff (linting):**
```bash
# Auto-fix issues
ruff check src --fix

# Check remaining issues
ruff check src
```

**Mypy (type checking):**
```bash
# Run type checker
mypy src

# Add type hints or assertions as needed
```

**Tests:**
```bash
# Run tests to see failures
python run_tests.py

# Or with verbose output
python -m unittest discover tests -v
```

## Troubleshooting

### Hook Not Running

Make sure hooks are installed:
```bash
./install-hooks.sh
```

Check hook is executable:
```bash
ls -la .git/hooks/pre-commit
# Should show: -rwxr-xr-x
```

### Virtual Environment Issues

The pre-commit hook automatically activates `venv/bin/activate` if it exists. If you have a different venv setup, you may need to modify the hook.

### Slow Pre-Commit Hook

The hook runs all tests (~4 seconds) on every commit. If this is too slow:

1. Temporarily skip with `--no-verify` (not recommended)
2. Make smaller, more frequent commits
3. Run quality checks manually before committing:
   ```bash
   ./quality.sh  # Run quality checks
   ./test.sh -q  # Run tests quietly
   ```

### False Positives

If a check fails incorrectly:

1. Verify the issue manually
2. Fix if it's a real problem
3. If it's a false positive, update tool configuration in `pyproject.toml`

## Customization

### Modifying Checks

Edit `hooks/pre-commit` to add/remove checks:

```bash
# Example: Add bandit security check
run_check "Security scan" bandit -r src || true
```

After editing, reinstall:
```bash
./install-hooks.sh
```

### Disabling Specific Checks

Comment out checks in `hooks/pre-commit`:

```bash
# run_check "Ruff linting" ruff check src || true  # Disabled
run_check "Mypy type checking" mypy src || true
run_check "Unit tests" python -m unittest discover tests || true
```

### Adjusting Commit Message Rules

Edit `hooks/commit-msg` to change validation rules, valid types, or message length limits.

## Best Practices

1. **Run checks before committing**: Use `./quality.sh` and `./test.sh` to catch issues early
2. **Keep commits small**: Smaller commits = faster hook execution
3. **Fix issues immediately**: Don't accumulate technical debt
4. **Use conventional commits**: Makes history readable and enables automated changelogs
5. **Don't skip hooks**: They exist to maintain code quality

## CI/CD Integration

The same quality checks run in CI/CD should also run in pre-commit hooks. This gives developers immediate feedback and prevents failed CI builds.

**Current checks:**
- ✅ Ruff linting
- ✅ Mypy type checking
- ✅ Unit tests (79 tests)

**Planned (Phase 2+):**
- Code coverage thresholds
- Integration tests
- Performance benchmarks

## References

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Git Hooks Documentation](https://git-scm.com/docs/githooks)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)

---

**Last Updated:** 2026-02-12
**Phase:** 1 (Baselines and Instrumentation)
