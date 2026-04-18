# Testing Guide for utala: kaos 9

Complete guide for running tests, generating reports, and measuring code coverage.

## Quick Start

```bash
# Run all tests with reports and coverage (recommended)
./test.sh

# Or manually
./run.sh run_tests.py --both --coverage
```

## Test Runner Options

### Basic Usage

```bash
# Verbose output (default)
./run.sh run_tests.py

# Quiet mode (minimal output)
./run.sh run_tests.py -q

# Silent mode (no output)
./run.sh run_tests.py -q > /dev/null
```

### Report Generation

```bash
# Generate text report only
./run.sh run_tests.py -r
./run.sh run_tests.py --report

# Generate HTML report only
./run.sh run_tests.py --html

# Generate both text and HTML reports
./run.sh run_tests.py --both
```

### Code Coverage

```bash
# Run tests with coverage tracking
./run.sh run_tests.py --coverage

# Combined: tests + reports + coverage
./run.sh run_tests.py --both --coverage

# Quiet mode with coverage
./run.sh run_tests.py -q --coverage
```

### Convenience Script

```bash
# Activates venv and runs tests with reports and coverage
./test.sh

# Pass additional arguments
./test.sh -q     # Quiet mode with reports and coverage
```

## Report Formats

### Text Report

Saved to: `test_reports/test_report_YYYYMMDD_HHMMSS.txt`

Includes:
- Overall summary (tests run, passed, failed, errors)
- Test breakdown by module
- Module-level timing
- Slowest tests (top 10)
- Full failure/error tracebacks

Example:
```
================================================================================
utala: kaos 9 - Test Report
================================================================================
Date: 2026-02-12 14:28:06
Total Time: 3.82 seconds

OVERALL SUMMARY
--------------------------------------------------------------------------------
Total Tests Run:  79
Passed:           79 (100.0%)
Failed:           0
Errors:           0
Skipped:          0

Status: ✓ ALL TESTS PASSED

TEST BREAKDOWN BY MODULE
--------------------------------------------------------------------------------
✓ test_actions                    18 passed    0 failed    0 errors    0.00s
✓ test_agents                     13 passed    0 failed    0 errors    3.79s
✓ test_engine                     18 passed    0 failed    0 errors    0.01s
✓ test_replay                     11 passed    0 failed    0 errors    0.01s
✓ test_state                      19 passed    0 failed    0 errors    0.00s

SLOWEST TESTS (Top 10)
--------------------------------------------------------------------------------
 3.503s  test_plays_full_game (test_agents.TestMonteCarloAgent.test_plays_full_game)
 0.279s  test_select_action (test_agents.TestMonteCarloAgent.test_select_action)
 ...
```

### HTML Report

Saved to: `test_reports/test_report_YYYYMMDD_HHMMSS.html`

Features:
- Professional styling with CSS
- Color-coded status (green=pass, red=fail, orange=error)
- Interactive hover effects on tables
- Summary boxes with large numbers
- Module breakdown table
- Slowest tests table
- Full failure/error tracebacks with formatting

Open in browser:
```bash
# macOS
open test_reports/test_report_20260212_142806.html

# Linux
xdg-open test_reports/test_report_20260212_142806.html

# Or copy the file:// URL from the test output
```

### Coverage Reports

When using `--coverage`, three types of reports are generated:

**1. Text Report** (`test_reports/coverage/coverage_YYYYMMDD_HHMMSS.txt`)
```
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/utala/__init__.py                       1      0   100%
src/utala/actions.py                       79      7    91%   38-43, 157-158
src/utala/agents/base.py                   15      2    87%   49, 72
src/utala/agents/heuristic_agent.py        92     19    79%   64, 88, 167...
src/utala/agents/monte_carlo_agent.py      71      4    94%   125-127, 189
src/utala/agents/random_agent.py           12      1    92%   47
src/utala/engine.py                       210      9    96%   103-105, 131-134...
src/utala/state.py                        138     22    84%   115-119, 223-246
src/utala/replays/format.py                54     10    81%   121-137
---------------------------------------------------------------------
TOTAL                                     754    156    79%
```

**2. HTML Report** (`test_reports/coverage/html_YYYYMMDD_HHMMSS/index.html`)
- Visual coverage report with file-by-file breakdown
- Color-coded lines (green=covered, red=not covered)
- Shows exact lines missing coverage
- Interactive navigation through source code

**3. XML Report** (`test_reports/coverage/coverage_YYYYMMDD_HHMMSS.xml`)
- Machine-readable format for CI/CD integration
- Compatible with Jenkins, GitLab CI, GitHub Actions, etc.

### Current Coverage

```
Overall: 79%
High coverage: engine.py (96%), monte_carlo_agent.py (94%), actions.py (91%)
Areas to improve: harness.py (0% - not directly tested), heuristic_agent.py (79%)
```

**Coverage by Module:**
- State representation: 84%
- Actions and masking: 91%
- Game engine: 96%
- Random agent: 92%
- Heuristic agent: 79%
- Monte Carlo agent: 94%
- Replay format: 81%

## Test Suite Overview

### Current Status

```
79 tests, all passing ✓
Total time: ~3.8 seconds
Coverage: All Phase 1 components
```

### Test Modules

| Module | Tests | Time | Coverage |
|--------|-------|------|----------|
| test_state | 19 | 0.00s | State representation, grid, resources |
| test_actions | 18 | 0.00s | Action space, masking, legality |
| test_engine | 18 | 0.01s | Game mechanics, determinism |
| test_agents | 13 | 3.79s | Agent implementations, full games |
| test_replay | 11 | 0.01s | Replay format, serialization |

### Performance Notes

- **Monte Carlo tests are slow** (~3.5s for full game)
- Most tests are very fast (<0.01s)
- Total runtime is dominated by agent tests
- Consider using fewer rollouts for faster testing

## Running Specific Tests

### Single Test File

```bash
python -m unittest tests.test_state
python -m unittest tests.test_actions
python -m unittest tests.test_engine
python -m unittest tests.test_agents
python -m unittest tests.test_replay
```

### Single Test Class

```bash
python -m unittest tests.test_state.TestPlayer
python -m unittest tests.test_actions.TestLegalActionMasking
python -m unittest tests.test_engine.TestPlacementPhase
```

### Single Test Method

```bash
python -m unittest tests.test_state.TestPlayer.test_opponent
python -m unittest tests.test_engine.TestDeterminism.test_same_seed_same_kaos
```

## Continuous Integration

### Pre-Commit Testing

Always run tests before committing:

```bash
./test.sh -q && git commit
```

Or add to git hooks:

```bash
# .git/hooks/pre-commit
#!/bin/bash
./test.sh -q
exit $?
```

### Automated Testing

For CI/CD pipelines:

```bash
# Run tests and fail on error
python run_tests.py -q || exit 1

# Generate report for CI artifacts
python run_tests.py -q --both
```

## Troubleshooting

### Tests Fail

1. Check the report for failure details
2. Run specific failing test for more info:
   ```bash
   python -m unittest tests.test_name.TestClass.test_method
   ```
3. Check git status for uncommitted changes

### Tests Slow

Monte Carlo tests are inherently slow. Options:

1. **Skip slow tests temporarily:**
   ```bash
   python -m unittest discover tests -k "not MonteCarlo"
   ```

2. **Reduce rollouts** (edit test file):
   ```python
   agent = MonteCarloAgent("MC", num_rollouts=5, seed=42)
   ```

3. **Profile tests:**
   ```bash
   python -m cProfile -s cumulative run_tests.py
   ```

### Import Errors

Make sure you're in the project root and venv is activated:

```bash
cd /path/to/game
source venv/bin/activate
python run_tests.py
```

### Report Not Generated

Reports are auto-generated on failure. For success, use:

```bash
./run.sh run_tests.py --both
```

## Best Practices

1. **Run tests with coverage before every commit**
   - Ensures no regressions
   - Validates changes work correctly
   - Tracks code coverage
   ```bash
   ./test.sh -q  # Quick check with coverage
   ```

2. **Use quiet mode for quick checks**
   ```bash
   ./test.sh -q
   ```

3. **Generate reports for documentation**
   ```bash
   ./test.sh  # Creates timestamped test and coverage reports
   ```

4. **Monitor code coverage**
   - Aim for >80% overall coverage
   - Focus on critical paths (engine, actions)
   - Review HTML coverage report for gaps

5. **Review slowest tests periodically**
   - Check report for performance regressions
   - Optimize slow tests

6. **Keep tests fast**
   - Mock expensive operations
   - Use small test data
   - Avoid unnecessary full game simulations

## Test Report Archive

Reports are saved with timestamps in `test_reports/`:

```
test_reports/
├── test_report_20260212_142806.txt
├── test_report_20260212_142806.html
├── test_report_20260212_151230.txt
├── test_report_20260212_151230.html
└── coverage/
    ├── coverage_20260212_142806.txt
    ├── coverage_20260212_142806.xml
    └── html_20260212_142806/
        ├── index.html
        ├── *.html (per-file coverage)
        └── ... (coverage assets)
```

Cleanup old reports:

```bash
# Keep only last 10 test reports
cd test_reports
ls -t test_report*.* | tail -n +21 | xargs rm

# Clean old coverage reports
cd coverage
ls -t -d html_* | tail -n +6 | xargs rm -rf
ls -t coverage_*.* | tail -n +11 | xargs rm
```

## IDE Integration

### VS Code

Add to `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "./run.sh run_tests.py",
      "group": {
        "kind": "test",
        "isDefault": true
      }
    },
    {
      "label": "Run Tests with Report",
      "type": "shell",
      "command": "./test.sh"
    }
  ]
}
```

### PyCharm

1. Right-click on `tests/` directory
2. Select "Run 'Unittests in tests'"
3. Or use `Ctrl+Shift+F10`

## Coverage Analysis

Coverage is now integrated into the test runner!

```bash
# Run tests with coverage (recommended method)
./test.sh

# Or manually
./run.sh run_tests.py --coverage

# View reports
open test_reports/coverage/html_*/index.html
```

**Manual coverage (advanced users):**

```bash
# Install coverage (already in requirements.txt)
pip install coverage

# Run tests with coverage manually
coverage run -m unittest discover tests

# View report
coverage report

# Generate HTML coverage report
coverage html
open htmlcov/index.html
```

**Current coverage statistics:**
- Overall: 79%
- Engine core: 96%
- Monte Carlo agent: 94%
- Actions: 91%
- Random agent: 92%

## Summary

- ✅ **79 tests, all passing**
- ✅ **~4 seconds runtime**
- ✅ **79% code coverage**
- ✅ **Text and HTML test reports**
- ✅ **Text, HTML, and XML coverage reports**
- ✅ **Module-level breakdown**
- ✅ **Performance tracking**
- ✅ **Integrated coverage tracking**

For detailed test documentation, see [tests/README.md](tests/README.md).
