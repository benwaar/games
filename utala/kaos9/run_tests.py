#!/usr/bin/env python3
"""
Test runner for utala: kaos 9.

Runs all unit tests and generates detailed reports.
"""

import sys
import unittest
import time
from datetime import datetime
from pathlib import Path
from io import StringIO

# Try to import coverage
try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

# Add src to path
sys.path.insert(0, 'src')


class DetailedTestResult(unittest.TestResult):
    """Enhanced test result with timing and details."""

    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super().__init__(stream, descriptions, verbosity)
        self.test_times = {}
        self.success_tests = []

    def startTest(self, test):
        super().startTest(test)
        self.test_times[test] = time.time()

    def stopTest(self, test):
        super().stopTest(test)
        elapsed = time.time() - self.test_times[test]
        self.test_times[test] = elapsed

    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_tests.append(test)


def run_all_tests(verbosity=2):
    """
    Run all unit tests.

    Args:
        verbosity: Test output verbosity (0=quiet, 1=normal, 2=verbose)

    Returns:
        DetailedTestResult object
    """
    # Discover all tests in tests/ directory
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Run tests with detailed result
    if verbosity > 0:
        runner = unittest.TextTestRunner(verbosity=verbosity, resultclass=DetailedTestResult)
        result = runner.run(suite)
    else:
        # Silent mode - capture output
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=0, resultclass=DetailedTestResult)
        result = runner.run(suite)

    return result, suite


def generate_text_report(result, suite, start_time, end_time):
    """
    Generate detailed text report.

    Returns:
        String containing the report
    """
    lines = []
    elapsed = end_time - start_time

    lines.append("=" * 80)
    lines.append("utala: kaos 9 - Test Report")
    lines.append("=" * 80)
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Time: {elapsed:.2f} seconds")
    lines.append("")

    # Overall summary
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total Tests Run:  {result.testsRun}")
    lines.append(f"Passed:           {len(result.success_tests)} ({len(result.success_tests)/result.testsRun*100:.1f}%)")
    lines.append(f"Failed:           {len(result.failures)}")
    lines.append(f"Errors:           {len(result.errors)}")
    lines.append(f"Skipped:          {len(result.skipped)}")
    lines.append("")

    # Status
    if result.wasSuccessful():
        lines.append("Status: ✓ ALL TESTS PASSED")
    else:
        lines.append("Status: ✗ SOME TESTS FAILED")
    lines.append("")

    # Test breakdown by module
    lines.append("TEST BREAKDOWN BY MODULE")
    lines.append("-" * 80)

    # Organize tests by module
    modules = {}
    for test in result.success_tests:
        module = test.__class__.__module__
        if module not in modules:
            modules[module] = {'passed': 0, 'failed': 0, 'errors': 0, 'time': 0}
        modules[module]['passed'] += 1
        modules[module]['time'] += result.test_times.get(test, 0)

    for test, _ in result.failures:
        module = test.__class__.__module__
        if module not in modules:
            modules[module] = {'passed': 0, 'failed': 0, 'errors': 0, 'time': 0}
        modules[module]['failed'] += 1
        modules[module]['time'] += result.test_times.get(test, 0)

    for test, _ in result.errors:
        module = test.__class__.__module__
        if module not in modules:
            modules[module] = {'passed': 0, 'failed': 0, 'errors': 0, 'time': 0}
        modules[module]['errors'] += 1
        modules[module]['time'] += result.test_times.get(test, 0)

    # Print module stats
    for module in sorted(modules.keys()):
        stats = modules[module]
        total = stats['passed'] + stats['failed'] + stats['errors']
        status = "✓" if stats['failed'] == 0 and stats['errors'] == 0 else "✗"
        lines.append(f"{status} {module:<30} {stats['passed']:>3} passed  {stats['failed']:>3} failed  {stats['errors']:>3} errors  {stats['time']:>6.2f}s")

    lines.append("")

    # Failures
    if result.failures:
        lines.append("FAILURES")
        lines.append("-" * 80)
        for test, traceback in result.failures:
            lines.append(f"FAIL: {test}")
            lines.append(traceback)
            lines.append("")

    # Errors
    if result.errors:
        lines.append("ERRORS")
        lines.append("-" * 80)
        for test, traceback in result.errors:
            lines.append(f"ERROR: {test}")
            lines.append(traceback)
            lines.append("")

    # Slowest tests
    if result.test_times:
        lines.append("SLOWEST TESTS (Top 10)")
        lines.append("-" * 80)
        slowest = sorted(result.test_times.items(), key=lambda x: x[1], reverse=True)[:10]
        for test, elapsed in slowest:
            lines.append(f"{elapsed:>6.3f}s  {test}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("End of Report")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_html_report(result, suite, start_time, end_time):
    """
    Generate HTML report.

    Returns:
        String containing HTML report
    """
    elapsed = end_time - start_time
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Status color
    if result.wasSuccessful():
        status_color = "#4CAF50"  # Green
        status_text = "✓ ALL TESTS PASSED"
    else:
        status_color = "#f44336"  # Red
        status_text = "✗ SOME TESTS FAILED"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>utala: kaos 9 - Test Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #2196F3;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        .header-info {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .status {{
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            margin: 20px 0;
            background-color: {status_color};
            color: white;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-box {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        .summary-box .number {{
            font-size: 36px;
            font-weight: bold;
            color: #2196F3;
        }}
        .summary-box .label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #2196F3;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .pass {{ color: #4CAF50; font-weight: bold; }}
        .fail {{ color: #f44336; font-weight: bold; }}
        .error {{ color: #ff9800; font-weight: bold; }}
        .failure-box {{
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .error-box {{
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>utala: kaos 9 - Test Report</h1>

        <div class="header-info">
            <strong>Generated:</strong> {timestamp}<br>
            <strong>Total Time:</strong> {elapsed:.2f} seconds
        </div>

        <div class="status">{status_text}</div>

        <div class="summary">
            <div class="summary-box">
                <div class="number">{result.testsRun}</div>
                <div class="label">Total Tests</div>
            </div>
            <div class="summary-box">
                <div class="number" style="color: #4CAF50;">{len(result.success_tests)}</div>
                <div class="label">Passed</div>
            </div>
            <div class="summary-box">
                <div class="number" style="color: #f44336;">{len(result.failures)}</div>
                <div class="label">Failed</div>
            </div>
            <div class="summary-box">
                <div class="number" style="color: #ff9800;">{len(result.errors)}</div>
                <div class="label">Errors</div>
            </div>
        </div>

        <h2>Test Breakdown by Module</h2>
        <table>
            <tr>
                <th>Module</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Errors</th>
                <th>Time</th>
            </tr>
"""

    # Organize tests by module
    modules = {}
    for test in result.success_tests:
        module = test.__class__.__module__
        if module not in modules:
            modules[module] = {'passed': 0, 'failed': 0, 'errors': 0, 'time': 0}
        modules[module]['passed'] += 1
        modules[module]['time'] += result.test_times.get(test, 0)

    for test, _ in result.failures:
        module = test.__class__.__module__
        if module not in modules:
            modules[module] = {'passed': 0, 'failed': 0, 'errors': 0, 'time': 0}
        modules[module]['failed'] += 1
        modules[module]['time'] += result.test_times.get(test, 0)

    for test, _ in result.errors:
        module = test.__class__.__module__
        if module not in modules:
            modules[module] = {'passed': 0, 'failed': 0, 'errors': 0, 'time': 0}
        modules[module]['errors'] += 1
        modules[module]['time'] += result.test_times.get(test, 0)

    for module in sorted(modules.keys()):
        stats = modules[module]
        html += f"""            <tr>
                <td>{module}</td>
                <td class="pass">{stats['passed']}</td>
                <td class="fail">{stats['failed']}</td>
                <td class="error">{stats['errors']}</td>
                <td>{stats['time']:.2f}s</td>
            </tr>
"""

    html += """        </table>
"""

    # Failures
    if result.failures:
        html += """        <h2>Failures</h2>
"""
        for test, traceback in result.failures:
            html += f"""        <div class="failure-box">
            <strong>FAIL:</strong> {test}<br>
            <pre>{traceback}</pre>
        </div>
"""

    # Errors
    if result.errors:
        html += """        <h2>Errors</h2>
"""
        for test, traceback in result.errors:
            html += f"""        <div class="error-box">
            <strong>ERROR:</strong> {test}<br>
            <pre>{traceback}</pre>
        </div>
"""

    # Slowest tests
    if result.test_times:
        html += """        <h2>Slowest Tests (Top 10)</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>Test</th>
            </tr>
"""
        slowest = sorted(result.test_times.items(), key=lambda x: x[1], reverse=True)[:10]
        for test, elapsed in slowest:
            html += f"""            <tr>
                <td>{elapsed:.3f}s</td>
                <td>{test}</td>
            </tr>
"""
        html += """        </table>
"""

    html += f"""
        <div class="footer">
            utala: kaos 9 Test Suite<br>
            Phase 1: Baselines and Instrumentation
        </div>
    </div>
</body>
</html>
"""

    return html


def save_report(content, filename):
    """Save report to file."""
    output_dir = Path("test_reports")
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath


def generate_coverage_reports(cov, timestamp, verbosity=1):
    """
    Generate coverage reports in multiple formats.

    Args:
        cov: Coverage object
        timestamp: Timestamp string for filenames
        verbosity: Output verbosity

    Returns:
        Dict with paths to generated reports
    """
    reports = {}

    # Create coverage output directory
    coverage_dir = Path("test_reports/coverage")
    coverage_dir.mkdir(parents=True, exist_ok=True)

    # Generate text report
    text_output = StringIO()
    cov.report(file=text_output, show_missing=True)
    text_report = text_output.getvalue()

    text_file = coverage_dir / f"coverage_{timestamp}.txt"
    with open(text_file, 'w') as f:
        f.write(text_report)
    reports['text'] = text_file

    # Generate HTML report
    html_dir = coverage_dir / f"html_{timestamp}"
    cov.html_report(directory=str(html_dir))
    reports['html'] = html_dir / 'index.html'

    # Generate XML report (for CI)
    xml_file = coverage_dir / f"coverage_{timestamp}.xml"
    cov.xml_report(outfile=str(xml_file))
    reports['xml'] = xml_file

    # Print coverage summary
    if verbosity > 0:
        print("\n" + "=" * 80)
        print("CODE COVERAGE")
        print("=" * 80)
        print(text_report)
        print("=" * 80)
        print(f"\nCoverage reports generated:")
        print(f"  Text:  {reports['text']}")
        print(f"  HTML:  {reports['html']}")
        print(f"  XML:   {reports['xml']}")
        print(f"\nOpen HTML report: file://{reports['html'].absolute()}")
        print("=" * 80)

    return reports


def main():
    """Main test runner."""
    print("=" * 80)
    print("utala: kaos 9 - Unit Tests")
    print("=" * 80)
    print()

    # Parse arguments
    verbosity = 2
    generate_report = False
    report_format = 'text'  # 'text' or 'html' or 'both'
    use_coverage = False

    args = sys.argv[1:]
    for arg in args:
        if arg in ['-q', '--quiet']:
            verbosity = 0
        elif arg in ['-v', '--verbose']:
            verbosity = 2
        elif arg in ['-r', '--report']:
            generate_report = True
        elif arg == '--html':
            report_format = 'html'
            generate_report = True
        elif arg == '--both':
            report_format = 'both'
            generate_report = True
        elif arg in ['-c', '--coverage']:
            use_coverage = True

    # Check if coverage is available
    if use_coverage and not COVERAGE_AVAILABLE:
        print("Warning: coverage package not installed. Install with: pip install coverage")
        print("Running tests without coverage tracking.")
        use_coverage = False

    # Start coverage if requested
    cov = None
    if use_coverage:
        cov = coverage.Coverage(
            source=['src/utala'],
            omit=['*/tests/*', '*/test_*']
        )
        cov.start()

    # Run tests
    start_time = time.time()
    result, suite = run_all_tests(verbosity=verbosity)
    end_time = time.time()

    # Stop coverage
    if cov:
        cov.stop()
        cov.save()

    # Print summary
    if verbosity > 0:
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Tests run: {result.testsRun}")
        print(f"Passed: {len(result.success_tests)}")
        print(f"Failed: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        print(f"Time: {end_time - start_time:.2f}s")
        print("=" * 80)

        if result.wasSuccessful():
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")

        print("=" * 80)

    # Generate reports
    timestamp = None
    if generate_report or not result.wasSuccessful():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if report_format in ['text', 'both']:
            text_report = generate_text_report(result, suite, start_time, end_time)
            text_file = save_report(text_report, f"test_report_{timestamp}.txt")
            print(f"\nText report saved to: {text_file}")

        if report_format in ['html', 'both']:
            html_report = generate_html_report(result, suite, start_time, end_time)
            html_file = save_report(html_report, f"test_report_{timestamp}.html")
            print(f"HTML report saved to: {html_file}")
            print(f"Open in browser: file://{html_file.absolute()}")

    # Generate coverage reports
    if cov:
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        generate_coverage_reports(cov, timestamp, verbosity)

    # Return exit code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
