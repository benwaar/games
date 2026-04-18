#!/usr/bin/env python3
"""
Code quality checker for utala: kaos 9.

Runs linting, type checking, and security analysis.
Generates detailed reports for code quality metrics.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class QualityChecker:
    """Run code quality checks and generate reports."""

    def __init__(self, source_dir: str = "src", verbose: bool = True):
        """
        Initialize quality checker.

        Args:
            source_dir: Directory to check
            verbose: Print detailed output
        """
        self.source_dir = source_dir
        self.verbose = verbose
        self.results = {}

    def run_ruff(self) -> Tuple[int, str, str]:
        """
        Run ruff linter.

        Returns:
            (exit_code, stdout, stderr)
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("RUFF LINTER")
            print("=" * 80)

        cmd = ["ruff", "check", self.source_dir, "--output-format=full"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if self.verbose:
            if result.returncode == 0:
                print("✓ No linting issues found")
            else:
                print(result.stdout)

        return result.returncode, result.stdout, result.stderr

    def run_mypy(self) -> Tuple[int, str, str]:
        """
        Run mypy type checker.

        Returns:
            (exit_code, stdout, stderr)
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("MYPY TYPE CHECKING")
            print("=" * 80)

        cmd = ["mypy", self.source_dir]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if self.verbose:
            if result.returncode == 0:
                print("✓ No type errors found")
            else:
                print(result.stdout)

        return result.returncode, result.stdout, result.stderr

    def run_bandit(self) -> Tuple[int, str, str]:
        """
        Run bandit security scanner.

        Returns:
            (exit_code, stdout, stderr)
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("BANDIT SECURITY SCAN")
            print("=" * 80)

        cmd = ["bandit", "-r", self.source_dir, "-f", "txt"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if self.verbose:
            # Bandit returns 0 only if no issues
            # 1 if issues found but not severe
            if result.returncode == 0:
                print("✓ No security issues found")
            else:
                print(result.stdout)

        return result.returncode, result.stdout, result.stderr

    def run_all(self) -> Dict[str, Tuple[int, str, str]]:
        """
        Run all quality checks.

        Returns:
            Dictionary of results {tool_name: (exit_code, stdout, stderr)}
        """
        self.results = {
            "ruff": self.run_ruff(),
            "mypy": self.run_mypy(),
            "bandit": self.run_bandit(),
        }

        return self.results

    def generate_report(self) -> str:
        """
        Generate a comprehensive quality report.

        Returns:
            Report string
        """
        lines = []

        lines.append("=" * 80)
        lines.append("CODE QUALITY REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Source Directory: {self.source_dir}")
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)

        all_passed = True
        for tool, (exit_code, stdout, stderr) in self.results.items():
            status = "✓ PASS" if exit_code == 0 else "✗ FAIL"
            lines.append(f"{tool.upper():<20} {status}")
            if exit_code != 0:
                all_passed = False

        lines.append("")

        if all_passed:
            lines.append("Overall Status: ✓ ALL CHECKS PASSED")
        else:
            lines.append("Overall Status: ✗ SOME CHECKS FAILED")

        lines.append("")

        # Detailed results
        lines.append("DETAILED RESULTS")
        lines.append("-" * 80)

        for tool, (exit_code, stdout, stderr) in self.results.items():
            lines.append("")
            lines.append(f"{tool.upper()} (exit code: {exit_code})")
            lines.append("-" * 80)

            if stdout:
                lines.append(stdout)
            else:
                lines.append("No output")

            if stderr and stderr.strip():
                lines.append("\nStderr:")
                lines.append(stderr)

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_report(self, report: str, filename: str):
        """
        Save report to file.

        Args:
            report: Report content
            filename: Output filename
        """
        output_dir = Path("test_reports/quality")
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            f.write(report)

        return filepath


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run code quality checks on utala: kaos 9"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "-s", "--source",
        default="src",
        help="Source directory to check (default: src)"
    )
    parser.add_argument(
        "-r", "--report",
        action="store_true",
        help="Generate and save detailed report"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 80)
        print("utala: kaos 9 - Code Quality Checker")
        print("=" * 80)
        print()

    # Run quality checks
    checker = QualityChecker(source_dir=args.source, verbose=verbose)
    results = checker.run_all()

    # Generate report
    if args.report or not verbose:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report = checker.generate_report()

        if args.report:
            filepath = checker.save_report(report, f"quality_report_{timestamp}.txt")
            print(f"\nQuality report saved to: {filepath}")

        if not verbose:
            print(report)

    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("QUALITY CHECK SUMMARY")
        print("=" * 80)

        all_passed = True
        for tool, (exit_code, _, _) in results.items():
            status = "✓ PASS" if exit_code == 0 else "✗ FAIL"
            print(f"{tool.upper():<20} {status}")
            if exit_code != 0:
                all_passed = False

        print("=" * 80)

        if all_passed:
            print("✓ ALL QUALITY CHECKS PASSED")
        else:
            print("✗ SOME QUALITY CHECKS FAILED")

        print("=" * 80)

    # Return exit code (0 if all passed)
    exit_code = 0 if all(r[0] == 0 for r in results.values()) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
