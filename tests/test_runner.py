#!/usr/bin/env python3
"""
Test runner for the percentile-grids application.

This script runs all tests and provides a summary of results.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests using pytest."""
    print("Running tests for percentile-grids application...")
    print("=" * 60)

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Run pytest with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=grids",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
    ]

    try:
        subprocess.run(cmd, cwd=project_root, check=True)
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False


def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"Running specific test: {test_file}")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    test_path = project_root / "tests" / test_file

    if not test_path.exists():
        print(f"❌ Test file {test_file} not found!")
        return False

    cmd = [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"]

    try:
        subprocess.run(cmd, cwd=project_root, check=True)
        print("\n" + "=" * 60)
        print("✅ Test passed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with exit code {e.returncode}")
        return False


def main():
    """Main function to run tests."""
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        success = run_specific_test(test_file)
    else:
        # Run all tests
        success = run_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
