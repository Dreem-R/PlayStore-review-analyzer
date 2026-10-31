"""Write deploy/streamlit/requirements.txt from uv.lock.

Streamlit Community Cloud installs from a requirements file, so the deployment
gets the exact versions CI tests. Run this after changing dependencies; CI runs
it with --check and fails if the file is out of date.

Usage: python scripts/export_deploy_requirements.py [--check]
"""

import subprocess
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "deploy" / "streamlit" / "requirements.txt"
HEADER = """\
# Generated from uv.lock by scripts/export_deploy_requirements.py. Don't edit by hand.
# CPU-only torch on Linux: the default build pulls in several GB of CUDA libraries.
--find-links https://download.pytorch.org/whl/cpu/torch/
"""
EXPORT = [
    "uv",
    "export",
    "--frozen",
    "--no-dev",
    "--extra",
    "ml",
    "--extra",
    "ui",
    "--no-hashes",
    "--no-emit-project",
    "--no-header",
]


def render() -> str:
    exported = subprocess.run(EXPORT, capture_output=True, text=True, check=True).stdout
    return HEADER + exported


def main() -> int:
    expected = render()
    if "--check" in sys.argv[1:]:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != expected:
            print(f"{TARGET} is out of date. Run: python scripts/export_deploy_requirements.py")
            return 1
        print(f"{TARGET} is up to date.")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(expected, encoding="utf-8", newline="\n")
    print(f"Wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
