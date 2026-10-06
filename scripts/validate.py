#!/usr/bin/env python3
"""Convenience launcher; install the package with pip install -e . first."""
import sys
from ocean_eddy.cli import main

if __name__ == "__main__":
    main(["validate", *sys.argv[1:]])
