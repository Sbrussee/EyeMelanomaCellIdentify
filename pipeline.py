#!/usr/bin/env python3
"""CLI entry point for the Eye Melanoma MRXS pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from eyemelanoma.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MRXS cell classification pipeline.")
    parser.add_argument("input_dir", type=Path, help="Directory containing .mrxs slides.")
    parser.add_argument("output_dir", type=Path, help="Output directory for pipeline artifacts.")
    parser.add_argument("--suffixes", nargs="*", default=[".mrxs"], help="Slide suffixes to include.")
    args = parser.parse_args()

    run_pipeline(args.input_dir, args.output_dir, suffixes=tuple(args.suffixes))


if __name__ == "__main__":
    main()
