"""Shared progress bar utilities for CLI and SLURM-friendly output."""

from __future__ import annotations

import sys
from typing import Iterable, Iterator, Optional, TypeVar


T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
) -> Iterable[T]:
    """
    Wrap an iterable with a tqdm progress bar if available.

    Parameters
    ----------
    iterable
        Iterable to wrap.
    total
        Optional total length for the progress bar.
    desc
        Optional description to display alongside the progress bar.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True,
    )
