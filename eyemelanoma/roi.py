"""Utilities for extracting ROI polygons from MRXS .dat annotation files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from shapely.geometry import Polygon


def find_last_dat(dat_dir: Path) -> Path:
    """Return the lexicographically-last .dat file in a directory."""
    dat_files = sorted(dat_dir.glob("*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {dat_dir}")
    return dat_files[-1]


def _extract_xml_payload(raw: bytes) -> str:
    """Extract an XML-like payload from a binary .dat file."""
    try:
        start = raw.index(b"<")
        end = raw.rindex(b">")
    except ValueError:
        return ""
    payload = raw[start : end + 1].replace(b"\x00", b"")
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace").strip()


def _parse_polygon_points(root: ET.Element) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for point in root.findall(".//polygon_point"):
        x = point.attrib.get("polypointX")
        y = point.attrib.get("polypointY")
        if x is None or y is None:
            continue
        try:
            points.append((float(x), float(y)))
        except ValueError:
            continue
    return points


def roi_polygon_from_dat(dat_path: Path) -> Polygon:
    """
    Parse a .dat file and return an ROI polygon.

    Parameters
    ----------
    dat_path : Path
        Path to a 3DHISTECH .dat annotation file.

    Returns
    -------
    Polygon
        ROI polygon in slide-level coordinates.
    """
    raw = dat_path.read_bytes()
    xml_text = _extract_xml_payload(raw)
    if not xml_text:
        raise ValueError(f"No XML payload found in {dat_path}")

    try:
        root = ET.fromstring(f"<Wrapper>{xml_text}</Wrapper>")
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse XML from {dat_path}: {exc}") from exc

    points = _parse_polygon_points(root)
    if len(points) < 3:
        raise ValueError(f"Not enough polygon points found in {dat_path}")

    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise ValueError(f"Parsed polygon is empty for {dat_path}")
    return polygon


def load_roi_from_mrxs_dir(mrxs_data_dir: Path) -> Polygon:
    """
    Load the ROI polygon from the last .dat file in an MRXS data directory.

    Parameters
    ----------
    mrxs_data_dir : Path
        Directory containing .dat files.
    """
    dat_path = find_last_dat(mrxs_data_dir)
    return roi_polygon_from_dat(dat_path)
