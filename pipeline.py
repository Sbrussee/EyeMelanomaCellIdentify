#!/usr/bin/env python3
"""
LazySlide: End‑to‑End MRXS Pipeline

What this script does per slide (.mrxs in a directory):
 1) Read slide
 2) Find tissue, tile, and run cell segmentation (InstanSeg)
 3) Classify cells (NuLite)
 4) Extract per‑cell morphology + color + texture features
 5) Summarize per‑slide:
      • Cell‑type distribution vector
      • Per‑celltype mean of nuclei features
      • TITAN slide‑level vector (foundation model)
 6) Cluster slides four ways (TITAN / composition / nuclei means / combined)
 7) UMAP + t‑SNE visualizations for each clustering

Outputs written under an output folder per slide plus cohort‑level CSV/plots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.affinity import translate

# Imaging + texture
from PIL import Image
from skimage.color import rgb2hsv
from skimage.draw import polygon as draw_polygon
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

# ML + viz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
try:
    # Preferred when umap-learn is installed as "umap"
    from umap import UMAP as _UMAP_CLS
except Exception:
    try:
        # Many envs expose the class in umap.umap_
        from umap.umap_ import UMAP as _UMAP_CLS
    except Exception:
        _UMAP_CLS = None  # No UMAP available; we'll skip it at runtime
        print("[WARN] UMAP not available; skipping UMAP visualizations.", file=sys.stderr)
        
import matplotlib.pyplot as plt

# LazySlide stack
import lazyslide as zs
from wsidata import open_wsi
from spatialdata.models import ShapesModel

# -------------------------------
# Config
# -------------------------------
@dataclass
class Params:
    tile_px: int = 512
    overlap: float = 0.20
    mpp_tiles: float = 0.5
    background_fraction: float = 0.95
    seg_batch: int = 16
    cls_batch: int = 8
    # Per‑cell patch sampling
    patch_expansion_px: int = 6  # pad bbox around a cell for color/texture
    reader_level_for_color: Optional[int] = None  # None: auto choose by mpp
    color_mpp: float = 0.5  # when auto‑choosing a level
    # Texture (GLCM) distances & angles
    glcm_distances: Tuple[int, ...] = (1, 2, 4)
    glcm_angles: Tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    random_state: int = 17
    # Clustering
    k_min: int = 4
    k_max: int = 8


# -------------------------------
# Utilities
# -------------------------------

from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple
import xml.etree.ElementTree as ET

def export_mrxs_annotations_asap(
    mrxs_directory: Union[str, Path],
    out_xml: Union[str, Path],
    *,
    recursive: bool = True,
    file_globs: Optional[List[str]] = None,      # e.g. ["*.mrxs", "*.dat", "*.ini"]
    max_bytes_per_file: int = 50 * 1024 * 1024,  # read cap per file (50MB)
) -> int:
    """
    Scan an MRXS directory for 3DHISTECH/CaseViewer annotation fragments and export to ASAP XML.

    - Finds XML-like blocks embedded in files (from the first '<' to the last '>').
    - Looks for: slide_flag_data/slide_flag + polygon_data/polygon_points/polygon_point
    - Maps each polygon to <Annotation Type="Polygon"> with ASAP XML structure.

    Returns
    -------
    int
        Number of polygon annotations written.
    """

    mrxs_directory = Path(mrxs_directory)
    out_xml = Path(out_xml)
    if not mrxs_directory.exists() or not mrxs_directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {mrxs_directory}")

    # ---------- ASAP XML scaffold ----------
    asap_root = ET.Element("ASAP_Annotations")
    asap_anns = ET.SubElement(asap_root, "Annotations")
    asap_groups = ET.SubElement(asap_root, "AnnotationGroups")

    # Track groups to emit once
    group_seen: Dict[str, str] = {}

    # ---------- helpers ----------
    def _decode_bytes(b: bytes) -> str:
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                return b.decode(enc)
            except UnicodeDecodeError:
                continue
        return b.decode("utf-8", errors="replace")

    def _extract_xml_text(raw: bytes) -> str:
        """Return everything between the first '<' and the last '>' (strip NULs)."""
        try:
            start = raw.index(b"<")
            end = raw.rindex(b">")
        except ValueError:
            return ""
        txt = _decode_bytes(raw[start : end + 1]).replace("\x00", "")
        return txt.strip()

    def _indent(elem: ET.Element) -> None:
        try:
            ET.indent(elem)  # py3.9+
        except AttributeError:
            def _rec(e: ET.Element, level: int = 0):
                i = "\n" + level * "  "
                if len(e):
                    if not e.text or not e.text.strip():
                        e.text = i + "  "
                    for c in e:
                        _rec(c, level + 1)
                    if not e.tail or not e.tail.strip():
                        e.tail = i
                else:
                    if level and (not e.tail or not e.tail.strip()):
                        e.tail = i
            _rec(elem)

    def _to_rgb_hex(color_int: str) -> str:
        """
        Convert decimal color attribute (e.g., '16711680') to '#RRGGBB'.
        Assumes it's standard 0xRRGGBB (16711680 -> #FF0000).
        """
        try:
            val = int(color_int)
            return f"#{val:06X}"
        except Exception:
            # Fallback to black if missing/invalid
            return "#000000"

    def _iter_files() -> List[Path]:
        if file_globs:
            paths: List[Path] = []
            for pattern in file_globs:
                paths.extend((mrxs_directory.rglob if recursive else mrxs_directory.glob)(pattern))
            # de-dupe
            unique, seen = [], set()
            for p in paths:
                if p.is_file() and p not in seen:
                    seen.add(p)
                    unique.append(p)
            return unique
        it = mrxs_directory.rglob("*") if recursive else mrxs_directory.glob("*")
        return [p for p in it if p.is_file()]

    def _parse_annotation_blocks(xml_text: str) -> List[Tuple[Dict[str, str], List[Tuple[float, float]]]]:
        """
        Parse MRXS annotation blocks and return a list of (meta, coordinates) where:
          meta: dict with keys like Text, LineColor, ObjectGUID, etc.
          coordinates: list of (x, y) floats
        """
        if not xml_text:
            return []
        # Wrap to allow multiple top-level nodes
        try:
            wrapper = ET.fromstring(f"<Wrapper>{xml_text}</Wrapper>")
        except ET.ParseError:
            return []

        results: List[Tuple[Dict[str, str], List[Tuple[float, float]]]] = []
        # Find slide_flag_data nodes (they contain both slide_flag and polygon_data)
        for sfd in wrapper.findall(".//slide_flag_data"):
            # slide_flag holds metadata (Text, LineColor, etc.)
            sf = sfd.find("slide_flag")
            meta: Dict[str, str] = {}
            if sf is not None and sf.attrib:
                meta.update(sf.attrib)

            # One polygon_data under the same slide_flag_data
            poly = sfd.find("polygon_data")
            if poly is None:
                # Sometimes polygon_data is a sibling; try search within sfd
                poly = sfd.find(".//polygon_data")
            if poly is None:
                continue

            pts_parent = poly.find("polygon_points")
            if pts_parent is None:
                continue

            coords: List[Tuple[float, float]] = []
            for pt in pts_parent.findall("polygon_point"):
                x = pt.attrib.get("polypointX")
                y = pt.attrib.get("polypointY")
                if x is None or y is None:
                    continue
                try:
                    coords.append((float(x), float(y)))
                except ValueError:
                    continue

            if coords:
                results.append((meta, coords))

        # Fallback: some files put polygon_data outside slide_flag_data
        if not results:
            for poly in wrapper.findall(".//polygon_data"):
                meta: Dict[str, str] = {}
                sf = wrapper.find(".//slide_flag")
                if sf is not None and sf.attrib:
                    meta.update(sf.attrib)
                pts_parent = poly.find("polygon_points")
                if pts_parent is None:
                    continue
                coords: List[Tuple[float, float]] = []
                for pt in pts_parent.findall("polygon_point"):
                    x = pt.attrib.get("polypointX")
                    y = pt.attrib.get("polypointY")
                    if x is None or y is None:
                        continue
                    try:
                        coords.append((float(x), float(y)))
                    except ValueError:
                        continue
                if coords:
                    results.append((meta, coords))

        return results

    # ---------- scan files and collect polygons ----------
    ann_counter = 0
    for p in _iter_files():
        try:
            raw = p.read_bytes()[:max_bytes_per_file]
        except Exception:
            continue
        text = _extract_xml_text(raw)
        if not text or "<polygon_point" not in text:
            continue

        for meta, coords in _parse_annotation_blocks(text):
            ann_name = meta.get("Text", f"Annotation {ann_counter+1}") or f"Annotation {ann_counter+1}"
            group = meta.get("Text", "Default") or "Default"  # simple grouping by Text label
            color = _to_rgb_hex(meta.get("LineColor", "0"))

            # Ensure group exists once
            if group not in group_seen:
                ET.SubElement(asap_groups, "Group", {
                    "Name": group,
                    "PartOfGroup": "None",
                    "Color": color
                })
                group_seen[group] = color

            # Build ASAP Annotation
            ann_el = ET.SubElement(asap_anns, "Annotation", {
                "Name": ann_name,
                "Type": "Polygon",
                "PartOfGroup": group,
                "Color": color
            })
            coords_el = ET.SubElement(ann_el, "Coordinates")
            for i, (x, y) in enumerate(coords):
                ET.SubElement(coords_el, "Coordinate", {
                    "Order": str(i),
                    "X": f"{x:.6f}",
                    "Y": f"{y:.6f}",
                })

            # Optional: stash extra metadata (source, GUIDs) as Attributes
            extra_attrs = {
                "SourceFile": str(p),
            }
            for k in ("ObjectGUID", "TextBkColor", "apX", "apY", "brLeft", "brTop", "brRight", "brBottom"):
                if k in meta:
                    extra_attrs[k] = meta[k]
            if extra_attrs:
                attrs_el = ET.SubElement(ann_el, "Attributes")
                for k, v in extra_attrs.items():
                    ET.SubElement(attrs_el, "Attribute", {"Name": k, "Value": str(v)})

            ann_counter += 1

    # ---------- write ASAP XML ----------
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    _indent(asap_root)
    tree = ET.ElementTree(asap_root)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)

    return ann_counter
    
def list_slides(input_dir: Path, suffixes=(".mrxs",)) -> List[Path]:
    slides = []
    for suf in suffixes:
        slides.extend(sorted(Path(input_dir).glob(f"**/*{suf}")))
    return slides


def best_k_kmeans(X: np.ndarray, k_min=2, k_max=8, random_state=0) -> Tuple[int, KMeans, np.ndarray]:
    best_k, best_score, best_model, best_labels = k_min, -1, None, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_model, best_labels = k, score, km, labels
    if best_model is None:
        best_model = KMeans(n_clusters=k_min, n_init=20, random_state=random_state).fit(X)
        best_labels = best_model.labels_
    return best_k, best_model, best_labels

def crop_wsi_to_pyramidal_tiff(
    src_path: str,
    bbox: tuple[int, int, int, int],   # (x, y, w, h) in level-0 pixels
    dst_path: str,
    tile: int = 512,
    compression: str = "jpeg",
    quality: int = 90,
    bigtiff: bool = True,
    try_copy_mpp: bool = True,
    ):
    import openslide
    import pyvips
    x, y, w, h = map(int, bbox)

    slide = openslide.OpenSlide(src_path)
    W, H = slide.dimensions

    # Clamp to level-0 dimensions
    x = max(0, x)
    y = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    w = x2 - x
    h = y2 - y
    if w <= 0 or h <= 0:
        slide.close()
        raise ValueError("Crop region empty after clamping to OpenSlide dimensions.")

    # Read RGBA region from level 0
    pil_rgba = slide.read_region((x, y), 0, (w, h))  # PIL image RGBA
    pil_rgb = pil_rgba.convert("RGB")
    arr = np.asarray(pil_rgb, dtype=np.uint8)        # H x W x 3

    # Optional MPP copy
    mpp_x = mpp_y = None
    if try_copy_mpp:
        try:
            mpp_x = slide.properties.get("openslide.mpp-x")
            mpp_y = slide.properties.get("openslide.mpp-y")
        except Exception:
            pass

    slide.close()

    # Wrap numpy -> vips image (memory is row-major; set correct stride)
    mem = arr.tobytes()
    vips_img = pyvips.Image.new_from_memory(mem, arr.shape[1], arr.shape[0], 3, "uchar")

    save_kwargs = dict(
        tile=True, tile_width=tile, tile_height=tile,
        pyramid=True, bigtiff=bigtiff, compression=compression,
    )
    if compression.lower() == "jpeg":
        save_kwargs["Q"] = int(quality)

    # Write resolution if we have MPP (OpenSlide gives microns/pixel)
    if try_copy_mpp and mpp_x and mpp_y:
        try:
            mpp_x = float(mpp_x); mpp_y = float(mpp_y)
            if mpp_x > 0 and mpp_y > 0:
                xres = 10000.0 / mpp_x  # pixels/cm
                yres = 10000.0 / mpp_y
                save_kwargs.update(dict(resunit="cm", xres=xres, yres=yres))
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    vips_img.tiffsave(dst_path, **save_kwargs)

    # Return path and offset for annotation shifting
    return dst_path, (x, y)

def load_xml_annotations_as_dataframe(xml_path: Path, offset: Tuple[int, int] = (0, 0)):
    """
    Parses ASAP XML, returns a GeoDataFrame suitable for wsi.shapes.
    """
    if not xml_path.exists():
        return None
    
    # Ensure GeoPandas is available
    try:
        import geopandas as gpd
    except ImportError:
        print("[ERROR] Geopandas not installed. Cannot load annotations.")
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        geoms = []
        classes = []
        names = []
        
        off_x, off_y = offset
        
        for ann in root.findall(".//Annotation"):
            name = ann.get("Name", "Cell")
            group = ann.get("PartOfGroup", "Default")
            
            coords_el = ann.find("Coordinates")
            if coords_el is None: continue
            
            points = []
            for c in coords_el.findall("Coordinate"):
                points.append((int(c.get("Order")), float(c.get("X")), float(c.get("Y"))))
            points.sort(key=lambda item: item[0])
            
            # Shift coordinates
            poly_pts = [(p[1] - off_x, p[2] - off_y) for p in points]
            
            if len(poly_pts) < 3: continue
            
            poly = Polygon(poly_pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
                
            if not poly.is_empty:
                geoms.append(poly)
                classes.append(group)
                names.append(name)
                
        if not geoms:
            return None
            
        # --- FIX: Return GeoDataFrame ---
        gdf = gpd.GeoDataFrame({
            "geometry": geoms,
            "class": classes,
            "orig_name": names
        })
        return gdf
        
    except Exception as e:
        print(f"[WARN] Failed to load XML to GeoDataFrame: {e}")
        return None

# -------------------------------
# Cell‑level features
# -------------------------------

def polygon_to_mask(poly: Polygon, H: int, W: int) -> np.ndarray:
    """Rasterize polygon (in this local patch coordinate frame) to a boolean mask."""
    x, y = poly.exterior.xy
    rr, cc = draw_polygon(np.array(y), np.array(x), (H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    return mask


def hsv_melanin_fraction(rgb_patch: np.ndarray, mask: np.ndarray) -> float:
    """Estimate melanin as fraction of brown/very dark pixels inside mask (0-1).
    Heuristic: HSV with S>0.5 and 15°≤H≤45° (brownish), OR V<0.35 (very dark).
    """
    if rgb_patch.dtype != np.float32 and rgb_patch.dtype != np.float64:
        arr = rgb_patch.astype(np.float32) / 255.0
    else:
        arr = rgb_patch
    hsv = rgb2hsv(arr)
    H = hsv[..., 0] * 360.0
    S = hsv[..., 1]
    V = hsv[..., 2]
    brown = (S > 0.5) & (H >= 15) & (H <= 45)
    dark = V < 0.35
    mel_mask = (brown | dark) & mask
    total = mask.sum()
    return float(mel_mask.sum()) / float(total) if total > 0 else 0.0


def texture_features(gray_u8: np.ndarray, mask: np.ndarray,
                     distances=(1, 2, 4), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)) -> Dict[str, float]:
    """Compute masked GLCM props by zeroing outside mask then sampling GLCM.
    Returns mean across distances,angles for: contrast, homogeneity, energy, correlation, dissimilarity.
    """
    # Apply mask; keep 8‑bit range
    arr = gray_u8.copy()
    arr[~mask] = 0
    glcm = graycomatrix(arr, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    feats = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]:
        v = graycoprops(glcm, prop)
        feats[prop] = float(np.nanmean(v))
    return feats


def geometry_features(poly: Polygon, mpp: float) -> Dict[str, float]:
    """Basic polygon morphology in microns using provided MPP."""
    area_px = poly.area
    perim_px = poly.length
    area_um2 = area_px * (mpp ** 2)
    perim_um = perim_px * mpp
    # Fit ellipse via second moments (approximate using shapely minimum bounding ellipse surrogate)
    try:
        circ = 4 * math.pi * area_px / (perim_px ** 2) if perim_px > 0 else 0.0
    except ZeroDivisionError:
        circ = 0.0
    return {
        "area_um2": float(area_um2),
        "perimeter_um": float(perim_um),
        "circularity": float(circ),
        # Compactness alternative
        "solidity": float(poly.area / poly.convex_hull.area) if poly.convex_hull.area > 0 else 0.0,
        "eccentricity_like": float(poly.minimum_rotated_rectangle.length / (poly.minimum_rotated_rectangle.minimum_clearance + 1e-6)),
    }

def _get_downsamples(reader) -> List[float]:
    """
    Return a list of level downsample factors (level 0 = 1.0).
    Tries OpenSlide first, then derives from level_dimensions, then a safe fallback.
    """
    # OpenSlide: already provides per-level downsample factors
    downs = getattr(reader, "level_downsamples", None)
    if downs is not None and len(downs) > 0:
        return [float(d) for d in downs]

    # If we have dimensions per level, compute relative to level-0
    level_dims = getattr(reader, "level_dimensions", None)
    if level_dims:
        w0, h0 = level_dims[0]
        derived = []
        for (w, h) in level_dims:
            # use max scale so we stay conservative
            d = max(w0 / float(w), h0 / float(h))
            derived.append(float(d))
        return derived

    # If we only know the count, assume powers of two (common default)
    level_count = getattr(reader, "level_count", None)
    if isinstance(level_count, int) and level_count > 0:
        return [float(2 ** i) for i in range(level_count)]

    # Single-level or unknown pyramid info
    return [1.0]


def _get_base_mpp(wsi, reader) -> float:
    """
    Try hard to obtain base (level-0) microns-per-pixel.
    """
    # Prefer attributes on the WSI wrapper if present
    for attr in ("mpp", "MPP", "mpp_x"):
        v = getattr(wsi, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)

    # OpenSlide properties often carry mpp-x / mpp-y
    props = getattr(reader, "properties", {}) or {}
    try:
        mppx = float(props.get("openslide.mpp-x", "nan"))
        mppy = float(props.get("openslide.mpp-y", "nan"))
        if np.isfinite(mppx) and np.isfinite(mppy) and mppx > 0 and mppy > 0:
            return float((mppx + mppy) / 2.0)
        if np.isfinite(mppx) and mppx > 0:
            return float(mppx)
        if np.isfinite(mppy) and mppy > 0:
            return float(mppy)
    except Exception:
        pass

    # Sensible default if everything else fails
    return 0.25

def read_cell_patch_rgba(wsi, bbox_xyxy: Tuple[int, int, int, int],
                         level: Optional[int], color_mpp: float) -> Tuple[np.ndarray, float]:
    """
    Read an RGB patch from the slide at the requested pyramid level (or auto-chosen by color_mpp).
    Returns (RGB uint8 HxWx3, effective_mpp).

    Order matters:
      • WSIData.read_region(x, y, width, height, level=...)
      • reader.get_region(x, y, width, height, level=...)
      • OpenSlide.read_region((x, y), level, (width, height))  # PIL Image
    """
    reader = getattr(wsi, "reader", None)
    if reader is None:
        raise RuntimeError("WSI reader not found on wsi object; adapt read_cell_patch_rgba().")

    # Pyramid info
    downs = _get_downsamples(reader)
    n_levels = len(downs)

    # Base (level-0) MPP
    base_mpp = _get_base_mpp(wsi, reader)

    # Choose level if not provided
    if level is None:
        target_down = max(color_mpp / float(base_mpp), 1.0)
        level = int(np.argmin([abs(d - target_down) for d in downs]))

    # Clamp level
    level = max(0, min(int(level), n_levels - 1))
    effective_mpp = float(base_mpp) * float(downs[level])

    # Compute read size on the target level's pixel grid
    x0, y0, x1, y1 = bbox_xyxy
    W = max(1, int(x1 - x0))
    H = max(1, int(y1 - y0))
    scale = float(downs[level])
    wL = max(1, int(math.ceil(W / scale)))
    hL = max(1, int(math.ceil(H / scale)))

    # ---------- Try WSIData/WSIData.reader APIs correctly ----------
    tile = None

    # 1) Preferred: WSIData.read_region(x, y, width, height, level=...)
    if hasattr(wsi, "read_region"):
        try:
            tile = wsi.read_region(int(x0), int(y0), int(wL), int(hL), level=int(level))
        except TypeError:
            # Some versions may require keywords
            tile = wsi.read_region(x=int(x0), y=int(y0), width=int(wL), height=int(hL), level=int(level))

    # 2) ReaderBase.get_region(x, y, width, height, level=0)
    if tile is None and hasattr(reader, "get_region"):
        try:
            tile = reader.get_region(int(x0), int(y0), int(wL), int(hL), level=int(level))
        except TypeError:
            tile = reader.get_region(x=int(x0), y=int(y0), width=int(wL), height=int(hL), level=int(level))

    # 3) OpenSlide.read_region((x, y), level, (w, h)) -> PIL.Image
    if tile is None and hasattr(reader, "read_region"):
        tile = reader.read_region((int(x0), int(y0)), int(level), (int(wL), int(hL)))

    if tile is None:
        raise AttributeError(
            "Could not obtain region: tried WSIData.read_region, reader.get_region, reader.read_region."
        )

    # ---------- Normalize to uint8 HxWx3 ----------
    from PIL import Image as _PILImage
    import numpy as _np

    if isinstance(tile, _PILImage.Image):
        arr = _np.array(tile.convert("RGB"))
    else:
        arr = _np.asarray(tile)
        # WSIData returns xyc (width, height, channels); convert to HWC if needed
        if arr.ndim == 3 and arr.shape[0] == int(wL) and arr.shape[1] == int(hL):
            arr = _np.transpose(arr, (1, 0, 2))  # (x,y,c) -> (y,x,c)
        # RGBA -> RGB
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        # Gray -> RGB
        if arr.ndim == 2:
            arr = _np.stack([arr, arr, arr], axis=-1)
        # Ensure uint8 [0,255]
        if arr.dtype != _np.uint8:
            arr = _np.clip(arr, 0, 255).astype(_np.uint8)

    return arr, effective_mpp


def extract_cell_features_for_slide(wsi, cell_gdf, params: Params) -> pd.DataFrame:
    """Compute per-cell features. If no reader exists, compute geometry-only."""
    rows = []

    reader = getattr(wsi, "reader", None)
    can_read_patches = (hasattr(wsi, "read_region") or
                        (reader is not None and (hasattr(reader, "get_region") or hasattr(reader, "read_region"))))

    # Try to get MPP; without a reader we still try WSI attrs; fallback to 0.5
    try:
        mpp = _get_base_mpp(wsi, reader) or 0.5
    except Exception:
        mpp = 0.5

    for idx, row in cell_gdf.iterrows():
        poly: Polygon = row.geometry
        if poly is None or poly.is_empty:
            continue

        # Geometry in microns (always available)
        geom = geometry_features(poly, mpp)

        out = {
            "cell_id": idx,
            "mpp_eff": float(mpp),      # if no patch, at least report mpp used for geometry
            **geom,
        }

        if "class" in cell_gdf.columns:
            out["cell_type"] = row["class"]

        if can_read_patches:
            # --- color/texture path as before ---
            minx, miny, maxx, maxy = poly.bounds
            minx = int(math.floor(minx)) - params.patch_expansion_px
            miny = int(math.floor(miny)) - params.patch_expansion_px
            maxx = int(math.ceil(maxx)) + params.patch_expansion_px
            maxy = int(math.ceil(maxy)) + params.patch_expansion_px

            local_poly = translate(poly, xoff=-minx, yoff=-miny)

            rgb_patch, eff_mpp = read_cell_patch_rgba(
                wsi, (minx, miny, maxx, maxy),
                params.reader_level_for_color, params.color_mpp
            )
            H, W, _ = rgb_patch.shape
            mask = polygon_to_mask(local_poly, H, W)

            mel_frac = hsv_melanin_fraction(rgb_patch, mask)
            masked = rgb_patch.copy()
            masked[~mask] = 0
            mean_rgb = masked.sum(axis=(0, 1)) / (mask.sum() + 1e-6)
            rgb = rgb_patch
            if rgb.dtype != np.uint8:
                # If float already in [0,1], scale up; otherwise clip to [0,255]
                vmax = float(np.nanmax(rgb))
                if vmax <= 1.0:
                    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            # Luma transform to uint8 grayscale
            gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
            tex = texture_features(gray, mask, params.glcm_distances, params.glcm_angles)

            out.update({
                "mpp_eff": float(eff_mpp),
                "mean_R": float(mean_rgb[0]),
                "mean_G": float(mean_rgb[1]),
                "mean_B": float(mean_rgb[2]),
                "melanin_fraction": float(mel_frac),
                **{f"tex_{k}": v for k, v in tex.items()},
            })
        else:
            # Fill color/texture with NaNs if not available
            out.update({
                "mean_R": np.nan, "mean_G": np.nan, "mean_B": np.nan,
                "melanin_fraction": np.nan,
                "tex_contrast": np.nan, "tex_homogeneity": np.nan,
                "tex_energy": np.nan, "tex_correlation": np.nan,
                "tex_dissimilarity": np.nan,
            })

        rows.append(out)

    return pd.DataFrame(rows)


# -------------------------------
# Plotting helpers (cells)
# -------------------------------
from matplotlib.collections import PathCollection
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches
import matplotlib as mpl
import warnings

def _gdf_centroids_xy(cell_gdf: "gpd.GeoDataFrame") -> np.ndarray:
    """Return (N,2) array of centroid XY for each polygon (index aligned)."""
    # Late import to avoid hard dep if geopandas is not needed elsewhere
    try:
        import geopandas as gpd  # noqa: F401
    except Exception:
        # Fallback: compute via shapely only
        return np.array([[geom.centroid.x, geom.centroid.y] for geom in cell_gdf.geometry.values], dtype=float)
    return np.array([cell_gdf.geometry.iloc[i].centroid.coords[0] for i in range(len(cell_gdf))], dtype=float)

def _safe_palette(categories: List[str]) -> Dict[str, tuple]:
    """Map categories to distinct matplotlib colors."""
    # Use tab20 then cycle
    cmap = mpl.cm.get_cmap("tab20")
    pal = {}
    for i, c in enumerate(sorted(categories)):
        pal[c] = cmap(i % 20)
    return pal

def plot_spatial_points(
    xy: np.ndarray,
    color_vals: List[str],
    palette: Optional[Dict[str, tuple]],
    title: str,
    out_path: Path,
    alpha: float = 0.8,
    s: float = 2.0,
):
    """Efficient scatter of centroids colored by a categorical vector."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cats = [str(c) if (c is not None and not (isinstance(c, float) and np.isnan(c))) else "NA" for c in color_vals]
    uniq = sorted(set(cats))
    if palette is None:
        palette = _safe_palette(uniq)
    colors = [palette.get(c, (0.5, 0.5, 0.5, 1.0)) for c in cats]

    plt.figure(figsize=(7, 7))
    plt.scatter(xy[:, 0], xy[:, 1], c=colors, s=s, alpha=alpha, linewidths=0)
    plt.gca().invert_yaxis()  # slide coords: y grows downward
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    # legend
    handles = [mpatches.Patch(color=palette[u], label=u) for u in uniq]
    # Keep legend reasonable
    if len(handles) <= 20:
        plt.legend(handles=handles, fontsize=8, frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_numeric_distributions(
    df: pd.DataFrame,
    out_dir: Path,
    per_celltype: bool = True,
    celltype_col: str = "cell_type",
    n_bins: int = 40
):
    """Plot histograms for all numeric columns and optional per-celltype violins/boxplots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"cell_id"}]
    if not num_cols:
        return
    # Individual histograms
    for c in num_cols:
        try:
            plt.figure(figsize=(6, 4))
            df[c].dropna().plot(kind="hist", bins=n_bins)
            plt.title(f"{c}")
            plt.xlabel(c); plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(out_dir / f"hist__{c}.png", dpi=160)
            plt.close()
        except Exception:
            continue

    # Per-celltype boxplots for a few key features if requested
    if per_celltype and (celltype_col in df.columns) and df[celltype_col].notna().any():
        # choose up to 8 informative columns (variance descending)
        variances = [(c, float(np.nanvar(df[c].values))) for c in num_cols]
        top = [c for c, _ in sorted(variances, key=lambda x: -x[1])[:8]]
        for c in top:
            try:
                plt.figure(figsize=(8, 4))
                # Use pandas boxplot grouped by cell_type (keeps deps light)
                df.boxplot(column=c, by=celltype_col, rot=45)
                plt.suptitle("")
                plt.title(f"{c} by {celltype_col}")
                plt.xlabel(celltype_col); plt.ylabel(c)
                plt.tight_layout()
                plt.savefig(out_dir / f"by_celltype__{c}.png", dpi=160)
                plt.close()
            except Exception:
                continue

def cluster_cells_and_embeddings(
    df_features: pd.DataFrame,
    p: Params,
    label_prefix: str,
    out_dir: Path
) -> Dict[str, np.ndarray]:
    """KMeans on PCA with model selection; returns dict and writes plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # pick numeric features
    X = df_features.select_dtypes(include=[np.number]).values
    if X.size == 0 or X.shape[0] < 5:
        return {"labels": np.array([]), "tsne": None, "umap": None, "k": 0}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(30, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)

    k, km, labels = best_k_kmeans(Xp, p.k_min, p.k_max, p.random_state)

    # UMAP if available
    X_umap = None
    if _UMAP_CLS is not None:
        um = _UMAP_CLS(n_neighbors=15, min_dist=0.2, random_state=p.random_state)
        X_umap = um.fit_transform(Xp)
        plt.figure(figsize=(6, 5))
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, s=5)
        plt.title(f"Cells UMAP — {label_prefix} (k={k})")
        plt.xlabel("dim1"); plt.ylabel("dim2")
        plt.tight_layout()
        plt.savefig(out_dir / f"{label_prefix}_cells_umap.png", dpi=160)
        plt.close()
    else:
        warnings.warn("UMAP not available; skipping cell-level UMAP.")

    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=p.random_state)
    X_tsne = tsne.fit_transform(Xp)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=5)
    plt.title(f"Cells t-SNE — {label_prefix} (k={k})")
    plt.xlabel("dim1"); plt.ylabel("dim2")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label_prefix}_cells_tsne.png", dpi=160)
    plt.close()

    # Save arrays
    np.save(out_dir / f"{label_prefix}_cells_pca.npy", Xp)
    if X_umap is not None:
        np.save(out_dir / f"{label_prefix}_cells_umap.npy", X_umap)
    np.save(out_dir / f"{label_prefix}_cells_tsne.npy", X_tsne)
    np.save(out_dir / f"{label_prefix}_cells_kmeans_labels.npy", labels)

    return {"labels": labels, "umap": X_umap, "tsne": X_tsne, "k": k}


# -------------------------------
# Per‑slide processing
# -------------------------------

def process_slide(slide_path: Path, out_dir: Path, p: Params) -> Dict:
    slide_out = out_dir / slide_path.stem
    slide_out.mkdir(parents=True, exist_ok=True)
    
    mrxs_data_dir = slide_path.parent / slide_path.stem
    if not mrxs_data_dir.exists():
        mrxs_data_dir = slide_path.parent / (slide_path.name)

    print(f"Found mrxs slide: {slide_path.name}")
        
    ann_xml_path = slide_out / "annotations.xml"
    cropped_tiff_path = slide_out / f"{slide_path.stem}_roi.tiff"
    
    actual_slide_path = slide_path 
    crop_offset = (0, 0) # Store (x, y) shift if we crop
    
    # --- 1. Export Annotations ---
    has_annotations = False
    if mrxs_data_dir.exists() and list(mrxs_data_dir.glob("*.dat")):
        print(f" Found data directory: {mrxs_data_dir}, extracting annotations...")
        count = export_mrxs_annotations_asap(
            mrxs_directory=mrxs_data_dir,
            out_xml=ann_xml_path,
            recursive=False, 
            file_globs=["*.dat"]
        )
        if count > 0 and ann_xml_path.exists():
            has_annotations = True
            
            # --- 2. Calculate BBox and Crop (With Robust Clipping) ---
            bbox = get_bbox_from_asap_xml(ann_xml_path)
            if bbox:
                print(f" Cropping WSI to bounding box: {bbox}...")
                try:
                    # UPDATED CALL: returns path AND offset
                    c_path, c_off = crop_wsi_to_pyramidal_tiff(
                        src_path=str(slide_path),
                        bbox=bbox,
                        dst_path=str(cropped_tiff_path),
                        tile=p.tile_px,
                        compression="jpeg",
                        quality=90
                    )
                    actual_slide_path = Path(c_path)
                    crop_offset = c_off # Important: Shift annotations by this amount
                    print(f" Crop successful. Processing: {actual_slide_path}")
                except Exception as e:
                    print(f"[ERROR] Cropping failed (using full slide): {e}")
            else:
                print("[WARN] Annotations found but could not calculate valid bounding box.")
        else:
            print("[INFO] No polygons found in .dat files.")
    else:
        print("[INFO] No MRXS data folder found.")
        
    # --- 3. Open WSI ---
    try:
        wsi = open_wsi(str(actual_slide_path))
    except Exception as e:
        print(f"[ERROR] Failed to open WSI: {e}")
        return {} # Return empty dict on critical failure

    # --- 4. Logic Branch: Use Existing Annotations OR Run Segmentation ---
    
    # Try to load existing annotations into a DataFrame
    pre_segmented_df = None
    if has_annotations:
        print(f" Attempting to load existing annotations mapped to cropped coordinates...")
        pre_segmented_df = load_xml_annotations_as_dataframe(ann_xml_path, offset=crop_offset)

    if pre_segmented_df is not None and len(pre_segmented_df) > 0:
        print(f" [INFO] Using {len(pre_segmented_df)} existing annotations as cells.")
        print(f" [INFO] Skipping Tissue Detection, Tiling, and Segmentation.")
        
        # Map GeoDataFrame -> ShapesModel
        # Coordinates are already in level-0 pixel space, so identity transform is fine.
        cell_shapes = ShapesModel.parse(pre_segmented_df)

        wsi.shapes["cell_types"] = cell_shapes
        has_cell_types = True
        has_cells = True
        has_tiles = False
        
    else:
        # --- Standard Pipeline (No annotations found, or failed to load) ---
        print(" [INFO] No existing cell annotations used. Running standard segmentation pipeline.")
        
        # Tissues
        zs.pp.find_tissues(wsi)
        
        # Tiles
        zs.pp.tile_tissues(wsi, p.tile_px, overlap=p.overlap, background_fraction=p.background_fraction, mpp=p.mpp_tiles)
        has_tiles = True
        
        # Segmentation
        if hasattr(wsi, "tile_spec") or hasattr(wsi, "reader"):
             zs.seg.cell_types(
                wsi, model="histoplus", tile_key="tiles", batch_size=p.cls_batch,
                num_workers=0, amp=True, size_filter=False, nucleus_size=(20, 1000),
                pbar=True, key_added="cell_types"
            )

    # Persist intermediate state
    try: wsi.write()
    except Exception: pass

    # --- 5. Feature Extraction (Common Path) ---
    
    # Check what layers we have
    shapes = getattr(wsi, "shapes", {})
    has_cell_types = "cell_types" in shapes and len(shapes["cell_types"]) > 0
    has_cells = "cells" in shapes and len(shapes["cells"]) > 0
    
    cell_layer_name = "cell_types" if has_cell_types else ("cells" if has_cells else None)

    if cell_layer_name is None:
        print(f"[WARN] No cells found.")
        df_cells = pd.DataFrame()
    else:
        cell_gdf = wsi.shapes[cell_layer_name]
        try:
            df_cells = extract_cell_features_for_slide(wsi, cell_gdf, p)
        except Exception as e:
            print(f"[WARN] Feature extraction failed: {e}")
            df_cells = pd.DataFrame()
            
    df_cells.to_csv(slide_out / "cells_features.csv", index=False)
    print(f" Extracted features for {len(df_cells)} cells.")

    # 5b) Spatial & distributions at the CELL LEVEL
    cells_fig_dir = slide_out / "cells_plots"
    cells_fig_dir.mkdir(parents=True, exist_ok=True)

    # Attach centroids (from geometry) if we had a cell layer
    xy_centroids = None
    if cell_layer_name is not None:
        try:
            cell_gdf = wsi.shapes[cell_layer_name]
            xy_centroids = _gdf_centroids_xy(cell_gdf)  # (N,2), order = cell_gdf index
            # Align to df_cells rows via cell_id = original GeoDataFrame index
            # Build a mapping index -> row position
            id_to_pos = {cid: i for i, cid in enumerate(df_cells["cell_id"].tolist())}
            # Reorder centroids accordingly
            # Not all cells may survive, so guard
            centroids_reordered = np.zeros((len(df_cells), 2), dtype=float)
            for j, cid in enumerate(df_cells["cell_id"].tolist()):
                # find original position in cell_gdf index; if missing, fall back to NaNs
                try:
                    pos = int(np.where(cell_gdf.index.values == cid)[0][0])
                    centroids_reordered[j] = xy_centroids[pos]
                except Exception:
                    centroids_reordered[j] = np.array([np.nan, np.nan])
            xy_centroids = centroids_reordered
            # drop rows with NaN centroids for spatial plots
            valid_centroid_mask = np.isfinite(xy_centroids).all(axis=1)
        except Exception as e:
            print(f"[WARN] Could not compute cell centroids for spatial plots: {e}")
            xy_centroids = None

    # Per-slide cell-type barplot already handled later in cohort section,
    # but save a higher-res copy here too for convenience:
    try:
        if "cell_type" in df_cells.columns and not df_cells["cell_type"].isna().all():
            plt.figure(figsize=(7, 4))
            df_cells["cell_type"].value_counts(normalize=True).sort_index().plot(kind="bar")
            plt.title(f"Cell-type distribution — {slide_path.stem}")
            plt.ylabel("fraction")
            plt.tight_layout()
            plt.savefig(cells_fig_dir / f"{slide_path.stem}__celltype_distribution.png", dpi=180)
            plt.close()
    except Exception as e:
        print(f"[WARN] Cell-type distribution plot failed: {e}")

    # Spatial distribution of cell-types (colored centroids)
    if xy_centroids is not None and "cell_type" in df_cells.columns and df_cells["cell_type"].notna().any():
        try:
            mask = valid_centroid_mask & df_cells["cell_type"].notna().values
            xy = xy_centroids[mask]
            vals = df_cells.loc[mask, "cell_type"].astype(str).tolist()
            palette = _safe_palette(sorted(set(vals)))
            plot_spatial_points(
                xy, vals, palette,
                title=f"Spatial distribution of cell-types — {slide_path.stem}",
                out_path=cells_fig_dir / f"{slide_path.stem}__spatial_celltypes.png",
                alpha=0.7, s=2.0
            )
        except Exception as e:
            print(f"[WARN] Spatial cell-type plot failed: {e}")

    # Feature distributions (histograms; plus per-celltype boxplots for top features)
    try:
        plot_numeric_distributions(df_cells, cells_fig_dir / "feature_distributions", per_celltype=True)
    except Exception as e:
        print(f"[WARN] Feature distribution plots failed: {e}")

    # Clustering on cellular features (geometry+color+texture) and spatial back-projection
    try:
        # Choose a clean numeric table for clustering (exclude IDs and coordinates)
        num_cols = [c for c in df_cells.columns
                    if pd.api.types.is_numeric_dtype(df_cells[c])
                    and c not in {"cell_id"}]
        df_num = df_cells[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        cell_clu_dir = slide_out / "cells_clustering"
        res_cells = cluster_cells_and_embeddings(df_num, p, label_prefix="cells", out_dir=cell_clu_dir)
        labels = res_cells.get("labels", None)
        if labels is not None and labels.size == len(df_cells):
            df_cells["cell_cluster"] = labels.astype(int)
            # Save an updated CSV with clusters
            df_cells.to_csv(slide_out / "cells_features_with_clusters.csv", index=False)

            # Spatial plot by cluster
            if xy_centroids is not None:
                mask = valid_centroid_mask
                xy = xy_centroids[mask]
                vals = [f"C{int(v)}" for v in df_cells.loc[mask, "cell_cluster"].tolist()]
                palette = _safe_palette(sorted(set(vals)))
                plot_spatial_points(
                    xy, vals, palette,
                    title=f"Spatial clusters (KMeans on PCA) — {slide_path.stem}",
                    out_path=cell_clu_dir / f"{slide_path.stem}__spatial_clusters.png",
                    alpha=0.7, s=2.0
                )
    except Exception as e:
        print(f"[WARN] Cell-level clustering/spatial plotting failed: {e}")
        
    # 6) TITAN slide vector
    titan_vec = None
    try:
        zs.tl.feature_extraction(wsi, "titan")
        print(" Extracted TITAN features for tiles.")
        zs.tl.feature_aggregation(wsi, "titan", encoder="titan")
        print(" Aggregated TITAN slide vector.")
        try:
            titan_vec = wsi.fetch.features_vector("titan")
        except Exception:
            agg_key = "titan_slide"
            titan_vec = np.asarray(wsi.tables[agg_key].X).ravel() if hasattr(wsi, "tables") and agg_key in wsi.tables else None
        titan_vec = np.array(titan_vec) if titan_vec is not None else None
    except Exception as e:
        print(f"[WARN] TITAN extraction/aggregation skipped on {slide_path.name}: {e}")

    print(f" TITAN vector shape: {titan_vec.shape if titan_vec is not None else 'None'}")
    # 7) Slide‑level cell‑type composition + mean features per type
    if "cell_type" in df_cells.columns:
        type_counts = df_cells["cell_type"].value_counts().sort_index()
        comp = (type_counts / type_counts.sum()).rename_axis("cell_type").reset_index(name="fraction")
    else:
        comp = pd.DataFrame(columns=["cell_type", "fraction"])  # empty
    print(f" Cell‑type composition:\n{comp}")
    # Mean features per celltype
    feature_cols = [c for c in df_cells.columns if c not in {"cell_id", "cell_type", "mpp_eff"}]
    means_by_type = (
        df_cells.groupby("cell_type")[feature_cols]
        .mean(numeric_only=True)
        .reset_index()
        if "cell_type" in df_cells.columns and not df_cells.empty
        else pd.DataFrame(columns=["cell_type"] + feature_cols)
    )

    comp.to_csv(slide_out / "celltype_composition.csv", index=False)
    means_by_type.to_csv(slide_out / "celltype_nuclei_means.csv", index=False)
    print(" Saved slide‑level summaries.")
    # Save TITAN vector
    if titan_vec is not None and titan_vec.size > 0:
        np.save(slide_out / "titan_vector.npy", titan_vec)

    meta = {
        "slide": slide_path.name,
        "n_cells": int(len(df_cells)),
        "cell_layer": cell_layer_name,
        "has_titan": bool(titan_vec is not None and titan_vec.size > 0),
    }
    with open(slide_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(" Saved slide metadata.")
    return {
        "slide": slide_path.name,
        "cells_features_path": str(slide_out / "cells_features.csv"),
        "celltype_comp_path": str(slide_out / "celltype_composition.csv"),
        "celltype_means_path": str(slide_out / "celltype_nuclei_means.csv"),
        "titan_vector_path": str(slide_out / "titan_vector.npy") if titan_vec is not None else None,
    }


# -------------------------------
# Cohort‑level aggregation, clustering & embeddings
# -------------------------------

def build_matrix_from_compositions(per_slide_comp: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    # Only consider frames that actually have 'cell_type' and 'fraction'
    valid = {k: df for k, df in per_slide_comp.items()
             if isinstance(df, pd.DataFrame) and {"cell_type", "fraction"}.issubset(df.columns)}
    if not valid:
        return pd.DataFrame(), []

    all_types = sorted({t for df in valid.values() for t in df["cell_type"].astype(str).tolist()})
    rows = []
    for slide, df in valid.items():
        vec = {t: 0.0 for t in all_types}
        vec.update(dict(zip(df["cell_type"].astype(str), df["fraction"].astype(float))))
        vec["slide"] = slide
        rows.append(vec)
    mat = pd.DataFrame(rows).set_index("slide").sort_index()
    return mat, all_types


def build_matrix_from_celltype_means(per_slide_means: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    # Keep only frames with the 'cell_type' column
    valid = {k: df for k, df in per_slide_means.items()
             if isinstance(df, pd.DataFrame) and "cell_type" in df.columns and not df.empty}
    if not valid:
        return pd.DataFrame(), []

    all_types = sorted({t for df in valid.values() for t in df["cell_type"].astype(str).tolist()})
    # Features = all numeric columns except 'cell_type'
    def _feat_cols(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c != "cell_type" and pd.api.types.is_numeric_dtype(df[c])]
    all_feats = sorted({c for df in valid.values() for c in _feat_cols(df)})

    rows = []
    for slide, df in valid.items():
        row = {f"{t}__{f}": 0.0 for t in all_types for f in all_feats}
        for _, r in df.iterrows():
            t = str(r["cell_type"])
            for f in all_feats:
                val = r.get(f, np.nan)
                if pd.notna(val):
                    row[f"{t}__{f}"] = float(val)
        row["slide"] = slide
        rows.append(row)

    mat = pd.DataFrame(rows).set_index("slide").sort_index()
    return mat, [f"{t}__{f}" for t in all_types for f in all_feats]


def run_embeddings_and_clustering(X: np.ndarray, label_prefix: str, out_dir: Path, p: Params) -> Dict[str, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA for speed
    pca = PCA(n_components=min(50, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)

    # KMeans with silhouette model selection
    k, km, labels = best_k_kmeans(Xp, p.k_min, p.k_max, p.random_state)

    # UMAP (optional)
    X_umap = None
    if _UMAP_CLS is not None:
        um = _UMAP_CLS(n_neighbors=15, min_dist=0.2, random_state=p.random_state)
        X_umap = um.fit_transform(Xp)
        # Plot
        plt.figure(figsize=(6, 5))
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, s=24)
        plt.title(f"UMAP – {label_prefix} (k={k})")
        plt.xlabel("dim1"); plt.ylabel("dim2")
        plt.tight_layout()
        plt.savefig(out_dir / f"{label_prefix}_umap.png", dpi=160)
        plt.close()
    else:
        print(f"[WARN] UMAP not available; install 'umap-learn' to enable UMAP embeddings for {label_prefix}.")

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=p.random_state)
    X_tsne = tsne.fit_transform(Xp)

    # Save arrays
    np.save(out_dir / f"{label_prefix}_pca.npy", Xp)
    if X_umap is not None:
        np.save(out_dir / f"{label_prefix}_umap.npy", X_umap)
    np.save(out_dir / f"{label_prefix}_tsne.npy", X_tsne)
    np.save(out_dir / f"{label_prefix}_kmeans_labels.npy", labels)

    # Plot t-SNE
    plt.figure(figsize=(6, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=24)
    plt.title(f"t-SNE – {label_prefix} (k={k})")
    plt.xlabel("dim1"); plt.ylabel("dim2")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label_prefix}_tsne.png", dpi=160)
    plt.close()

    return {
        "labels": labels,
        "umap": X_umap,
        "tsne": X_tsne,
        "k": k,
    }

def get_bbox_from_asap_xml(xml_path: Path) -> Optional[Tuple[int, int, int, int]]:
    """
    Parses ASAP XML to find the bounding box (min_x, min_y, w, h) covering all polygons.
    Returns None if no coordinates found.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        all_x = []
        all_y = []
        
        for coord in root.findall(".//Coordinate"):
            all_x.append(float(coord.get("X")))
            all_y.append(float(coord.get("Y")))
            
        if not all_x:
            return None
            
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Round to integers for cropping
        x = int(math.floor(min_x))
        y = int(math.floor(min_y))
        w = int(math.ceil(max_x - min_x))
        h = int(math.ceil(max_y - min_y))
        
        return (x, y, w, h)
    except Exception as e:
        print(f"[WARN] Failed to parse annotation XML: {e}")
        return None
    
# -------------------------------
# Orchestration
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="LazySlide MRXS end‑to‑end pipeline")
    ap.add_argument("input_dir", type=str, help="Directory containing .mrxs slides")
    ap.add_argument("output_dir", type=str, help="Output directory")
    ap.add_argument("--suffixes", nargs="*", default=[".mrxs"], help="Slide suffixes to include")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    #Set Huggingface cache dir
    os.environ["HF_HOME"] = str(out_dir / "huggingface_cache")
    #Set regular cache dir
    os.environ["XDG_CACHE_DIR"] = str(out_dir / "cache")
    
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["XDG_CACHE_DIR"], exist_ok=True)

    p = Params()

    slides = list_slides(in_dir, tuple(args.suffixes))
    if not slides:
        print(f"No slides with suffix {args.suffixes} under {in_dir}")
        sys.exit(1)

    print(f"Found {len(slides)} slide(s). Processing…")

    manifest = []
    per_slide_comp: Dict[str, pd.DataFrame] = {}
    per_slide_means: Dict[str, pd.DataFrame] = {}
    titan_vectors: Dict[str, np.ndarray] = {}

    for sp in slides:
        print(f"\n— Processing {sp.name}")
        try:
            info = process_slide(sp, out_dir, p)
        except Exception as e:
            print(f"[ERROR] Processing failed for {sp.name}: {e}")
            continue
        manifest.append(info)

        slide_key = sp.stem
        # Load outputs for cohort matrices
        comp = pd.read_csv(info["celltype_comp_path"]) if info.get("celltype_comp_path") and os.path.exists(info["celltype_comp_path"]) else pd.DataFrame(columns=["cell_type", "fraction"])
        means = pd.read_csv(info["celltype_means_path"]) if info.get("celltype_means_path") and os.path.exists(info["celltype_means_path"]) else pd.DataFrame()
        per_slide_comp[slide_key] = comp
        per_slide_means[slide_key] = means
        tit_path = info.get("titan_vector_path")
        if tit_path and os.path.exists(tit_path):
            try:
                titan_vectors[slide_key] = np.load(tit_path)
            except Exception as e:
                print(f"[ERROR] Loading TITAN vector failed for {slide_key}: {e}")
                continue
    # Save manifest
    pd.DataFrame(manifest).to_json(out_dir / "manifest.json", orient="records", indent=2)

    # ------------------
    # Build matrices
    # ------------------
    print("\nBuilding slide‑level matrices…")
    comp_mat, comp_cols = build_matrix_from_compositions(per_slide_comp)
    comp_mat.to_csv(out_dir / "matrix_celltype_composition.csv")

    means_mat, means_cols = build_matrix_from_celltype_means(per_slide_means)
    means_mat.to_csv(out_dir / "matrix_celltype_nuclei_means.csv")

    if titan_vectors:
        # Align order to comp_mat index
        common_slides = sorted(set(comp_mat.index) | set(means_mat.index) | set(titan_vectors.keys()))
        titan_dim = len(next(iter(titan_vectors.values())))
        titan_mat = np.zeros((len(common_slides), titan_dim), dtype=np.float32)
        for i, s in enumerate(common_slides):
            if s in titan_vectors:
                titan_mat[i] = titan_vectors[s]
        titan_df = pd.DataFrame(titan_mat, index=common_slides)
        titan_df.to_csv(out_dir / "matrix_titan.csv")
    else:
        titan_df = pd.DataFrame(index=comp_mat.index)

    # Harmonize slide order
    slides_all = sorted(set(comp_mat.index) | set(means_mat.index) | set(titan_df.index))
    comp_mat = comp_mat.reindex(slides_all).fillna(0.0)
    means_mat = means_mat.reindex(slides_all).fillna(0.0)
    titan_df = titan_df.reindex(slides_all).fillna(0.0)

    # Combined vector
    combined_df = pd.concat([titan_df, comp_mat, means_mat], axis=1)
    combined_df.to_csv(out_dir / "matrix_combined.csv")

    # ------------------
    # Clustering + embeddings
    # ------------------
    print("Running clustering & embeddings…")
    emb_dir = out_dir / "embeddings"

    results = {}
    if titan_df.shape[1] > 0:
        results["titan"] = run_embeddings_and_clustering(titan_df.values, "titan", emb_dir, p)
    results["composition"] = run_embeddings_and_clustering(comp_mat.values, "composition", emb_dir, p)
    results["nuclei_means"] = run_embeddings_and_clustering(means_mat.values, "nuclei_means", emb_dir, p)
    results["combined"] = run_embeddings_and_clustering(combined_df.values, "combined", emb_dir, p)

    # Save quick distribution plots per slide
    print("Plotting per‑slide distributions…")
    dist_dir = out_dir / "distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # For cohort‑level melanin distribution
    melanin_all = []  # list of (slide, value)

    for sp in slides:
        slide_key = sp.stem
        slide_folder = out_dir / sp.stem
        cf_path = slide_folder / "cells_features.csv"
        if not cf_path.exists():
            continue
        dfc = pd.read_csv(cf_path)

        # Per‑slide cell‑type composition
        plt.figure(figsize=(6, 4))
        if "cell_type" in dfc.columns and not dfc["cell_type"].isna().all():
            dfc["cell_type"].value_counts(normalize=True).plot(kind="bar")
            plt.title(f"Cell‑type distribution — {slide_key}")
            plt.ylabel("fraction")
            plt.tight_layout()
            plt.savefig(dist_dir / f"{slide_key}__celltype_distribution.png", dpi=160)
        plt.close()

        # Per‑slide melanin histogram + accumulate for cohort plot
        if "melanin_fraction" in dfc.columns:
            plt.figure(figsize=(6, 4))
            dfc["melanin_fraction"].plot(kind="hist", bins=30)
            plt.title(f"Melanin fraction — {slide_key}")
            plt.xlabel("fraction")
            plt.tight_layout()
            plt.savefig(dist_dir / f"{slide_key}__melanin_hist.png", dpi=160)
            plt.close()

            # accumulate
            melanin_all.extend([(slide_key, float(v)) for v in dfc["melanin_fraction"].dropna().values])

    # ----- NEW: Cohort‑level melanin distribution -----
    if melanin_all:
        dfm = pd.DataFrame(melanin_all, columns=["slide", "melanin_fraction"]) 
        dfm.to_csv(dist_dir / "melanin_all_slides.csv", index=False)

        # Overall distribution across all slides
        plt.figure(figsize=(7, 5))
        dfm["melanin_fraction"].plot(kind="hist", bins=40)
        plt.title("Melanin fraction distribution — ALL SLIDES")
        plt.xlabel("fraction")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(dist_dir / "melanin_all_slides_hist.png", dpi=180)
        plt.close()

    print("Done. Artifacts saved to:", out_dir)


if __name__ == "__main__":
    main()
