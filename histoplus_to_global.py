"""
Convert HistoPLUS cell masks to GeoJSON format.

This corrected version properly handles the coordinate transformation from
tile-local coordinates to slide-level coordinates.
"""

from openslide.deepzoom import DeepZoomGenerator
import json
import random
from typing import Dict, Iterable, Optional


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert hex color to RGB tuple.
    
    Parameters
    ----------
    hex_color : str
        Hex color string (e.g., "#ff5733" or "ff5733")
    
    Returns
    -------
    tuple[int, int, int]
        RGB values as integers (0-255)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def generate_he_appropriate_color() -> tuple[str, tuple[int, int, int]]:
    """
    Generate a random hex color appropriate for H&E visualization.
    
    Uses colors that contrast well with the typical pink/purple H&E background:
    - Blues, greens, reds, oranges, yellows, magentas
    - Avoids light pinks/purples that would blend with H&E staining
    
    Returns
    -------
    tuple[str, tuple[int, int, int]]
        Hex color string and RGB tuple (e.g., ("#ff5733", (255, 87, 51)))
    """
    # Color palette appropriate for H&E visualization
    # Avoiding light pinks/purples that blend with H&E
    he_colors = [
        # Blues
        "#0066cc", "#3399ff", "#0080ff", "#1e90ff", "#4169e1", "#0000cd",
        # Greens  
        "#00cc66", "#33ff99", "#00ff80", "#32cd32", "#228b22", "#008000",
        # Reds (darker to contrast with eosin)
        "#cc0000", "#ff3333", "#dc143c", "#b22222", "#8b0000", "#ff0000",
        # Oranges
        "#ff6600", "#ff8c00", "#ff7f50", "#ff4500", "#d2691e", "#cd853f",
        # Yellows
        "#ffd700", "#ffff00", "#ffa500", "#daa520", "#b8860b", "#f0e68c",
        # Magentas/Purples (darker ones)
        "#cc00cc", "#ff00ff", "#8b008b", "#9932cc", "#4b0082", "#6a5acd",
        # Cyans
        "#00cccc", "#00ffff", "#20b2aa", "#48d1cc", "#40e0d0", "#00ced1",
        # Additional contrasting colors
        "#ff1493", "#ff69b4", "#7fff00", "#adff2f", "#98fb98", "#87ceeb"
    ]
    
    hex_color = random.choice(he_colors)
    rgb_color = hex_to_rgb(hex_color)
    return hex_color, rgb_color


def _deepzoom_tile_origin_and_scale(
    dz: DeepZoomGenerator,
    tile_col: int,
    tile_row: int,
    dz_level: int,
) -> Optional[tuple[float, float, float]]:
    """
    Resolve DeepZoom tile origin and scale factor to level-0 coordinates.

    Returns
    -------
    Optional[tuple[float, float, float]]
        (tile_l0_x, tile_l0_y, level_scale) or None if the tile lookup fails.
    """
    try:
        (tile_l0_x, tile_l0_y), _, _ = dz.get_tile_coordinates(dz_level, (tile_col, tile_row))
    except Exception as exc:
        print(
            f"Warning: Could not get tile coordinates for tile ({tile_col}, {tile_row}) "
            f"at level {dz_level}: {exc}"
        )
        return None

    # At level 0, each pixel is native resolution; higher levels downsample.
    level_scale = 2 ** (dz.level_count - 1 - dz_level)
    return float(tile_l0_x), float(tile_l0_y), float(level_scale)


def _iter_global_centroids(
    cell_masks: list[dict],
    slide,
    *,
    tile_size: Optional[int] = None,
    overlap: int = 0,
    apply_bounds_offset: bool = False,
) -> Iterable[tuple[dict, float, float]]:
    """
    Yield (mask, global_x, global_y) for each valid centroid in a HistoPLUS mask payload.

    Parameters
    ----------
    cell_masks : list[dict]
        Cell mask data from HistoPLUS JSON output.
    slide : OpenSlide
        The slide object.
    tile_size : int, optional
        Tile size (defaults to first item's width if not provided).
    overlap : int
        Overlap used during inference (default 0).
    apply_bounds_offset : bool
        Whether to subtract openslide bounds offsets (QuPath-specific). Leave False
        to align with ASAP XML coordinates.
    """
    if not cell_masks:
        return

    if tile_size is None:
        tile_size = int(cell_masks[0].get("width", 224))

    dz = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=False)
    offset_x = int(slide.properties.get("openslide.bounds-x", 0))
    offset_y = int(slide.properties.get("openslide.bounds-y", 0))

    for item in cell_masks:
        tile_col = int(item.get("x", 0))
        tile_row = int(item.get("y", 0))
        dz_level = int(item.get("level", 0))

        tile_info = _deepzoom_tile_origin_and_scale(dz, tile_col, tile_row, dz_level)
        if tile_info is None:
            continue
        tile_l0_x, tile_l0_y, level_scale = tile_info

        for mask in item.get("masks", []):
            centroid = mask.get("centroid")
            if not centroid or len(centroid) < 2:
                continue

            local_x, local_y = float(centroid[0]), float(centroid[1])
            global_x = tile_l0_x + local_x * level_scale
            global_y = tile_l0_y + local_y * level_scale
            if apply_bounds_offset:
                global_x -= offset_x
                global_y -= offset_y
            yield mask, global_x, global_y


def cell_masks_to_geojson(
    cell_masks: list[dict],
    slide,
    *,
    tile_size: Optional[int] = None,
    overlap: int = 0,
    inference_mpp: float = 0.25,
    seed: Optional[int] = None,
    apply_bounds_offset: bool = True,
) -> dict:
    """
    Convert HistoPLUS cell masks to GeoJSON Points with cell type colors.

    1. The centroids in the JSON are in TILE-LOCAL coordinates at the DeepZoom level
    2. The tile coordinates (x, y) are DeepZoom tile addresses
    3. The level is the DeepZoom level used during inference
    4. We transform from tile-local coords to slide-level coords

    Parameters
    ----------
    cell_masks : list[dict]
        Cell mask data from HistoPLUS JSON output.
    slide : OpenSlide
        The slide object.
    tile_size : int, optional
        Tile size (defaults to first item's width if not provided).
    overlap : int
        Overlap used during inference (default 0).
    inference_mpp : float
        MPP used during inference (should match the inference_mpp from JSON).
    seed : int, optional
        Random seed for reproducible color assignment.
    apply_bounds_offset : bool
        Whether to subtract openslide bounds offsets (QuPath-specific). Leave False
        to align with ASAP XML coordinates.

    Returns
    -------
    dict
        GeoJSON FeatureCollection with colored cell type features.
    """
    if not cell_masks:
        return {"type": "FeatureCollection", "features": []}

    if tile_size is None:
        tile_size = int(cell_masks[0].get("width", 224))

    if seed is not None:
        random.seed(seed)

    cell_type_colors: Dict[str, tuple[str, tuple[int, int, int]]] = {}
    features = []

    for mask, global_x, global_y in _iter_global_centroids(
        cell_masks,
        slide,
        tile_size=tile_size,
        overlap=overlap,
        apply_bounds_offset=apply_bounds_offset,
    ):
        cell_type = mask.get("cell_type", "Unknown")
        if cell_type not in cell_type_colors:
            cell_type_colors[cell_type] = generate_he_appropriate_color()

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [global_x, global_y],
                },
                "properties": {
                    "name": cell_type,
                    "cell_type": cell_type,
                    "confidence": mask.get("confidence"),
                    "cell_id": mask.get("cell_id"),
                },
            }
        )

    geojson_result = {
        "type": "FeatureCollection",
        "features": features,
    }

    return geojson_result


def cell_masks_to_global_detections(
    cell_masks: list[dict],
    slide,
    *,
    tile_size: Optional[int] = None,
    overlap: int = 0,
    apply_bounds_offset: bool = False,
) -> list[dict]:
    """
    Convert HistoPLUS cell masks to a list of detections in global coordinates.

    This output is intentionally aligned with ASAP XML (level-0) coordinates so downstream
    code can compare HistoPLUS detections to annotation polygons directly.
    """
    detections: list[dict] = []
    for mask, global_x, global_y in _iter_global_centroids(
        cell_masks,
        slide,
        tile_size=tile_size,
        overlap=overlap,
        apply_bounds_offset=apply_bounds_offset,
    ):
        detections.append(
            {
                "centroid": [global_x, global_y],
                "cell_type": mask.get("cell_type", "Unknown"),
                "confidence": mask.get("confidence"),
                "cell_id": mask.get("cell_id"),
            }
        )
    return detections


def load_and_convert_histoplus_json(json_path: str, slide_path: str, output_path: str):
    """
    Load HistoPLUS JSON output and convert to GeoJSON.
    
    Parameters
    ----------
    json_path : str
        Path to the HistoPLUS JSON output file
    slide_path : str
        Path to the slide file
    output_path : str
        Path to save the GeoJSON output
    """
    import openslide
    
    # Load the HistoPLUS JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Open the slide
    slide = openslide.open_slide(slide_path)
    
    # Extract parameters from the JSON
    cell_masks = data.get("cell_masks", [])
    inference_mpp = data.get("inference_mpp", 0.25)
    
    # Convert to GeoJSON
    geojson = cell_masks_to_geojson(
        cell_masks=cell_masks,
        slide=slide,
        inference_mpp=inference_mpp,
        seed=42,  # For reproducible colors
        apply_bounds_offset=True,
    )
    
    # Save the result
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    slide.close()
    # Print summary
    print(f"Converted {len(geojson['features'])} cells to GeoJSON")
    print(f"Output saved to: {output_path}")


def save_histoplus_global_json(json_path: str, slide_path: str, output_path: str) -> None:
    """
    Convert HistoPLUS JSON output to global-coordinate detections and save to disk.

    This preserves slide-level (level-0) coordinates so detections line up with ASAP XML
    annotations.
    """
    import openslide

    with open(json_path, "r") as handle:
        data = json.load(handle)

    slide = openslide.open_slide(slide_path)
    cell_masks = data.get("cell_masks", [])
    detections = cell_masks_to_global_detections(cell_masks=cell_masks, slide=slide)
    slide.close()

    with open(output_path, "w") as handle:
        json.dump(detections, handle, indent=2)

    print(f"Converted {len(detections)} cells to global detections JSON")
    print(f"Output saved to: {output_path}")
    
if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python cell_masks_to_geojson_corrected.py <histoplus_json> <slide_file> <output_geojson>")
        sys.exit(1)
    
    json_path, slide_path, output_path = sys.argv[1:4]
    load_and_convert_histoplus_json(json_path, slide_path, output_path)
