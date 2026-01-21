from pathlib import Path

import pytest

pytest.importorskip("shapely")

from eyemelanoma.roi import roi_polygon_from_dat


def test_roi_polygon_from_dat(tmp_path: Path) -> None:
    Polygon = pytest.importorskip("shapely.geometry").Polygon
    dat_path = tmp_path / "Data0001.dat"
    xml_payload = (
        "header"
        "<Diff><SubDiff>"
        "<polygon_data><polygon_points>"
        '<polygon_point polypointX="0" polypointY="0"/>'
        '<polygon_point polypointX="10" polypointY="0"/>'
        '<polygon_point polypointX="10" polypointY="10"/>'
        '<polygon_point polypointX="0" polypointY="10"/>'
        "</polygon_points></polygon_data>"
        "</SubDiff></Diff>"
    )
    dat_path.write_bytes(xml_payload.encode("utf-8"))

    polygon = roi_polygon_from_dat(dat_path)
    assert isinstance(polygon, Polygon)
    assert polygon.area == 100
