"""Roof Age instance link on gridded AOIs: link by the outline the API would return.

Gridding disables parcelMode, so the stored roof geometry is the whole roof while
the Roof Age instances follow the API's parcel rule: an instance is the roof
within its parcel when the API clips the roof there, the whole roof otherwise.
Linking the whole outline understates the IoU of a roof crossing the parcel
boundary and, when most of the roof lies next door, matches the neighbouring
parcel's instance. ``process_chunk`` therefore clips the roof outlines of gridded
AOIs with ``_clip_roof_geoms_to_aoi`` before ``link_roof_instances_to_roofs``,
exactly as it does for the component recalculation, and that clip applies the
API's rule (``geometry_utils.should_clip_roof_to_parcel``): a roof whose pieces
inside and outside the parcel are both building-like (>= 30 m², >= 3 m wide) is
clipped; a slight overhang stays whole. A neighbour's roof that barely reaches
in, which parcelMode would not assign to the parcel at all, is clipped to its
in-parcel piece, since a gridded AOI keeps it. Non-gridded parcelMode AOIs are
untouched: their multiparcel roofs arrive pre-clipped (``clippedGeometry``).
"""

import math

import geopandas as gpd
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from nmaipy import parcels
from nmaipy.constants import API_CRS, AREA_CRS
from nmaipy.exporter import _clip_roof_geoms_to_aoi, _gridded_aoi_geometries
from nmaipy.geometry_utils import is_building_like, should_clip_roof_to_parcel

# Rectangles in metres east and north of a point in Delaware, as WGS84 boxes.
_LON0, _LAT0 = -75.5, 39.1
_M_PER_DEG_LAT = 111320.0
_M_PER_DEG_LON = _M_PER_DEG_LAT * math.cos(math.radians(_LAT0))
_COUNTRY = "us"


def _metres_box(x0, y0, w, h):
    return box(
        _LON0 + x0 / _M_PER_DEG_LON,
        _LAT0 + y0 / _M_PER_DEG_LAT,
        _LON0 + (x0 + w) / _M_PER_DEG_LON,
        _LAT0 + (y0 + h) / _M_PER_DEG_LAT,
    )


_PARCEL = _metres_box(0, 0, 100, 100)
_ROOF_HALF_NEXT_DOOR = _metres_box(90, 10, 20, 10)  # 100 m² inside, 100 m² outside: the API clips
_ROOF_SLIGHT_OVERHANG = _metres_box(81, 10, 20, 10)  # 1 m overhang, 10 m² outside: the API leaves it whole
_ROOF_NEIGHBOURS = _metres_box(98, 10, 20, 10)  # 2 m in, 20 m² inside: the API does not assign it to the parcel
# A 20 m × 20 m roof inside the parcel with an L-shaped strip of the given width
# hooked round the parcel corner outside it: 2 m is narrow however long the legs
# are, 4 m is wide, while a bounding rectangle would call both wide.
_ROOF_L_STRIP_2M = unary_union([_metres_box(80, 80, 20, 20), _metres_box(100, 80, 2, 22), _metres_box(80, 100, 22, 2)])
_ROOF_L_STRIP_4M = unary_union([_metres_box(80, 80, 20, 20), _metres_box(100, 80, 4, 24), _metres_box(80, 100, 24, 4)])


def _metric(*geoms):
    return gpd.GeoSeries(list(geoms), crs=API_CRS).to_crs(AREA_CRS[_COUNTRY]).values


def _roofs_gdf(roof):
    return gpd.GeoDataFrame(
        [{"feature_id": "roof-1", "aoi_id": "parcel-a", "geometry": roof}],
        geometry="geometry",
        crs=API_CRS,
    ).set_index("aoi_id")


def _instances_gdf(roof):
    """This parcel's instance and the neighbour's, as the API scopes them for ``roof``."""
    ours = roof.intersection(_PARCEL)
    neighbour = roof.difference(_PARCEL)
    return gpd.GeoDataFrame(
        [
            {"feature_id": "ours", "aoi_id": "parcel-a", "kind": "roof", "geometry": ours},
            {"feature_id": "neighbour", "aoi_id": "parcel-a", "kind": "roof", "geometry": neighbour},
        ],
        geometry="geometry",
        crs=API_CRS,
    ).set_index("aoi_id")


def _aoi_geometries():
    aoi_gdf = gpd.GeoDataFrame([{"aoi_id": "parcel-a", "geometry": _PARCEL}], geometry="geometry", crs=API_CRS)
    return _gridded_aoi_geometries(aoi_gdf, {"parcel-a"})


def _clip(roofs):
    return _clip_roof_geoms_to_aoi(roofs, _aoi_geometries(), _COUNTRY)


def _link(roofs, instances):
    link_roofs = roofs.copy()
    link_roofs["geometry"] = _clip(roofs)
    return parcels.link_roof_instances_to_roofs(instances, link_roofs)


def test_whole_outline_links_the_neighbours_instance():
    """The bug: linking the whole roof outline of a roof half next door picks the neighbour's instance (IoU 0.6 against 0.4)."""
    roof = _metres_box(92, 10, 20, 10)  # 60 % next door
    _, roofs_linked = parcels.link_roof_instances_to_roofs(_instances_gdf(roof), _roofs_gdf(roof))
    assert roofs_linked.loc["parcel-a", "primary_child_roof_age_feature_id"] == "neighbour"
    assert roofs_linked.loc["parcel-a", "primary_child_roof_age_iou"] == pytest.approx(0.6, abs=1e-3)


def test_parcel_clipped_outline_links_the_parcels_instance():
    """Clipped to the AOI as process_chunk does for gridded AOIs, the roof links to this parcel's instance."""
    roof = _metres_box(92, 10, 20, 10)  # 60 % next door
    ri_linked, roofs_linked = _link(_roofs_gdf(roof), _instances_gdf(roof))

    assert roofs_linked.loc["parcel-a", "primary_child_roof_age_feature_id"] == "ours"
    assert roofs_linked.loc["parcel-a", "primary_child_roof_age_iou"] == pytest.approx(1.0, abs=1e-6)
    by_id = ri_linked.set_index("feature_id")
    assert by_id.loc["ours", "parent_id"] == "roof-1"
    assert by_id.loc["neighbour", "parent_id"] is None


def test_slight_overhang_keeps_the_whole_outline():
    """The API leaves a roof with under 30 m² outside the parcel whole, and so is its instance: clipping would give IoU 0.95."""
    roofs = _roofs_gdf(_ROOF_SLIGHT_OVERHANG)
    assert _clip(roofs).iloc[0].equals(_ROOF_SLIGHT_OVERHANG)

    instances = gpd.GeoDataFrame(
        [{"feature_id": "whole", "aoi_id": "parcel-a", "kind": "roof", "geometry": _ROOF_SLIGHT_OVERHANG}],
        geometry="geometry",
        crs=API_CRS,
    ).set_index("aoi_id")
    _, roofs_linked = _link(roofs, instances)
    assert roofs_linked.loc["parcel-a", "primary_child_roof_age_feature_id"] == "whole"
    assert roofs_linked.loc["parcel-a", "primary_child_roof_age_iou"] == pytest.approx(1.0, abs=1e-6)


def test_neighbours_roof_is_clipped_to_its_piece_in_the_parcel():
    """A roof the API would not assign to the parcel is kept on a gridded AOI: analyse its in-parcel piece, not the whole roof."""
    clipped = _clip(_roofs_gdf(_ROOF_NEIGHBOURS)).iloc[0]
    assert clipped.equals(_ROOF_NEIGHBOURS.intersection(_PARCEL))


@pytest.mark.parametrize(
    "name, geom, want",
    [
        ("20 m × 20 m", _metres_box(0, 0, 20, 20), True),
        ("just over 3 m wide, 12 m long", _metres_box(0, 0, 3.1, 12), True),
        ("2 m wide strip, 80 m²", _metres_box(0, 0, 2, 40), False),
        ("L-shaped 2 m strip, 88 m²", _ROOF_L_STRIP_2M.difference(_metres_box(80, 80, 20, 20)), False),
        ("L-shaped 4 m strip", _ROOF_L_STRIP_4M.difference(_metres_box(80, 80, 20, 20)), True),
        ("4 m × 4 m, under 30 m²", _metres_box(0, 0, 4, 4), False),
        ("two 4 m × 4 m blobs, 32 m² together", unary_union([_metres_box(0, 0, 4, 4), _metres_box(10, 0, 4, 4)]), True),
    ],
)
def test_is_building_like(name, geom, want):
    assert is_building_like(_metric(geom)[0]) is want, name


@pytest.mark.parametrize(
    "name, roof, want",
    [
        ("inside the parcel", _metres_box(10, 10, 20, 10), False),
        ("half next door", _ROOF_HALF_NEXT_DOOR, True),
        ("1 m overhang, 10 m² outside", _ROOF_SLIGHT_OVERHANG, False),
        ("2 m wide strip outside, 80 m² but narrow", _metres_box(62, 10, 40, 40), False),
        ("4 m wide strip outside", _metres_box(64, 10, 40, 40), True),
        ("L-shaped 2 m strip outside, narrow round the corner", _ROOF_L_STRIP_2M, False),
        ("L-shaped 4 m strip outside", _ROOF_L_STRIP_4M, True),
        # Not building-like itself, so parcelMode would not assign it either; a gridded AOI keeps it, so clip it.
        ("20 m² shed crossing the boundary", _metres_box(98, 10, 5, 4), True),
        ("5x5 m shed with 5 m² inside", _metres_box(99, 10, 5, 5), True),
        ("20 m² shed wholly inside the parcel", _metres_box(10, 10, 5, 4), False),
        ("roof wholly outside the parcel", _metres_box(200, 10, 20, 10), False),
        # parcelMode would not assign this roof to the parcel; a gridded AOI keeps it, so clip it.
        ("neighbour's roof reaching 2 m in, 20 m² inside", _ROOF_NEIGHBOURS, True),
    ],
)
def test_should_clip_roof_to_parcel_follows_the_api_rule(name, roof, want):
    roof_m, parcel_m = _metric(roof, _PARCEL)
    assert should_clip_roof_to_parcel(roof_m, parcel_m) is want, name


def test_gridded_aoi_geometries_is_restricted_to_gridded_aois():
    aoi_gdf = gpd.GeoDataFrame(
        [{"aoi_id": "parcel-a", "geometry": _PARCEL}, {"aoi_id": "parcel-b", "geometry": _PARCEL}],
        geometry="geometry",
        crs=API_CRS,
    )
    assert set(_gridded_aoi_geometries(aoi_gdf, {"parcel-a"})) == {"parcel-a"}
    assert _gridded_aoi_geometries(aoi_gdf, set()) is None
    assert _gridded_aoi_geometries(aoi_gdf.drop(columns="geometry"), {"parcel-a"}) is None
