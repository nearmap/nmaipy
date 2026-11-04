"""Live checks of the Feature API response semantics that nmaipy's flattening and grid merge rely on.

A bundled NJ parcel is split in two through its main roof, so each half is a synthetic parcel that
shares the roof (a multiparcel feature in both). The survey is pinned by date so the expected
geometry does not move when newer surveys land. See "parcelMode response semantics" in CLAUDE.md.
"""

import copy

import geopandas as gpd
import pytest
from shapely.geometry import box, shape

from nmaipy.constants import API_CRS, AREA_CRS, CONNECTED_CLASS_IDS, ROOF_ID, TREE_OVERHANG_ID
from nmaipy.feature_api import FeatureApi
from nmaipy.feature_attributes import flatten_roof_attributes

PARCEL_ID = "1f5ea3af-ac86-5b9e-b206-94ac79683840"
SURVEY_DATE = "2026-10-08"
COUNTRY = "us"
PACKS = ["building", "roof_char", "vegetation"]
INCLUDE = ["hurricaneScore"]


def _area_sqm(geom) -> float:
    return gpd.GeoSeries([geom], crs=API_CRS).to_crs(AREA_CRS[COUNTRY]).iloc[0].area


def _tree_overhang_component(feature: dict) -> dict:
    for attribute in feature.get("attributes") or []:
        for component in attribute.get("components") or []:
            if component["classId"] == TREE_OVERHANG_ID:
                return component
    raise AssertionError(f"Roof {feature['id']} has no Tree Overhang component")


def _hurricane_tree_overhang_ratio(feature: dict) -> float:
    return feature["hurricaneScore"]["modelInputFeatures"]["treeOverhangRatio"]


@pytest.fixture(scope="module")
def split(parcels_2_gdf, tmp_path_factory):
    """Raw parcelMode payloads for the parcel and its two halves, plus the shared roof's id."""
    api = FeatureApi(cache_dir=tmp_path_factory.mktemp("parcel_mode_cache"), parcel_mode=True)
    parcel = parcels_2_gdf.loc[PARCEL_ID].geometry

    def fetch(geom):
        return api.get_features(geom, COUNTRY, packs=PACKS, include=INCLUDE, since=SURVEY_DATE, until=SURVEY_DATE)

    whole = fetch(parcel)
    roofs = [f for f in whole["features"] if f["classId"] == ROOF_ID and f.get("belongsToParcel", True)]
    assert roofs, f"Parcel {PARCEL_ID} returned no roof for survey {SURVEY_DATE}"
    roof = max(roofs, key=lambda f: f["unclippedAreaSqm"])
    cx = shape(roof["geometry"]).centroid.x
    minx, miny, maxx, maxy = parcel.bounds
    halves = {
        "W": parcel.intersection(box(minx - 1, miny - 1, cx, maxy + 1)),
        "E": parcel.intersection(box(cx, miny - 1, maxx + 1, maxy + 1)),
    }
    payloads = {side: fetch(geom) for side, geom in halves.items()}
    return dict(roof_id=roof["id"], halves=halves, payloads=payloads)


def _roof(split, side) -> dict:
    matches = [f for f in split["payloads"][side]["features"] if f["id"] == split["roof_id"]]
    assert len(matches) == 1, f"Roof {split['roof_id']} should be returned once in half {side}"
    return matches[0]


@pytest.mark.live_api
def test_shared_roof_is_multiparcel_in_both_halves(split):
    for side in ("W", "E"):
        roof = _roof(split, side)
        assert roof["belongsToParcel"], f"Half {side}: the split roof should belong to the parcel"
        assert roof.get("clippedGeometry"), f"Half {side}: a roof split down the middle should come back clipped"
        assert roof["areaSqm"] == roof["unclippedAreaSqm"], "A discrete feature's areaSqm is its whole area"
        assert roof["clippedAreaSqm"] < 0.75 * roof["unclippedAreaSqm"], f"Half {side} should hold part of the roof"


@pytest.mark.live_api
def test_roof_attributes_are_whole_feature_in_every_parcel(split):
    """parcelMode clips geometry and area only: component areas and ratios stay whole-roof."""
    west, east = _roof(split, "W"), _roof(split, "E")
    assert west["attributes"] == east["attributes"], "Roof attributes should be identical in both halves"
    component = _tree_overhang_component(west)
    assert component["areaSqm"] > 0, "Fixture roof should carry some tree overhang"
    assert component["ratio"] == pytest.approx(
        component["areaSqm"] / west["unclippedAreaSqm"], abs=0.005
    ), "Component ratio should be relative to the whole roof, not the parcel's clip"


@pytest.mark.live_api
def test_includes_are_computed_per_parcel(split):
    west, east = _roof(split, "W"), _roof(split, "E")
    assert _hurricane_tree_overhang_ratio(west) != pytest.approx(
        _hurricane_tree_overhang_ratio(east), abs=0.01
    ), "The hurricane include's tree overhang ratio should be computed on each half's clip"


@pytest.mark.live_api
def test_children_returned_whole_whether_or_not_inside_parcel(split):
    """Every child of a returned roof comes back with its full geometry; parcelMode only flags the outside ones."""
    children = {
        side: {f["id"]: f for f in split["payloads"][side]["features"] if f.get("parentId") == split["roof_id"]}
        for side in ("W", "E")
    }
    assert children["W"].keys() == children["E"].keys(), "Both halves should receive the roof's full child set"
    n_outside = 0
    for side, kids in children.items():
        half = split["halves"][side]
        for fid, child in kids.items():
            assert child["geometry"] == children["W"][fid]["geometry"], "Child geometry should be the full outline"
            if shape(child["geometry"]).intersection(half).is_empty:
                n_outside += 1
                assert child["belongsToParcel"] is False, f"Child {fid} wholly outside half {side} should not belong"
                assert child["clippedAreaSqm"] == 0, f"Child {fid} wholly outside half {side} has no in-parcel area"
    assert n_outside > 0, "Fixture should have at least one roof child wholly outside one of the halves"


@pytest.mark.live_api
def test_nmaipy_tree_overhang_ratio_is_the_per_parcel_value(split):
    """nmaipy recomputes components on the clipped roof; that matches the API's own per-parcel value."""
    for side in ("W", "E"):
        gdf, _ = FeatureApi.payload_gdf(copy.deepcopy(split["payloads"][side]), aoi_id=side, parcel_mode=True)
        gdf = gdf.reset_index()
        roof = gdf[gdf["feature_id"] == split["roof_id"]].iloc[0]
        flattened = flatten_roof_attributes(
            [roof.to_dict()], country=COUNTRY, child_features=gdf[gdf["class_id"] != ROOF_ID]
        )
        raw_roof = _roof(split, side)
        expected = _hurricane_tree_overhang_ratio(raw_roof)
        assert flattened["tree_overhang_ratio"] == pytest.approx(expected, abs=0.001), (
            f"Half {side}: nmaipy tree_overhang_ratio {flattened['tree_overhang_ratio']} should equal the API's "
            f"per-parcel treeOverhangRatio {expected}"
        )
        assert flattened["tree_overhang_ratio"] != pytest.approx(
            _tree_overhang_component(raw_roof)["ratio"], abs=0.01
        ), f"Half {side}: the recomputed ratio should differ from the whole-roof component ratio"


@pytest.mark.live_api
def test_area_semantics_discrete_vs_connected(split):
    """Discrete classes report their whole area; connected classes are clipped to the AOI."""
    connected = set(CONNECTED_CLASS_IDS)
    n_connected_extending_past = 0
    for side in ("W", "E"):
        half = split["halves"][side]
        for f in split["payloads"][side]["features"]:
            if f["classId"] in connected:
                assert f["areaSqm"] == f["clippedAreaSqm"], f"Connected {f['description']}: areaSqm should be in-AOI"
                assert not f.get("clippedGeometry"), "Connected features are already clipped to the AOI"
                outside = shape(f["geometry"]).difference(half)
                assert _area_sqm(outside) < max(0.5, 0.01 * f["areaSqm"]), "Connected geometry should lie in the AOI"
                n_connected_extending_past += f["unclippedAreaSqm"] > f["clippedAreaSqm"]
            else:
                assert f["areaSqm"] == f["unclippedAreaSqm"], f"Discrete {f['description']}: areaSqm should be whole"
    assert n_connected_extending_past > 0, "Fixture should include vegetation that extends past a half"
