"""Regression tests: clipped-roof component areas for GRIDDED AOIs must respect
the clipped roof, while non-gridded parcelMode analysis stays whole-roof.

Bug (gridded only): gridding disables parcelMode, so ``combine_features_from_grid``
stores the *whole* roof outline while summing ``clipped_area`` per grid cell — the
stored geometry is unclipped but ``clipped_area`` is clipped. Component areas are
recomputed by intersecting child features against the parent roof geometry
(``calculate_child_feature_attributes``), so passing the whole outline as the parent
inflated every component to whole-roof magnitude (a single component could exceed
the whole clipped roof area — physically impossible).

Fix: clip the roof parent geometry to its AOI/parcel polygon before the recalc —
but ONLY for AOIs answered by gridded queries (tracked via the ``gridded`` metadata
flag set in ``_attempt_gridding``). ``process_chunk`` builds the ``aoi_geometries``
map for gridded AOIs only; ``parcel_rollup`` receives ``gridded_aoi_ids``.

Non-gridded parcelMode AOIs must be untouched: the API already encodes intent
per-roof. Multiparcel roofs arrive pre-clipped (``clippedGeometry``); roofs that
merely poke over the parcel boundary (``multiparcel=False`` with
``clipped_area < unclipped_area`` — parcel misalignment is common) are judged to
belong to the parcel and must keep FULL-ROOF component analysis.
"""

import geopandas as gpd
import pandas as pd
import pytest

from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS, ROOF_ID
from nmaipy.exporter import (
    _batch_project_geometries,
    _clip_roof_geoms_to_aoi,
    _compute_all_per_class_data,
)
from nmaipy.feature_api import FeatureApi
from nmaipy.parcels import flatten_roof_attributes, parcel_rollup


def test_clip_roof_geoms_to_aoi_yields_clipped_parent(clipped_roof):
    """`_clip_roof_geoms_to_aoi` returns the parcel-clipped roof, not the whole outline."""
    roof_gdf = gpd.GeoDataFrame([clipped_roof.roof_row()], geometry="geometry", crs=API_CRS)
    clipped = _clip_roof_geoms_to_aoi(roof_gdf, {"parcel-a": clipped_roof.parcel})
    assert clipped_roof.proj_area_sqft(clipped.iloc[0]) == pytest.approx(clipped_roof.clipped_sqft, rel=1e-6)

    # No AOI geometry available for this AOI → original (whole) outline kept.
    unchanged = _clip_roof_geoms_to_aoi(roof_gdf, {})
    assert clipped_roof.proj_area_sqft(unchanged.iloc[0]) == pytest.approx(clipped_roof.unclipped_sqft, rel=1e-6)


def test_flatten_component_area_is_unclipped_without_aoi_clip(clipped_roof):
    """Without AOI clipping the recomputed Tile area is whole-roof.

    This is the gridded-case bug reproduction AND the intended behaviour for
    non-gridded parcelMode slight-overhang roofs (which never get a clip map)."""
    roof_gdf = gpd.GeoDataFrame([clipped_roof.roof_row()], geometry="geometry", crs=API_CRS)
    child = gpd.GeoDataFrame([clipped_roof.tile_child_row()], geometry="geometry", crs=API_CRS)
    parent_projected, child_proj = _batch_project_geometries(roof_gdf, {"parcel-a": child}, "us")

    attrs = flatten_roof_attributes(
        [clipped_roof.roof_row()],
        country="us",
        child_features=child,
        parent_projected=parent_projected.iloc[0],
        children_projected=child_proj.get("parcel-a"),
    )
    # The bug: component area ≈ whole roof, and exceeds the clipped roof area.
    assert attrs["tile_area_sqft"] == pytest.approx(clipped_roof.unclipped_sqft, rel=1e-3)
    assert attrs["tile_area_sqft"] > clipped_roof.clipped_sqft


def test_flatten_component_area_respects_clip_with_aoi_geometries(clipped_roof):
    """With AOI clipping, the recomputed Tile area is the parcel-clipped area and
    never exceeds the clipped roof area."""
    roof_gdf = gpd.GeoDataFrame([clipped_roof.roof_row()], geometry="geometry", crs=API_CRS)
    child = gpd.GeoDataFrame([clipped_roof.tile_child_row()], geometry="geometry", crs=API_CRS)
    parent_projected, child_proj = _batch_project_geometries(
        roof_gdf, {"parcel-a": child}, "us", aoi_geometries={"parcel-a": clipped_roof.parcel}
    )

    attrs = flatten_roof_attributes(
        [clipped_roof.roof_row()],
        country="us",
        child_features=child,
        parent_projected=parent_projected.iloc[0],
        children_projected=child_proj.get("parcel-a"),
    )
    # Fixed: component area equals the parcel-clipped child area (matches the API's
    # per-child clipped area) and is ≤ the clipped roof area.
    assert attrs["tile_area_sqft"] == pytest.approx(clipped_roof.clipped_sqft, rel=1e-3)
    assert attrs["tile_area_sqft"] <= clipped_roof.clipped_sqft * 1.001


def test_per_class_export_clips_components_to_parcel(clipped_roof):
    """End-to-end per-class path, gridded AOI: the roof export threads AOI geometry
    through and the exported component area does not exceed the clipped roof area.

    ``aoi_geometries`` simulates the map ``process_chunk`` builds — which contains
    entries only for AOIs answered by gridded queries.

    Uses the streaming layout (aoi_id as a column, RangeIndex) — the layout the
    real per-chunk export path produces and the one where child features are
    grouped per AOI for the recalc.
    """
    chunk = gpd.GeoDataFrame([clipped_roof.roof_row(), clipped_roof.tile_child_row()], geometry="geometry", crs=API_CRS)
    aoi_geometries = {"parcel-a": clipped_roof.parcel}

    res = _compute_all_per_class_data(
        chunk,
        country="us",
        aoi_input_columns=[],
        whitelisted_classes=[ROOF_ID],
        threads=1,
        aoi_geometries=aoi_geometries,
    )
    roof = res[ROOF_ID]["tabular"].to_pandas().iloc[0]
    assert roof["tile_area_sqft"] == pytest.approx(clipped_roof.clipped_sqft, rel=1e-3)
    assert roof["tile_area_sqft"] <= roof["clipped_area_sqft"] * 1.001


def test_parcel_rollup_primary_component_respects_clip(clipped_roof):
    """parcel_rollup path, gridded AOI: the primary roof's component area is clipped
    to the parcel (feature_attributes clips the primary roof to parcel_geom before
    the recalc, gated on the AOI being in gridded_aoi_ids)."""
    features = gpd.GeoDataFrame(
        [clipped_roof.roof_row(), clipped_roof.tile_child_row()], geometry="geometry", crs=API_CRS
    ).set_index(AOI_ID_COLUMN_NAME)
    parcels_gdf = gpd.GeoDataFrame(
        {"geometry": [clipped_roof.parcel]},
        index=pd.Index(["parcel-a"], name=AOI_ID_COLUMN_NAME),
        crs=API_CRS,
    )
    classes_df = pd.DataFrame({"id": [ROOF_ID], "description": ["roof"]}).set_index("id")

    rollup = parcel_rollup(
        parcels_gdf,
        features,
        classes_df,
        country="us",
        primary_decision="largest_intersection",
        gridded_aoi_ids={"parcel-a"},
    )
    row = rollup.loc["parcel-a"]
    assert row["primary_roof_tile_area_sqft"] == pytest.approx(clipped_roof.clipped_sqft, rel=1e-3)
    assert row["primary_roof_tile_area_sqft"] <= row["primary_roof_clipped_area_sqft"] * 1.001


def test_per_class_export_non_gridded_keeps_full_roof_analysis(clipped_roof):
    """Non-gridded AOI (no aoi_geometries entry): a slight-overhang roof keeps
    FULL-ROOF component analysis — the API judged it belongs to the parcel, and
    parcel misalignment must not shave component areas."""
    chunk = gpd.GeoDataFrame([clipped_roof.roof_row(), clipped_roof.tile_child_row()], geometry="geometry", crs=API_CRS)

    res = _compute_all_per_class_data(
        chunk,
        country="us",
        aoi_input_columns=[],
        whitelisted_classes=[ROOF_ID],
        threads=1,
        aoi_geometries=None,  # process_chunk builds no entry for non-gridded AOIs
    )
    roof = res[ROOF_ID]["tabular"].to_pandas().iloc[0]
    assert roof["tile_area_sqft"] == pytest.approx(clipped_roof.unclipped_sqft, rel=1e-3)
    assert roof["tile_area_sqft"] > clipped_roof.clipped_sqft


def test_parcel_rollup_non_gridded_keeps_full_roof_analysis(clipped_roof):
    """parcel_rollup without gridded_aoi_ids: primary roof component areas stay at
    full-roof magnitude for a slight-overhang roof (pre-existing behaviour)."""
    features = gpd.GeoDataFrame(
        [clipped_roof.roof_row(), clipped_roof.tile_child_row()], geometry="geometry", crs=API_CRS
    ).set_index(AOI_ID_COLUMN_NAME)
    parcels_gdf = gpd.GeoDataFrame(
        {"geometry": [clipped_roof.parcel]},
        index=pd.Index(["parcel-a"], name=AOI_ID_COLUMN_NAME),
        crs=API_CRS,
    )
    classes_df = pd.DataFrame({"id": [ROOF_ID], "description": ["roof"]}).set_index("id")

    rollup = parcel_rollup(parcels_gdf, features, classes_df, country="us", primary_decision="largest_intersection")
    row = rollup.loc["parcel-a"]
    assert row["primary_roof_tile_area_sqft"] == pytest.approx(clipped_roof.unclipped_sqft, rel=1e-3)
    assert row["primary_roof_tile_area_sqft"] > row["primary_roof_clipped_area_sqft"]


def test_attempt_gridding_metadata_marks_gridded(clipped_roof, monkeypatch):
    """_attempt_gridding stamps ``gridded: True`` on the AOI metadata — the flag
    process_chunk uses to build the clip map and gridded_aoi_ids set."""
    api = FeatureApi(api_key="fake")

    fake_metadata_df = pd.DataFrame(
        [
            {
                AOI_ID_COLUMN_NAME: 0,
                "system_version": "gen6-test",
                "link": "https://example.com",
                "survey_date": "2026-03-28",
                "survey_id": "survey-1",
                "survey_resource_id": "resource-1",
                "perspective": "vertical",
                "postcat": False,
                "mesh_date": "",
            }
        ]
    ).set_index(AOI_ID_COLUMN_NAME)
    fake_features_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=API_CRS)
    fake_errors_df = pd.DataFrame([])

    fake_grid_summary = {"grid_cells": 4, "grid_cells_answered": 3, "answered_area_fraction": 0.8}

    def fake_gridded(**kwargs):
        return fake_features_gdf, fake_metadata_df, fake_errors_df, fake_grid_summary

    monkeypatch.setattr(api, "get_features_gdf_gridded", fake_gridded)

    _, metadata, error, _ = api._attempt_gridding(
        geometry=clipped_roof.roof_full,
        region="us",
        aoi_id="gridded-aoi",
    )
    assert error is None
    assert metadata["gridded"] is True
    assert {k: metadata[k] for k in fake_grid_summary} == fake_grid_summary, "grid coverage must reach the metadata"
