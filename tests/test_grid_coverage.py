"""Tests for how much of a gridded AOI the Feature API answered.

A large AOI is split into grid cells (the grid overlaid on the AOI, so edge cells are clipped
slivers) and each cell is queried. ``get_features_gdf_gridded`` reports how many cells answered
and the answered cells' share of the AOI's area, applies ``aoi_grid_min_pct``, and names a failed
grid's status so an outage is never reported as a coverage gap. Each cell's response is stubbed,
keyed on its cell id, so no request leaves the process.
"""

import logging

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

import nmaipy.feature_api as feature_api_mod
from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS, AREA_CRS
from nmaipy.feature_api import AIFeatureAPIGridError, FeatureApi
from nmaipy.geometry_utils import split_geometry_into_grid

# Not grid-aligned and with a slanted edge, so edge cells are clipped to different sizes.
AOI = Polygon([(151.2003, -33.9003), (151.2161, -33.9003), (151.2131, -33.8862), (151.2003, -33.8862)])
CELL_SIZE = 0.005


def _cells():
    """The grid get_features_gdf_gridded builds for AOI, with the cell ids it assigns."""
    cells = split_geometry_into_grid(geometry=AOI, cell_size=CELL_SIZE)
    cells.index = range(len(cells))
    return cells


def _cell_metadata(cell_id):
    return {
        AOI_ID_COLUMN_NAME: cell_id,
        "system_version": "gen6-test",
        "link": "https://example.com",
        "survey_date": "2024-01-01",
        "survey_id": "s1",
        "survey_resource_id": "r1",
        "perspective": "Vert",
        "postcat": False,
        "mesh_date": "",
    }


def _cell_error(cell_id, status_code):
    return {
        AOI_ID_COLUMN_NAME: cell_id,
        "status_code": status_code,
        "message": "no survey resources found" if status_code == 404 else "boom",
        "text": "",
        "request": "r",
        "failure_type": "standard",
    }


def _api(tmp_path, monkeypatch, failures, aoi_grid_min_pct):
    """A FeatureApi whose per-cell query answers every cell except those in ``failures`` ({cell id: status})."""
    api = FeatureApi(api_key="test-key", cache_dir=tmp_path, aoi_grid_min_pct=aoi_grid_min_pct, grid_size=CELL_SIZE)
    calls = []

    def stub_get_features_gdf(geometry, region, aoi_id, **kwargs):
        calls.append(aoi_id)
        if aoi_id in failures:
            return None, None, _cell_error(aoi_id, failures[aoi_id]), None
        return None, _cell_metadata(aoi_id), None, None

    monkeypatch.setattr(api, "get_features_gdf", stub_get_features_gdf)
    api.cell_calls = calls
    return api


def _area_share(cells, cell_ids):
    """Independent reference: the given cells' area over the AOI's area, in the region's area CRS."""
    crs = AREA_CRS["au"]
    answered = gpd.GeoSeries([cells.geometry[cells.index.isin(cell_ids)].union_all()], crs=API_CRS).to_crs(crs)
    return answered.area.iloc[0] / gpd.GeoSeries([AOI], crs=API_CRS).to_crs(crs).area.iloc[0]


def _gridded(api):
    return api.get_features_gdf_gridded(geometry=AOI, region="au", aoi_id="big", grid_size=CELL_SIZE)


class TestPartialGrid:
    def test_reports_cells_and_area_answered(self, tmp_path, monkeypatch, caplog):
        cells = _cells()
        n = len(cells)
        assert n > 6, f"the AOI should grid into several cells, got {n}"
        # Fail the two smallest (edge) cells, so the area share is well above the cell share.
        smallest = cells.to_crs(AREA_CRS["au"]).area.nsmallest(2).index.tolist()
        api = _api(tmp_path, monkeypatch, {cid: 404 for cid in smallest}, aoi_grid_min_pct=50)

        with caplog.at_level(logging.DEBUG, logger="nmaipy"):
            _, metadata_df, errors_df, summary = _gridded(api)

        assert summary["grid_cells"] == n and summary["grid_cells_answered"] == n - 2
        expected = _area_share(cells, [c for c in cells.index if c not in smallest])
        assert summary["answered_area_fraction"] == pytest.approx(expected, rel=1e-6)
        assert summary["answered_area_fraction"] > (n - 2) / n, "edge slivers must weigh less than whole cells"
        assert len(metadata_df) == n - 2 and len(errors_df) == 2
        assert f"{n - 2} of {n} grid cells answered" in caplog.text, "the log must count cells, not features"

    def test_attempt_gridding_puts_coverage_on_the_metadata(self, tmp_path, monkeypatch):
        api = _api(tmp_path, monkeypatch, {0: 404}, aoi_grid_min_pct=50)
        _, metadata, error, grid_errors = api._attempt_gridding(geometry=AOI, region="au", aoi_id="big")
        assert error is None
        n = len(_cells())
        assert (metadata["grid_cells"], metadata["grid_cells_answered"]) == (n, n - 1)
        assert 0 < metadata["answered_area_fraction"] < 1
        assert len(grid_errors) == 1 and (grid_errors["failure_type"] == "grid").all()


class TestFailedGrid:
    @pytest.mark.parametrize(
        "statuses,expected_status",
        [
            ([404, 404, 404], 404),  # every failed cell had no coverage
            ([404, 404, 404, 503, 503], 503),  # an outage is not a coverage gap, however the cells finished
            ([503, 404, 404, 404, 404], 503),
        ],
    )
    def test_status_is_404_only_when_every_failed_cell_was(self, tmp_path, monkeypatch, statuses, expected_status):
        api = _api(tmp_path, monkeypatch, dict(enumerate(statuses)), aoi_grid_min_pct=100)
        with pytest.raises(AIFeatureAPIGridError) as exc:
            _gridded(api)
        assert exc.value.status_code == expected_status

    def test_error_row_carries_how_far_the_grid_got(self, tmp_path, monkeypatch):
        n = len(_cells())
        api = _api(tmp_path, monkeypatch, {0: 404, 1: 503}, aoi_grid_min_pct=100)
        _, metadata, error, _ = api._attempt_gridding(geometry=AOI, region="au", aoi_id="big")
        assert metadata is None
        assert (error["grid_cells"], error["grid_cells_answered"]) == (n, n - 2)
        assert 0 < error["answered_area_fraction"] < 1
        assert f"{n - 2} of {n} grid cells answered" in error["message"]
        assert error["status_code"] == 503 and error["failure_type"] == "grid"

    def test_every_cell_is_queried_even_when_the_grid_fails(self, tmp_path, monkeypatch):
        n = len(_cells())
        api = _api(tmp_path, monkeypatch, {0: 503, 1: 503}, aoi_grid_min_pct=100)
        with pytest.raises(AIFeatureAPIGridError) as exc:
            _gridded(api)
        assert sorted(api.cell_calls) == list(range(n))
        assert exc.value.grid_cells_answered == n - 2, "the count must cover every cell, not stop at the threshold"

    def test_no_cell_answered_with_zero_threshold_keeps_the_real_status(self, tmp_path, monkeypatch):
        n = len(_cells())
        api = _api(tmp_path, monkeypatch, {i: 503 for i in range(n)}, aoi_grid_min_pct=0)
        with pytest.raises(AIFeatureAPIGridError) as exc:
            _gridded(api)
        assert exc.value.status_code == 503, "an all-503 grid must not be reported as a 404 coverage gap"
        assert (exc.value.grid_cells_answered, exc.value.answered_area_fraction) == (0, 0.0)


class TestThreshold:
    """aoi_grid_min_pct is applied as before: floor(cells * (100 - min_pct) / 100) failures are allowed."""

    @pytest.fixture
    def hundred_cells(self, monkeypatch):
        grid = gpd.GeoDataFrame(
            geometry=[
                box(151.2 + 0.001 * i, -33.9 + 0.001 * j, 151.201 + 0.001 * i, -33.899 + 0.001 * j)
                for i in range(10)
                for j in range(10)
            ],
            crs=API_CRS,
        )
        monkeypatch.setattr(feature_api_mod.geometry_utils, "split_geometry_into_grid", lambda **kwargs: grid.copy())

    @pytest.mark.parametrize("n_failed,passes", [(1, True), (2, False)])
    def test_99_pct_of_100_cells_allows_one_failure(self, tmp_path, monkeypatch, hundred_cells, n_failed, passes):
        api = _api(tmp_path, monkeypatch, {i: 404 for i in range(n_failed)}, aoi_grid_min_pct=99)
        if passes:
            assert _gridded(api)[3]["grid_cells_answered"] == 100 - n_failed
        else:
            with pytest.raises(AIFeatureAPIGridError):
                _gridded(api)
