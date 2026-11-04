"""What happens when the Feature API returns a realtime score (an include) as ``{"skipped": true}``.

The API's scorer gives up on parcels with very many roofs, or under load. With ``retry_on_realtime_skip`` (the
default), FeatureApi first requests the AOI once more, unchanged and past the cache, after a delay. If the score is
still skipped and the AOI spans more than one small grid cell, it grids the AOI so each cell is a request the scorer
can handle. Otherwise the response is kept and the skipped scores stay null.
"""

import geopandas as gpd
import pytest
import requests
from shapely.geometry import Polygon, box, mapping

import nmaipy.feature_api as feature_api_module
from nmaipy.api_common import AIFeatureAPIError
from nmaipy.constants import REALTIME_SKIP_GRID_SIZE_DEGREES, REALTIME_SKIP_RETRY_DELAY_SECONDS, ROOF_ID
from nmaipy.feature_api import FeatureApi, _response_has_include_skips

# Inside one realtime-skip grid cell, so gridding it wouldn't split the request.
SMALL_AOI = box(144.9, -37.8, 144.9015, -37.7985)
# About 0.7 x 0.9 km (under the 1 sq km proactive gridding size), spanning 4 x 4 realtime-skip grid cells.
LARGE_AOI = box(144.9, -37.8, 144.908, -37.792)
SKIPPED = {"skipped": True}
SCORED = {"value": 50, "confidence": 0.9}


def _roof(feature_id, geometry=SMALL_AOI, **includes):
    """A raw Feature API roof feature, with includes keyed by their camelCase API names."""
    return {
        "id": feature_id,
        "classId": ROOF_ID,
        "description": "Roof",
        "confidence": 0.9,
        "parentId": None,
        "geometry": mapping(geometry),
        "areaSqm": 100.0,
        "clippedAreaSqm": 100.0,
        "unclippedAreaSqm": 100.0,
        "areaSqft": 1076.4,
        "clippedAreaSqft": 1076.4,
        "unclippedAreaSqft": 1076.4,
        "attributes": [],
        "surveyDate": "2026-01-01",
        "meshDate": "",
        **includes,
    }


def _payload(*features):
    return {
        "systemVersion": "gen6-test",
        "link": "https://apps.nearmap.com/maps/#/@-37.8,144.9,20.00z,0d/V/20260101",
        "surveyDate": "2026-01-01",
        "surveyId": "survey",
        "resourceId": "resource",
        "perspective": "Vert",
        "postcat": False,
        "aggregate": {"defensibleSpace": {"zones": []}},
        "features": list(features),
    }


def _skip_check(*features):
    """Run the skip check on payload_gdf output, the form get_features_gdf passes it."""
    features_gdf, _ = FeatureApi.payload_gdf(_payload(*features), aoi_id="aoi", parcel_mode=True)
    return _response_has_include_skips(features_gdf)


class TestResponseHasIncludeSkips:
    def test_no_skips_returns_false(self):
        any_skipped, skipped = _skip_check(_roof("a", roofSpotlightIndex=SCORED), _roof("b", hurricaneScore={}))
        assert any_skipped is False
        assert skipped == []

    def test_per_roof_skip_detected(self):
        any_skipped, skipped = _skip_check(_roof("a", hurricaneScore=SKIPPED), _roof("b"))
        assert any_skipped is True
        assert skipped == ["hurricane_score_skipped"]

    def test_none_input_returns_false(self):
        assert _response_has_include_skips(None) == (False, [])

    def test_multiple_skips_all_reported(self):
        any_skipped, skipped = _skip_check(
            _roof("a", roofSpotlightIndex=SKIPPED, windScore={"vulnerabilityScore": 3}),
            _roof("b", defensibleSpace=SKIPPED),
        )
        assert any_skipped is True
        assert set(skipped) == {"roof_spotlight_index_skipped", "defensible_space_skipped"}

    def test_roofs_without_the_include_are_not_skips(self):
        """Roofs lacking an include another roof carries come back NaN in that column, which is not a skip."""
        assert _skip_check(_roof("a", roofSpotlightIndex=SCORED), _roof("b"))[0] is False

    def test_skipped_false_is_not_a_skip(self):
        assert _skip_check(_roof("a", roofSpotlightIndex={"skipped": False, "value": 50}))[0] is False


@pytest.fixture
def calls(monkeypatch):
    """Record the skip-retry delays instead of sleeping; feature_api sleeps for other reasons too, so count only those."""
    record = {"sleeps": []}

    def sleep(seconds):
        if seconds == REALTIME_SKIP_RETRY_DELAY_SECONDS:
            record["sleeps"].append(seconds)

    monkeypatch.setattr(feature_api_module.time, "sleep", sleep)
    return record


def _install_stubs(api, monkeypatch, calls, *, responses, grid_fails=False):
    """Stub `get_features` to return `responses` in turn, and `_attempt_gridding`; the real `payload_gdf` parses.

    Each response is the roofSpotlightIndex include for the AOI's one roof, or an exception for the request to raise.
    `calls` collects the get_features keyword arguments under "requests" and the gridding keyword arguments under
    "grids".
    """
    calls.update(requests=[], grids=[])
    queue = list(responses)

    def stub_get_features(**kwargs):
        calls["requests"].append(kwargs)
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return _payload(_roof("roof", geometry=kwargs["geometry"], roofSpotlightIndex=response))

    def stub_attempt_gridding(**kwargs):
        calls["grids"].append(kwargs)
        if grid_fails:
            return None, None, {"aoi_id": kwargs["aoi_id"], "message": "Multiple dates", "failure_type": "grid"}, None
        return (
            gpd.GeoDataFrame({"feature_id": ["cell-roof"], "geometry": [SMALL_AOI]}, geometry="geometry"),
            {"aoi_id": kwargs["aoi_id"], "system_version": "v", "gridded": True},
            None,
            None,
        )

    monkeypatch.setattr(api, "get_features", stub_get_features)
    monkeypatch.setattr(api, "_attempt_gridding", stub_attempt_gridding)


def _skip_state(features):
    return features.set_index("feature_id")["roof_spotlight_index"].iloc[0]


class TestIncludeSkipHandling:
    def _api(self, tmp_path, **kwargs):
        return FeatureApi(api_key="test-key", cache_dir=tmp_path, **kwargs)

    def test_defaults(self, tmp_path):
        api = self._api(tmp_path)
        assert api.retry_on_realtime_skip is True

    def test_no_skip_makes_one_request(self, tmp_path, monkeypatch, calls):
        api = self._api(tmp_path)
        _install_stubs(api, monkeypatch, calls, responses=[SCORED])
        api.get_features_gdf(geometry=LARGE_AOI, region="au", aoi_id="aoi")
        assert len(calls["requests"]) == 1
        assert calls["sleeps"] == [] and calls["grids"] == []

    def test_skip_is_retried_once_unchanged_past_the_cache(self, tmp_path, monkeypatch, calls):
        api = self._api(tmp_path)
        _install_stubs(api, monkeypatch, calls, responses=[SKIPPED, SCORED])
        features, _, error, _ = api.get_features_gdf(geometry=LARGE_AOI, region="au", aoi_id="aoi")
        first, retry = calls["requests"]
        assert retry.pop("refresh_cache") is True, "The retry must bypass the cached skipped response"
        assert retry == first, "The retry should repeat the request, parcelMode included"
        assert calls["sleeps"] == [REALTIME_SKIP_RETRY_DELAY_SECONDS]
        assert calls["grids"] == [], "A retry that scores should not be followed by gridding"
        assert error is None and _skip_state(features) == SCORED

    @pytest.mark.parametrize(
        "failure",
        [
            AIFeatureAPIError(None, "request", message="Service unavailable"),
            requests.exceptions.RetryError("gave up"),
            requests.exceptions.Timeout("timed out"),
        ],
        ids=["api-error", "retry-error", "timeout"],
    )
    def test_failed_retry_keeps_the_first_response(self, tmp_path, monkeypatch, calls, failure):
        """The retry only tries to fill the skipped scores; a failing API must not cost the AOI its features."""
        api = self._api(tmp_path)
        _install_stubs(api, monkeypatch, calls, responses=[SKIPPED, failure])
        features, metadata, error, _ = api.get_features_gdf(geometry=LARGE_AOI, region="au", aoi_id="aoi")
        assert len(calls["requests"]) == 2
        assert error is None, f"A failed retry should not fail the AOI, got {error}"
        assert list(features["feature_id"]) == ["roof"] and metadata["survey_resource_id"] == "resource"
        assert _skip_state(features) == SKIPPED
        assert calls["grids"] == [], "Gridding after a failed retry would send more requests into the same failures"

    def test_persisting_skip_on_a_multi_cell_aoi_is_gridded_on_the_same_survey(self, tmp_path, monkeypatch, calls):
        api = self._api(tmp_path)
        _install_stubs(api, monkeypatch, calls, responses=[SKIPPED, SKIPPED])
        features, metadata, error, _ = api.get_features_gdf(geometry=LARGE_AOI, region="au", aoi_id="aoi")
        assert len(calls["requests"]) == 2
        assert len(calls["grids"]) == 1
        assert calls["grids"][0]["survey_resource_id"] == "resource", "Cells should be pinned to the response's survey"
        assert calls["grids"][0]["grid_size"] == REALTIME_SKIP_GRID_SIZE_DEGREES, "Cells small enough to score"
        assert error is None
        assert list(features["feature_id"]) == ["cell-roof"], "The gridded features should replace the response's"
        assert metadata["aggregate"] == {"defensibleSpace": {"zones": []}}, "Cells return no aggregate; keep this one"

    def test_persisting_skip_within_one_cell_keeps_the_response(self, tmp_path, monkeypatch, calls):
        """Gridding an AOI inside one cell would just re-ask for the same polygon without parcelMode."""
        api = self._api(tmp_path)
        _install_stubs(api, monkeypatch, calls, responses=[SKIPPED, SKIPPED])
        features, _, error, _ = api.get_features_gdf(geometry=SMALL_AOI, region="au", aoi_id="aoi")
        assert calls["grids"] == []
        assert error is None and _skip_state(features) == SKIPPED

    def test_failed_grid_keeps_the_ungridded_response(self, tmp_path, monkeypatch, calls):
        api = self._api(tmp_path)
        _install_stubs(api, monkeypatch, calls, responses=[SKIPPED, SKIPPED], grid_fails=True)
        features, metadata, error, grid_errors = api.get_features_gdf(geometry=LARGE_AOI, region="au", aoi_id="aoi")
        assert len(calls["grids"]) == 1
        assert error is None, f"A failed grid should not fail the AOI, got {error}"
        assert grid_errors is None
        assert list(features["feature_id"]) == ["roof"] and metadata["survey_resource_id"] == "resource"

    def test_off_leaves_skipped_scores_null(self, tmp_path, monkeypatch, calls):
        api = self._api(tmp_path, retry_on_realtime_skip=False)
        _install_stubs(api, monkeypatch, calls, responses=[SKIPPED])
        features, _, error, _ = api.get_features_gdf(geometry=LARGE_AOI, region="au", aoi_id="aoi")
        assert len(calls["requests"]) == 1 and calls["sleeps"] == [] and calls["grids"] == []
        assert error is None and _skip_state(features) == SKIPPED

    def test_deprecated_regrid_on_skip_false_turns_it_off(self, tmp_path):
        with pytest.warns(DeprecationWarning, match="regrid_on_skip is deprecated"):
            api = self._api(tmp_path, regrid_on_skip=False)
        assert api.retry_on_realtime_skip is False

    def test_deprecated_regrid_on_skip_attribute_aliases_retry_on_realtime_skip(self, tmp_path):
        api = self._api(tmp_path)
        with pytest.warns(DeprecationWarning, match="regrid_on_skip is deprecated"):
            assert api.regrid_on_skip is True
        with pytest.warns(DeprecationWarning, match="regrid_on_skip is deprecated"):
            api.regrid_on_skip = False
        assert api.retry_on_realtime_skip is False

    def test_grid_cells_are_not_retried_or_regridded(self, tmp_path, monkeypatch, calls):
        api = self._api(tmp_path)
        _install_stubs(api, monkeypatch, calls, responses=[SKIPPED])
        api.get_features_gdf(geometry=LARGE_AOI, region="au", aoi_id="cell", in_gridding_mode=True)
        assert len(calls["requests"]) == 1 and calls["grids"] == [] and calls["sleeps"] == []


def test_refresh_cache_fetches_and_replaces_the_cached_response(tmp_path, monkeypatch):
    api = FeatureApi(api_key="test-key", cache_dir=tmp_path)
    monkeypatch.setattr(api, "_read_cached_response", lambda path: {"cached": True})
    monkeypatch.setattr(api, "_fetch_results", lambda url, body, cache_path, in_gridding_mode: {"fresh": True})
    assert api.get_features(SMALL_AOI, "au") == {"cached": True}
    assert api.get_features(SMALL_AOI, "au", refresh_cache=True) == {"fresh": True}
