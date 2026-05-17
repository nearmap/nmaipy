"""Unit tests for the `prefer3d` flag on FeatureApi.

These tests run without API_KEY by patching ``requests.Session.post`` to return
canned responses. They use a real cached Feature API payload from
``tests/data/test_defensible_space_raw_payload.json`` as the 200 OK body — no
synthetic mock data (see project memory: real data only).
"""

import json
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS, SURVEY_RESOURCE_ID_COL_NAME
from nmaipy.feature_api import FeatureApi


# A representative real payload from a US 3D-capable survey. We reuse it for
# both passes in the mock — the mesh_date value will simply reflect the canned
# payload (empty string here, since the cached sample is from a 2D-perspective
# capture). Tests assert on flow control, not on the 3D content of the payload.
@pytest.fixture(scope="module")
def real_feature_payload() -> dict:
    path = Path(__file__).parent / "data" / "test_defensible_space_raw_payload.json"
    return json.loads(path.read_text())


def _aoi_gdf(aoi_id: str, polygon: Polygon, **extra_cols) -> gpd.GeoDataFrame:
    """Build a single-row AOI GeoDataFrame indexed by aoi_id, with optional extra columns."""
    row = {"geometry": polygon, **extra_cols}
    gdf = gpd.GeoDataFrame([row], crs=API_CRS)
    gdf[AOI_ID_COLUMN_NAME] = [aoi_id]
    gdf = gdf.set_index(AOI_ID_COLUMN_NAME)
    return gdf


def _square(lon: float, lat: float, size: float = 0.0005) -> Polygon:
    return Polygon(
        [
            (lon, lat),
            (lon + size, lat),
            (lon + size, lat + size),
            (lon, lat + size),
            (lon, lat),
        ]
    )


class _MockResponse:
    """Minimal stand-in for requests.Response used by FeatureApi._get_results."""

    def __init__(self, status_code: int, body: dict | str):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._body = body
        self.history = []
        if isinstance(body, dict):
            self.text = json.dumps(body)
        else:
            self.text = body

    def json(self):
        if isinstance(self._body, dict):
            return self._body
        return json.loads(self._body)


def test_only3d_and_prefer3d_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        FeatureApi(api_key="dummy", only3d=True, prefer3d=True)


def test_prefer3d_implies_only3d_internally():
    api = FeatureApi(api_key="dummy", prefer3d=True)
    assert api.only3d is True
    assert api.prefer3d is True


def test_prefer3d_default_false():
    api = FeatureApi(api_key="dummy")
    assert api.prefer3d is False
    assert api.only3d is False


def test_payload_gdf_extracts_mesh_date(real_feature_payload):
    _features, metadata = FeatureApi.payload_gdf(real_feature_payload, aoi_id="aoi-1")
    assert "mesh_date" in metadata
    assert metadata["mesh_date"] == real_feature_payload["3dDate"]


def test_payload_gdf_handles_missing_mesh_date():
    payload = {
        "systemVersion": "gen6-test",
        "link": "https://example/",
        "surveyDate": "2025-01-01",
        "surveyId": "s",
        "resourceId": "r",
        "perspective": "Vert",
        "postcat": False,
        "features": [],
    }
    _features, metadata = FeatureApi.payload_gdf(payload, aoi_id="aoi-1")
    assert metadata["mesh_date"] == ""


def test_prefer3d_falls_back_to_2d_on_404(tmp_path, real_feature_payload):
    """First pass with only3d=True gets a 404 → AOI retried without 3dCoverage param."""
    api = FeatureApi(
        api_key="dummy",
        prefer3d=True,
        cache_dir=tmp_path,
        threads=1,
    )

    captured_urls: list[str] = []

    def fake_post(self, url, *args, **kwargs):
        captured_urls.append(url)
        if "3dCoverage=true" in url:
            return _MockResponse(404, {"message": "No 3D survey available"})
        return _MockResponse(200, real_feature_payload)

    gdf = _aoi_gdf("aoi-1", _square(-111.926, 33.414))

    with patch("requests.Session.post", new=fake_post):
        features, metadata, errors = api.get_features_gdf_bulk(gdf=gdf, region="us")

    # Two requests must have been made for the same AOI: one with 3dCoverage=true (3D pass),
    # one without (2D fallback).
    three_d_calls = [u for u in captured_urls if "3dCoverage=true" in u]
    two_d_calls = [u for u in captured_urls if "3dCoverage=true" not in u]
    assert len(three_d_calls) >= 1, "first pass should have requested with 3dCoverage=true"
    assert len(two_d_calls) >= 1, "fallback pass should have requested without 3dCoverage=true"

    # The AOI should be present in the final results, not in errors.
    assert "aoi-1" in metadata.index
    assert "aoi-1" not in errors.index
    assert len(features) == len(real_feature_payload["features"])


def test_prefer3d_no_retry_when_first_pass_succeeds(tmp_path, real_feature_payload):
    api = FeatureApi(
        api_key="dummy",
        prefer3d=True,
        cache_dir=tmp_path,
        threads=1,
    )

    call_count = {"n": 0}

    def fake_post(self, url, *args, **kwargs):
        call_count["n"] += 1
        return _MockResponse(200, real_feature_payload)

    gdf = _aoi_gdf("aoi-1", _square(-111.926, 33.414))

    with patch("requests.Session.post", new=fake_post):
        features, metadata, errors = api.get_features_gdf_bulk(gdf=gdf, region="us")

    assert call_count["n"] == 1, "should not retry when first pass succeeded"
    assert "aoi-1" in metadata.index
    assert len(errors) == 0


def test_prefer3d_keeps_non_404_errors_without_retry(tmp_path):
    """A 500 in the first pass is NOT a 'no 3D coverage' signal — leave it as an error."""
    api = FeatureApi(
        api_key="dummy",
        prefer3d=True,
        cache_dir=tmp_path,
        threads=1,
    )

    captured_urls: list[str] = []

    def fake_post(self, url, *args, **kwargs):
        captured_urls.append(url)
        return _MockResponse(500, {"message": "internal server error"})

    gdf = _aoi_gdf("aoi-1", _square(-111.926, 33.414))

    with patch("requests.Session.post", new=fake_post):
        features, metadata, errors = api.get_features_gdf_bulk(gdf=gdf, region="us")

    # 500s already exhaust their own urllib3 retries, so we just need to confirm
    # no second-pass URL without 3dCoverage was issued.
    assert all("3dCoverage=true" in u for u in captured_urls), "non-404 errors must not trigger the 2D fallback pass"
    assert "aoi-1" in errors.index


def test_prefer3d_skips_retry_for_survey_resource_id_aois(tmp_path):
    """AOIs pinned to a survey_resource_id must NOT be retried — survey is already fixed."""
    api = FeatureApi(
        api_key="dummy",
        prefer3d=True,
        cache_dir=tmp_path,
        threads=1,
    )

    captured_urls: list[str] = []

    def fake_post(self, url, *args, **kwargs):
        captured_urls.append(url)
        return _MockResponse(404, {"message": "no coverage"})

    gdf = _aoi_gdf(
        "aoi-1",
        _square(-111.926, 33.414),
        **{SURVEY_RESOURCE_ID_COL_NAME: "fixed-resource-uuid"},
    )

    with patch("requests.Session.post", new=fake_post):
        features, metadata, errors = api.get_features_gdf_bulk(gdf=gdf, region="us")

    # Only one request should have been made — the one with the survey_resource_id path.
    # No 2D fallback because the AOI is pinned to a specific survey.
    assert len(captured_urls) == 1
    assert "aoi-1" in errors.index


def test_clone_for_2d_fallback_preserves_config(tmp_path):
    api = FeatureApi(
        api_key="dummy",
        prefer3d=True,
        cache_dir=tmp_path,
        threads=3,
        alpha=True,
        beta=True,
        prerelease=False,
        system_version_prefix="gen6-",
        parcel_mode=True,
        rapid=False,
    )
    sibling = api._clone_for_2d_fallback()
    assert sibling.only3d is False
    assert sibling.prefer3d is False
    assert sibling.alpha is True
    assert sibling.beta is True
    assert sibling.system_version_prefix == "gen6-"
    assert sibling.parcel_mode is True
    assert sibling.api_key == "dummy"
    # Cache dir is converted to str by BaseApiClient.__init__
    assert str(sibling.cache_dir) == str(tmp_path)
    assert sibling.threads == 3


def test_prefer3d_wholesale_fallback_when_threshold_exceeded(tmp_path, real_feature_payload):
    """When the 3D-only pass exceeds max_allowed_error_pct, the whole bulk should
    fall back to standard 2D rather than aborting the export."""
    api = FeatureApi(
        api_key="dummy",
        prefer3d=True,
        cache_dir=tmp_path,
        threads=1,
    )

    captured_urls: list[str] = []

    def fake_post(self, url, *args, **kwargs):
        captured_urls.append(url)
        if "3dCoverage=true" in url:
            # Both AOIs 404 in 3D-only mode
            return _MockResponse(404, {"message": "no 3D"})
        return _MockResponse(200, real_feature_payload)

    # Two AOIs; with max_allowed_error_pct=0, even one 3D error trips the threshold.
    gdf1 = _aoi_gdf("aoi-1", _square(-111.926, 33.414))
    gdf2 = _aoi_gdf("aoi-2", _square(-111.927, 33.415))
    gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2]), crs=API_CRS)

    with patch("requests.Session.post", new=fake_post):
        features, metadata, errors = api.get_features_gdf_bulk(gdf=gdf, region="us", max_allowed_error_pct=0)

    # Both AOIs should resolve via the wholesale 2D fallback — no AOIs in errors.
    assert "aoi-1" in metadata.index
    assert "aoi-2" in metadata.index
    assert len(errors) == 0
    # The first pass aborts as soon as the threshold trips, so we should see far fewer
    # 3dCoverage=true calls than the 2 the per-AOI fallback path would have made.
    two_d_calls = [u for u in captured_urls if "3dCoverage=true" not in u]
    assert len(two_d_calls) == 2, "wholesale fallback should issue one 2D call per AOI"


def test_prefer3d_bypassed_when_in_gridding_mode(tmp_path, real_feature_payload):
    """Gridded sub-requests must not trigger their own prefer3d dispatch."""
    api = FeatureApi(
        api_key="dummy",
        prefer3d=True,
        cache_dir=tmp_path,
        threads=1,
    )

    captured_urls: list[str] = []

    def fake_post(self, url, *args, **kwargs):
        captured_urls.append(url)
        if "3dCoverage=true" in url:
            return _MockResponse(404, {"message": "no 3D"})
        return _MockResponse(200, real_feature_payload)

    gdf = _aoi_gdf("aoi-1", _square(-111.926, 33.414))

    with patch("requests.Session.post", new=fake_post):
        # in_gridding_mode=True must bypass the prefer3d wrapper entirely.
        features, metadata, errors = api.get_features_gdf_bulk(gdf=gdf, region="us", in_gridding_mode=True)

    # With prefer3d bypassed, the single only3d=True attempt 404s and there is NO 2D retry.
    assert all("3dCoverage=true" in u for u in captured_urls)
    assert "aoi-1" in errors.index


def test_prefer3d_dedupes_partial_gridded_aoi_across_passes(tmp_path):
    """A gridded AOI with partial 3D coverage shows up in BOTH meta_3d (some cells
    returned 3D data) and errors_3d (other cells 404'd). The 2D fallback retries
    that AOI and produces its own meta_2d row. The merged metadata must not have
    duplicate aoi_id entries — otherwise downstream rollup code that calls
    metadata_df["mesh_date"].reindex(...) (e.g. via fillna) blows up with
    "cannot reindex on an axis with duplicate labels".
    """
    api = FeatureApi(api_key="dummy", prefer3d=True, cache_dir=tmp_path, threads=1)

    aoi_id = "aoi-partial-3d"
    geom = _square(-111.926, 33.414)
    gdf = _aoi_gdf(aoi_id, geom)

    def _meta_row(survey_date: str) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                {
                    "system_version": "gen6-test",
                    "link": "https://example/",
                    "survey_date": survey_date,
                    "survey_id": f"s-{survey_date}",
                    "survey_resource_id": f"r-{survey_date}",
                    "perspective": "Vert",
                    "postcat": False,
                    "mesh_date": "2025-09-03" if "2025" in survey_date else "",
                }
            ],
            index=pd.Index([aoi_id], name=AOI_ID_COLUMN_NAME),
        )
        return df

    def _features_row(survey_date: str) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            [{"feature_id": f"f-{survey_date}", "class_id": "x", "geometry": geom}],
            index=pd.Index([aoi_id], name=AOI_ID_COLUMN_NAME),
            crs=API_CRS,
        )

    # 3D pass: AOI has metadata + features (some cells succeeded) AND a 404 error row
    # in errors_3d (the failed cells were collapsed under the parent aoi_id with
    # status_code=404, mirroring get_features_gdf_gridded's behaviour at the seam
    # where failed grid cells get the parent AOI's aoi_id reassigned).
    features_3d = _features_row("2025-09-03")
    meta_3d = _meta_row("2025-09-03")
    errors_3d = pd.DataFrame(
        [
            {
                "status_code": 404,
                "message": "No 3D coverage at this grid cell",
                "text": "",
                "request": "grid",
                "failure_type": "grid",
            }
        ],
        index=pd.Index([aoi_id], name=AOI_ID_COLUMN_NAME),
    )

    # 2D pass: AOI resolves fully via the 2D fallback.
    features_2d = _features_row("2026-03-04")
    meta_2d = _meta_row("2026-03-04")
    errors_2d = pd.DataFrame()
    errors_2d.index.name = AOI_ID_COLUMN_NAME

    call_count = {"n": 0}

    def fake_inner(self, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return features_3d, meta_3d, errors_3d
        return features_2d, meta_2d, errors_2d

    with patch.object(FeatureApi, "_get_features_gdf_bulk_inner", new=fake_inner):
        merged_features, merged_meta, merged_errors = api.get_features_gdf_bulk(gdf=gdf, region="us")

    assert call_count["n"] == 2, "should have run a 2D fallback pass after the 3D 404"
    assert merged_meta.index.is_unique, (
        f"metadata index must not contain duplicates after prefer3d merge — got " f"{merged_meta.index.tolist()!r}"
    )
    # 2D fallback wins for the retried AOI.
    assert aoi_id in merged_meta.index
    assert merged_meta.loc[aoi_id, "survey_date"] == "2026-03-04"
    # The 404 error from the 3D pass is dropped (the AOI was successfully retried).
    assert aoi_id not in merged_errors.index
    # The retried AOI's 3D features are also dropped — 2D supersedes.
    assert len(merged_features) == 1
    assert merged_features["feature_id"].iloc[0] == "f-2026-03-04"

    # Sanity check that the downstream fillna pattern from exporter.py works on the
    # merged metadata — this is the operation that was throwing
    # "cannot reindex on an axis with duplicate labels" in production.
    rollup_mesh = pd.Series([None], index=pd.Index([aoi_id], name=AOI_ID_COLUMN_NAME))
    rollup_mesh.fillna(merged_meta["mesh_date"])
