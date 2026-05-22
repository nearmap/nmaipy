"""
Regression tests: gridded metadata dedup must survive dict-valued payload fields.

When an AOI is large enough to be gridded and ``include=["defensibleSpace"]`` is
requested, each grid-cell response carries a populated nested ``aggregate``
object. ``payload_gdf`` copies that into per-grid-cell metadata; the gridded
caller then assembles those dicts into ``metadata_df`` and previously ran
``metadata_df.drop_duplicates().iloc[0]``. With no ``subset=``, pandas hashes
every column — including ``aggregate`` — and crashed with
``TypeError: unhashable type: 'dict'``. The error was swallowed by the
bare-Exception handler in ``_attempt_gridding`` and the AOI silently lost all
its features.

The same failure mode would also surface if ``postcat`` returns its object form
(per the v4 API spec it may be either a boolean or a nested object describing
post-catastrophe imagery). The fix uses ``subset=`` whitelisting of
guaranteed-scalar columns rather than relying on dropping any one column.
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS
from nmaipy.feature_api import FeatureApi

data_directory = Path(__file__).parent / "data"


@pytest.fixture
def defensible_space_payload():
    raw_payload_file = data_directory / "test_defensible_space_raw_payload.json"
    if not raw_payload_file.exists():
        pytest.skip(f"Raw payload file {raw_payload_file} does not exist.")
    with open(raw_payload_file, "r") as f:
        return json.load(f)


def _build_gridded_metadata_df(payload, n_cells=3):
    """Build a metadata_df shaped like the one get_features_gdf_gridded produces."""
    metadata = [FeatureApi.payload_gdf(payload, aoi_id=i)[1] for i in range(n_cells)]
    return pd.DataFrame(metadata).set_index(AOI_ID_COLUMN_NAME)


def test_aggregate_is_dict_in_payload(defensible_space_payload):
    """Precondition: the cached defensibleSpace payload has a dict aggregate.
    If this fails the regression tests below are no longer exercising the bug."""
    assert isinstance(defensible_space_payload.get("aggregate"), dict)
    assert defensible_space_payload["aggregate"], "aggregate must be non-empty"


def test_raw_drop_duplicates_on_dict_column_would_raise(defensible_space_payload):
    """Negative control: without subset=, drop_duplicates() raises TypeError on
    the dict column. Pins the failure mode the fix addresses."""
    metadata_df = _build_gridded_metadata_df(defensible_space_payload, n_cells=2)
    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        metadata_df.drop_duplicates()


def test_attempt_gridding_does_not_swallow_dict_aggregate(monkeypatch, defensible_space_payload):
    """End-to-end regression: ``_attempt_gridding`` must succeed when grid-cell
    metadata contains dict-valued ``aggregate``. Before the fix, the inner
    ``drop_duplicates()`` raised TypeError, the bare-Exception handler turned it
    into a per-AOI grid error with message ``"Gridding failed: unhashable type:
    'dict'"``, and the AOI silently lost all features.

    This test patches ``get_features_gdf_gridded`` so we exercise the dedup-and-
    consume logic in ``_attempt_gridding`` directly, without making any HTTP
    calls. If [feature_api.py:1067-1078] is reverted, this test catches it.
    """
    api = FeatureApi(api_key="fake")

    fake_metadata_df = _build_gridded_metadata_df(defensible_space_payload, n_cells=3)
    fake_features_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=API_CRS)
    fake_errors_df = pd.DataFrame([])

    def fake_gridded(**kwargs):
        return fake_features_gdf, fake_metadata_df, fake_errors_df

    monkeypatch.setattr(api, "get_features_gdf_gridded", fake_gridded)

    features_gdf, metadata, error, errors_df = api._attempt_gridding(
        geometry=Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
        region="us",
        aoi_id=999,
    )

    # Pre-fix symptom: error dict populated with "Gridding failed: unhashable type: 'dict'".
    assert error is None, f"_attempt_gridding silently absorbed an error: {error}"

    # Post-fix: metadata is a real dict the consumer can use.
    assert metadata is not None, "expected real metadata, got None"
    # The outer aoi_id must be attached — not the per-grid-cell temp id assigned
    # internally by get_features_gdf_gridded.
    assert metadata[AOI_ID_COLUMN_NAME] == 999
    # Spot-check that scalar consumer fields are intact.
    for field in (
        "system_version",
        "link",
        "survey_date",
        "survey_id",
        "survey_resource_id",
        "perspective",
        "postcat",
        "mesh_date",
    ):
        assert field in metadata, f"{field} missing from metadata"
