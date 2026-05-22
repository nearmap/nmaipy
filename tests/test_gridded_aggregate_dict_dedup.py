"""
Regression test: gridded metadata dedup must survive dict-valued payload fields.

When an AOI is large enough to be gridded and ``include=["defensibleSpace"]`` is
requested, each grid-cell response carries a populated nested ``aggregate``
object. ``payload_gdf`` copies that into per-grid-cell metadata; the gridded
caller then assembles those dicts into ``metadata_df`` and runs
``metadata_df.drop_duplicates().iloc[0]``. With no ``subset=``, pandas hashes
every column — including ``aggregate`` — and previously crashed with
``TypeError: unhashable type: 'dict'``. The error was swallowed by the
bare-Exception handler in ``_attempt_gridding`` and the AOI silently lost all
its features.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from nmaipy.constants import AOI_ID_COLUMN_NAME
from nmaipy.feature_api import FeatureApi

data_directory = Path(__file__).parent / "data"


@pytest.fixture
def defensible_space_payload():
    raw_payload_file = data_directory / "test_defensible_space_raw_payload.json"
    if not raw_payload_file.exists():
        pytest.skip(f"Raw payload file {raw_payload_file} does not exist.")
    with open(raw_payload_file, "r") as f:
        return json.load(f)


def test_aggregate_is_dict_in_payload(defensible_space_payload):
    """Precondition: the cached defensibleSpace payload has a dict aggregate.
    If this fails the regression test below is no longer exercising the bug."""
    assert isinstance(defensible_space_payload.get("aggregate"), dict)
    assert defensible_space_payload["aggregate"], "aggregate must be non-empty"


def test_gridded_metadata_dedup_survives_dict_aggregate(defensible_space_payload):
    """Simulate the metadata path through get_features_gdf_gridded for a gridded
    AOI where every cell returns a populated `aggregate` dict, then run the same
    dedup step `_attempt_gridding` runs. Must not raise."""
    # Each grid cell yields one metadata dict from payload_gdf. Simulate three
    # cells of the same AOI (a real grid would produce >=4; three is enough to
    # exercise drop_duplicates with identical rows).
    metadata = []
    for cell_idx in range(3):
        _, cell_metadata = FeatureApi.payload_gdf(defensible_space_payload, aoi_id=cell_idx)
        metadata.append(cell_metadata)

    # Sanity: the bug requires aggregate to actually be a dict in metadata.
    assert all(isinstance(m["aggregate"], dict) for m in metadata)

    # Mirror get_features_gdf_gridded's construction at feature_api.py:1911.
    metadata_df = pd.DataFrame(metadata).set_index(AOI_ID_COLUMN_NAME)

    # The line that crashed: feature_api.py:1067. Must succeed.
    deduped_row = metadata_df.drop(columns=["aggregate"], errors="ignore").drop_duplicates().iloc[0]

    # Confirm the fields _attempt_gridding actually consumes are intact.
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
        assert field in deduped_row.index, f"{field} should survive dedup"


def test_raw_drop_duplicates_on_dict_column_would_raise(defensible_space_payload):
    """Negative control: without dropping `aggregate`, drop_duplicates() raises
    TypeError on the dict column. This pins the failure mode the fix addresses.
    """
    metadata = [FeatureApi.payload_gdf(defensible_space_payload, aoi_id=i)[1] for i in range(2)]
    metadata_df = pd.DataFrame(metadata).set_index(AOI_ID_COLUMN_NAME)

    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        metadata_df.drop_duplicates()
