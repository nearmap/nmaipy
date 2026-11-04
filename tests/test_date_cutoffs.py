"""Per-AOI and bulk since/until cutoffs are normalised on read and validated before an export sends anything."""

import datetime
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from nmaipy.aoi_io import normalise_date_column, read_from_file, validate_date_cutoffs
from nmaipy.constants import AOI_ID_COLUMN_NAME, BUILDING_NEW_ID
from nmaipy.exporter import AOIExporter


def _aois(**columns):
    n = len(next(iter(columns.values()))) if columns else 2
    return pd.DataFrame(columns, index=pd.Index([f"aoi{i}" for i in range(n)], name=AOI_ID_COLUMN_NAME))


class TestNormaliseDateColumn:
    def test_datetimes_become_date_strings(self):
        values = pd.Series(pd.to_datetime(["2025-03-06", None]))
        out = normalise_date_column(values)
        assert out.iloc[0] == "2025-03-06"
        assert pd.isna(out.iloc[1]), "NaT should stay null"

    def test_date_objects_become_date_strings(self):
        """A typed parquet date column reads back as datetime.date objects."""
        out = normalise_date_column(pd.Series([datetime.date(2025, 3, 6), None], dtype=object))
        assert out.iloc[0] == "2025-03-06"
        assert pd.isna(out.iloc[1])

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_strings_become_null(self, blank):
        out = normalise_date_column(pd.Series(["2025-03-06", blank, np.nan], dtype=object))
        assert out.iloc[0] == "2025-03-06"
        assert pd.isna(out.iloc[1]), f"{blank!r} means no per-AOI override, so it should be null"
        assert pd.isna(out.iloc[2])

    def test_other_values_are_left_for_validation(self):
        out = normalise_date_column(pd.Series(["2025/03/06", 20250306], dtype=object))
        assert list(out) == ["2025/03/06", 20250306]


class TestReadFromFileDates:
    def test_typed_parquet_date_column_comes_back_as_strings(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {AOI_ID_COLUMN_NAME: ["a", "b"], "until": [datetime.date(2025, 12, 31), None]},
            geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
            crs="EPSG:4326",
        )
        path = tmp_path / "aois.parquet"
        gdf.to_parquet(path, index=False)
        result = read_from_file(path)
        assert result.loc["a", "until"] == "2025-12-31"
        assert pd.isna(result.loc["b", "until"])

    def test_empty_string_cell_is_null(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {AOI_ID_COLUMN_NAME: ["a", "b"], "since": ["2025-03-06", ""]},
            geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
            crs="EPSG:4326",
        )
        path = tmp_path / "aois.parquet"
        gdf.to_parquet(path, index=False)
        result = read_from_file(path)
        assert pd.isna(result.loc["b", "since"]), "An empty cell should fall back to the bulk value, not drop the date"


class TestValidateDateCutoffs:
    def test_valid_columns_and_bulk_pass(self):
        aois = _aois(since=["2024-01-01", None], until=[None, "2025-06-30"])
        validate_date_cutoffs(aois, since="2023-01-01", until="2025-12-31")

    def test_no_columns_and_no_bulk_pass(self):
        validate_date_cutoffs(_aois(other=[1, 2]))

    @pytest.mark.parametrize("bad", ["2025/03/06", "06-03-2025", "2025-02-30", "2025-3-6", " 2025-03-06", 20250306])
    def test_malformed_column_value_is_rejected(self, bad):
        aois = _aois(until=["2025-03-06", bad])
        with pytest.raises(ValueError, match=r"column 'until'.*'aoi1'"):
            validate_date_cutoffs(aois)

    @pytest.mark.parametrize("flag,kwargs", [("--since", {"since": "2025/01/01"}), ("--until", {"until": "yesterday"})])
    def test_malformed_bulk_value_is_rejected(self, flag, kwargs):
        with pytest.raises(ValueError, match=flag):
            validate_date_cutoffs(_aois(other=[1, 2]), **kwargs)

    def test_inverted_row_is_rejected(self):
        aois = _aois(since=["2024-01-01", "2025-07-01"], until=["2025-01-01", "2025-06-30"])
        with pytest.raises(ValueError, match=r"'aoi1': since 2025-07-01 > until 2025-06-30"):
            validate_date_cutoffs(aois)

    def test_row_inverted_against_bulk_is_rejected(self):
        """A row's own since is compared with the bulk until it will be sent with."""
        aois = _aois(since=["2024-01-01", "2026-01-01"])
        with pytest.raises(ValueError, match=r"'aoi1': since 2026-01-01 > until 2025-12-31"):
            validate_date_cutoffs(aois, until="2025-12-31")

    def test_inverted_bulk_is_rejected(self):
        with pytest.raises(ValueError, match="since date after their until"):
            validate_date_cutoffs(_aois(other=[1, 2]), since="2025-12-31", until="2025-01-01")

    def test_same_day_is_allowed(self):
        validate_date_cutoffs(_aois(since=["2025-10-08"], until=["2025-10-08"]))


@pytest.mark.parametrize("bad_row", ["a", "b"])
def test_export_rejects_bad_dates_before_querying_even_with_survey_resource_ids(tmp_path, bad_row):
    """The check runs when the file has a survey_resource_id column, on every row.

    Row "b" has no resource id, so the Feature API uses its dates. Row "a" is pinned to one, so the Feature API
    ignores its dates, but the Roof Age API still sends them.
    """
    aois = gpd.GeoDataFrame(
        {
            AOI_ID_COLUMN_NAME: ["a", "b"],
            "survey_resource_id": ["resource", None],
            "until": ["2025/12/31" if row == bad_row else None for row in ("a", "b")],
        },
        geometry=[box(151.20, -33.90, 151.2005, -33.8995), box(151.21, -33.90, 151.2105, -33.8995)],
        crs="EPSG:4326",
    )
    aoi_file = tmp_path / "aois.csv"
    aois.to_wkt().to_csv(aoi_file, index=False)  # CSV keeps the string as written; GeoJSON readers parse dates
    exporter = AOIExporter(aoi_file=str(aoi_file), output_dir=str(tmp_path / "out"), country="au", packs=["building"])
    classes_df = pd.DataFrame({"description": ["Building"]}, index=pd.Index([BUILDING_NEW_ID], name="id"))
    with (
        patch("nmaipy.exporter.FeatureApi") as feature_api,
        patch.object(AOIExporter, "split_into_chunks") as split_into_chunks,
    ):
        feature_api.return_value.get_feature_classes.return_value = classes_df
        with pytest.raises(ValueError, match=rf"column 'until'.*'{bad_row}'"):
            exporter.run()
    split_into_chunks.assert_not_called()
