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
from nmaipy.feature_api import FeatureApi
from nmaipy.roof_age_api import RoofAgeApi
from nmaipy.roof_age_exporter import RoofAgeExporter


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


def test_nat_in_an_object_column_is_null():
    out = normalise_date_column(pd.Series([pd.Timestamp("2025-03-06"), pd.NaT], dtype=object))
    assert out.iloc[0] == "2025-03-06"
    assert pd.isna(out.iloc[1]), "NaT means no per-AOI override"


@pytest.mark.parametrize("blank", ["", "  "])
def test_blank_bulk_date_means_no_date(blank):
    """A blank bulk date sends no date filter, as before, and as a blank per-AOI cell does."""
    validate_date_cutoffs(_aois(since=["2025-01-01", None]), since=blank, until=blank)


def test_many_rows_with_few_dates_are_checked_by_distinct_value():
    aois = _aois(until=["2025-01-01", "2025-06-30"] * 50_000 + ["2025-13-01"])
    with pytest.raises(ValueError, match=r"1 row\(s\) don't, e.g. 'aoi100000': '2025-13-01'"):
        validate_date_cutoffs(aois)


def _bad_until_aois():
    return gpd.GeoDataFrame(
        {AOI_ID_COLUMN_NAME: ["a", "b"], "until": ["2025-12-31", "2025/12/31"]},
        geometry=[box(-74.29, 40.63, -74.2895, 40.6305), box(-74.28, 40.63, -74.2795, 40.6305)],
        crs="EPSG:4326",
    ).set_index(AOI_ID_COLUMN_NAME)


def test_feature_api_bulk_rejects_bad_per_row_dates_before_querying(tmp_path):
    api = FeatureApi(api_key="test-key", cache_dir=tmp_path)
    with patch.object(api, "_get_features_gdf_bulk_inner") as inner:
        with pytest.raises(ValueError, match=r"column 'until'.*'b'"):
            api.get_features_gdf_bulk(_bad_until_aois(), region="us")
    inner.assert_not_called()


def test_feature_api_bulk_sends_date_typed_per_row_dates_as_strings(tmp_path):
    """A date-typed per-row value used to fall back to the bulk date unseen; it is now sent as YYYY-MM-DD."""
    aois = _bad_until_aois().assign(until=pd.to_datetime(["2025-12-31", "2025-06-30"]))
    api = FeatureApi(api_key="test-key", cache_dir=tmp_path)
    with patch.object(api, "_get_features_gdf_bulk_inner", return_value=(None, None, pd.DataFrame())) as inner:
        api.get_features_gdf_bulk(aois, region="us", until_bulk="2026-01-01")
    sent = inner.call_args.kwargs["gdf"]
    assert list(sent["until"]) == ["2025-12-31", "2025-06-30"]
    assert aois["until"].dtype.kind == "M", "The caller's frame should not be modified"


@pytest.mark.parametrize("blank", ["", "  "])
def test_feature_api_bulk_sends_no_blank_bulk_date(tmp_path, blank):
    api = FeatureApi(api_key="test-key", cache_dir=tmp_path)
    with patch.object(api, "_get_features_gdf_bulk_inner", return_value=(None, None, pd.DataFrame())) as inner:
        api.get_features_gdf_bulk(
            _bad_until_aois().drop(columns="until"), region="us", since_bulk=blank, until_bulk=blank
        )
    sent = inner.call_args.kwargs
    assert sent["since_bulk"] is None and sent["until_bulk"] is None, "A blank bulk date means no date"


@pytest.mark.parametrize("blank", ["", "  "])
def test_roof_age_api_sends_no_blank_cutoff(tmp_path, blank):
    api = RoofAgeApi(api_key="test-key", cache_dir=tmp_path, until_as_of_date=blank, since_as_of_date=blank)
    with (
        patch.object(api, "_fetch_all_pages", return_value={"features": []}) as fetch,
        patch.object(api, "_parse_response"),
    ):
        api.get_roof_age_by_aoi(box(-74.29, 40.63, -74.2895, 40.6305), "a")
    sent = fetch.call_args.kwargs
    assert sent["until_as_of_date"] is None and sent["since_as_of_date"] is None, "A blank cutoff means no cutoff"
    (cached,) = tmp_path.rglob("*.json*")
    assert (cached.parent.parent.name, cached.parent.name) == ("none", "none"), f"Cache path names a cutoff: {cached}"


def test_roof_age_bulk_rejects_bad_per_row_dates_before_querying(tmp_path):
    api = RoofAgeApi(api_key="test-key", cache_dir=tmp_path)
    with patch.object(api, "_fetch_all_pages") as query, pytest.raises(ValueError, match=r"column 'until'.*'b'"):
        api.get_roof_age_bulk(_bad_until_aois())
    query.assert_not_called()


def test_standalone_roof_age_export_rejects_bad_dates_before_querying(tmp_path):
    aoi_file = tmp_path / "aois.csv"
    _bad_until_aois().reset_index().to_wkt().to_csv(aoi_file, index=False)
    exporter = RoofAgeExporter(aoi_file=str(aoi_file), output_dir=str(tmp_path / "out"), country="us", api_key="test")
    with patch.object(RoofAgeExporter, "split_into_chunks") as split_into_chunks:
        with pytest.raises(ValueError, match=r"column 'until'.*'b'"):
            exporter.run()
    split_into_chunks.assert_not_called()
