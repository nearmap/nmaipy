"""Roof objects are reported on the roof the API names as their parent, within the roof's own AOI."""

from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from nmaipy.aoi_io import read_from_file
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    BUILDING_NEW_ID,
    ROOF_ID,
    SOLAR_ID,
    SQUARED_METERS_TO_SQUARED_FEET,
    VENT_ID,
)
import nmaipy.exporter as exporter_mod
from nmaipy.exporter import NearmapAIExporter, _compute_all_per_class_data, export_feature_class
from nmaipy.feature_attributes import roof_object_attributes
import nmaipy.parcels as parcels_mod
from nmaipy.parcels import feature_attributes, parcel_rollup, unavailable_classes_by_aoi

ROOF_FID = "roof-shared"
BUILDING_FID = "building-shared"
ROOF_OBJECTS = {VENT_ID: "Vent", SOLAR_ID: "Solar Panel"}
SOLAR_ONLY_AT_A = [{"systemVersion": "gen6-a", "perspective": "Vert", "status": "prod"}]
CLASSES = pd.DataFrame(
    {"description": ["Roof", "Building", "Vent", "Solar Panel"], "availability": [None, None, None, SOLAR_ONLY_AT_A]},
    index=pd.Index([ROOF_ID, BUILDING_NEW_ID, VENT_ID, SOLAR_ID], name="id"),
)
OBJECT_COLUMNS = ("present", "count", "area_sqm", "confidence")


def _row(aoi_id, class_id, feature_id, geom, clipped_sqm, parent_id=None, confidence=0.9) -> dict:
    description = CLASSES.description.get(class_id, "Other")
    return {
        AOI_ID_COLUMN_NAME: aoi_id,
        "feature_id": feature_id,
        "class_id": class_id,
        "description": description,
        "confidence": confidence,
        "fidelity": 0.9,
        "parent_id": parent_id,
        "is_primary": class_id in (ROOF_ID, BUILDING_NEW_ID),
        "geometry": geom,
        "attributes": [],
        "area_sqm": clipped_sqm,
        "clipped_area_sqm": clipped_sqm,
        "unclipped_area_sqm": clipped_sqm,
        "area_sqft": clipped_sqm * SQUARED_METERS_TO_SQUARED_FEET,
        "clipped_area_sqft": clipped_sqm * SQUARED_METERS_TO_SQUARED_FEET,
        "unclipped_area_sqft": clipped_sqm * SQUARED_METERS_TO_SQUARED_FEET,
        "survey_date": "2024-01-01",
        "mesh_date": "2024-01-01",
        "belongs_to_parcel": True,
    }


def _shared_roof_chunk(drop: tuple = ()) -> gpd.GeoDataFrame:
    """One roof and building split between AOIs a and b (same feature ids), with vents on the roof in both."""
    roof_a, roof_b = box(0, 0, 0.0001, 0.0001), box(0.0001, 0, 0.0002, 0.0001)
    rows = [
        _row("a", ROOF_ID, ROOF_FID, roof_a, 100.0),
        _row("a", BUILDING_NEW_ID, BUILDING_FID, roof_a, 100.0),
        _row("a", VENT_ID, "vent-1", box(0.00001, 0.00001, 0.00002, 0.00002), 2.0, ROOF_FID, 0.6),
        _row("a", VENT_ID, "vent-split", box(0.00009, 0.00001, 0.0001, 0.00002), 0.3, ROOF_FID, 0.8),
        _row("a", VENT_ID, "vent-elsewhere", box(0.0005, 0.0005, 0.0006, 0.0006), 1.0, "another-roof"),
        _row("a", VENT_ID, "vent-orphan", box(0.00003, 0.00003, 0.00004, 0.00004), 1.0),
        _row("b", ROOF_ID, ROOF_FID, roof_b, 50.0),
        _row("b", BUILDING_NEW_ID, BUILDING_FID, roof_b, 50.0),
        _row("b", VENT_ID, "vent-split", box(0.0001, 0.00001, 0.00011, 0.00002), 0.2, ROOF_FID, 0.7),
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=API_CRS).drop(columns=list(drop))


def _per_class(chunk, classes=(ROOF_ID, BUILDING_NEW_ID), **kwargs) -> dict:
    res = _compute_all_per_class_data(
        chunk,
        country="au",
        aoi_input_columns=[],
        whitelisted_classes=list(classes),
        roof_object_classes=ROOF_OBJECTS,
        **kwargs,
    )
    return {cid: res[cid]["tabular"].to_pandas().set_index(AOI_ID_COLUMN_NAME) for cid in res}


def _rollup(chunk, aoi_geoms: dict, **kwargs) -> pd.DataFrame:
    parcels = gpd.GeoDataFrame(
        geometry=list(aoi_geoms.values()), index=pd.Index(list(aoi_geoms), name=AOI_ID_COLUMN_NAME), crs=API_CRS
    )
    return parcel_rollup(parcels, chunk, CLASSES, "au", "largest_intersection", **kwargs)


def _two_version_metadata() -> list:
    meta = pd.DataFrame({"system_version": ["gen6-a", "gen6-b"]}, index=pd.Index(["a", "b"], name=AOI_ID_COLUMN_NAME))
    return [(meta, CLASSES)]


def test_shared_roof_reports_each_aois_own_objects():
    roof = _per_class(_shared_roof_chunk())[ROOF_ID]
    assert roof.loc["a", "vent_count"] == 2, "only the vents with this roof as parent, in this AOI"
    assert roof.loc["a", "vent_area_sqm"] == pytest.approx(2.3)
    assert roof.loc["a", "vent_confidence"] == pytest.approx(0.8)
    assert roof.loc["b", "vent_count"] == 1, "AOI a's vents must not leak into AOI b's copy of the roof"
    assert roof.loc["b", "vent_area_sqm"] == pytest.approx(0.2), "the split vent keeps AOI b's clipped area"
    assert roof.loc["a", "vent_present"] == "Y"


def test_unavailable_class_is_null_and_available_class_with_no_objects_is_zero():
    chunk = _shared_roof_chunk()
    unavailable = unavailable_classes_by_aoi(["a", "b"], _two_version_metadata())
    assert unavailable == {"b": frozenset({SOLAR_ID})}
    files = _per_class(chunk, unavailable_by_aoi=unavailable)
    rollup = _rollup(
        chunk,
        {"a": box(0, 0, 0.0001, 0.0001), "b": box(0.0001, 0, 0.0002, 0.0001)},
        api_metadata=_two_version_metadata(),
    )
    outputs = {
        "roof": (files[ROOF_ID], "solar_panel_"),
        "building": (files[BUILDING_NEW_ID], "primary_child_roof_solar_panel_"),
        "rollup": (rollup, "primary_roof_solar_panel_"),
    }
    for name, (df, prefix) in outputs.items():
        for col in OBJECT_COLUMNS:
            assert pd.isna(df.loc["b", prefix + col]), (
                f"{name} {prefix + col} must be null where the class is unavailable"
            )
        assert df.loc["a", prefix + "present"] == "N", name
        assert df.loc["a", prefix + "count"] == 0, name
        assert df.loc["a", prefix + "area_sqm"] == 0.0, name
        assert pd.isna(df.loc["a", prefix + "confidence"]), name


@pytest.mark.parametrize("country", ["us", "au"])
def test_gridded_aoi_counts_only_objects_inside_it(country):
    aoi = box(0, 0, 0.0001, 0.0001)
    roofs = pd.DataFrame({AOI_ID_COLUMN_NAME: ["g"], "feature_id": [ROOF_FID]})
    children = gpd.GeoDataFrame(
        [
            _row("g", VENT_ID, "inside-rounded-to-zero", box(0.00001, 0.00001, 0.000011, 0.000011), 0.0, ROOF_FID),
            _row("g", VENT_ID, "outside", box(0.0003, 0.0003, 0.0004, 0.0004), 0.0, ROOF_FID),
            _row("g", VENT_ID, "no-geometry", Polygon(), 1.0, ROOF_FID),
        ],
        geometry="geometry",
        crs=API_CRS,
    )
    out = roof_object_attributes(roofs, children, ROOF_OBJECTS, country, aoi_geometries={"g": aoi})
    assert out.loc[0, "vent_count"] == 1, "inside counts even at area 0; outside the grid AOI and empty do not"
    assert out.loc[0, "vent_present"] == "Y"


def test_building_file_has_roof_objects_without_the_roof_class():
    building = _per_class(_shared_roof_chunk(), classes=(BUILDING_NEW_ID,))[BUILDING_NEW_ID]
    assert building.loc["a", "primary_child_roof_vent_count"] == 2
    assert building.loc["b", "primary_child_roof_vent_count"] == 1


def test_export_feature_class_without_the_new_keywords_writes_no_roof_object_columns(tmp_path):
    chunk = _shared_roof_chunk()
    path, _ = export_feature_class(chunk, ROOF_ID, "Roof", "au", str(tmp_path), export_geo_parquet=False)
    columns = pd.read_csv(path).columns
    assert not [c for c in columns if c.startswith(("vent_", "solar_panel_"))], list(columns)


def test_roof_file_matches_rollup_for_the_primary_roof_of_a_gridded_aoi():
    aoi = box(0, 0, 0.0001, 0.0001)
    chunk = _shared_roof_chunk()
    chunk = chunk[chunk[AOI_ID_COLUMN_NAME] == "a"]
    empty = gpd.GeoDataFrame([_row("a", VENT_ID, "vent-empty", Polygon(), 4.0, ROOF_FID)], crs=API_CRS)
    outside = gpd.GeoDataFrame(
        [_row("a", VENT_ID, "vent-outside", box(0.0003, 0.0003, 0.0004, 0.0004), 0.0, ROOF_FID)], crs=API_CRS
    )
    chunk = pd.concat([chunk, empty, outside], ignore_index=True)
    roof = _per_class(chunk, classes=(ROOF_ID,), aoi_geometries={"a": aoi})[ROOF_ID]
    rollup = _rollup(chunk, {"a": aoi}, gridded_aoi_ids={"a"})
    for col in OBJECT_COLUMNS:
        roof_value, rollup_value = roof.loc["a", f"vent_{col}"], rollup.loc["a", f"primary_roof_vent_{col}"]
        if col == "confidence":
            roof_value = round(roof_value, 2)
        assert roof_value == pytest.approx(rollup_value) if col != "present" else roof_value == rollup_value, col
    assert roof.loc["a", "vent_count"] == 2, "the empty and the outside vents are not counted"


@pytest.mark.parametrize("classes", [(ROOF_ID, BUILDING_NEW_ID), (BUILDING_NEW_ID,)], ids=["cache", "building"])
def test_missing_clipped_area_fails_the_per_class_computation(classes):
    with pytest.raises(ValueError, match="clipped_area_sqm"):
        _per_class(_shared_roof_chunk(drop=("clipped_area_sqm",)), classes=classes)


def test_rollup_reports_the_objects_on_the_primary_roof():
    """feature_attributes on one parcel indexed by aoi_id, as a direct caller passes it."""
    rows = [
        _row("p", ROOF_ID, "main", box(0, 0, 0.0001, 0.0001), 100.0),
        _row("p", ROOF_ID, "shed", box(0.0002, 0, 0.00025, 0.00005), 25.0),
        _row("p", VENT_ID, "v1", box(0.00004, 0.00004, 0.00005, 0.00005), 0.7, "main", 0.6),
        _row("p", VENT_ID, "v2", box(0.00006, 0.00006, 0.00007, 0.00007), 0.8, "main", 0.8),
        _row("p", VENT_ID, "v3", box(0.00021, 0.00001, 0.00022, 0.00002), 0.9, "shed"),
    ]
    features = gpd.GeoDataFrame(rows, geometry="geometry", crs=API_CRS).set_index(AOI_ID_COLUMN_NAME)
    result = feature_attributes(
        features, CLASSES, country="au", parcel_geom=None, primary_decision="largest_intersection"
    )
    assert result["vent_count"] == 3, "the parcel-level count is unchanged"
    assert result["primary_roof_vent_present"] == "Y"
    assert result["primary_roof_vent_count"] == 2
    assert result["primary_roof_vent_area_sqm"] == pytest.approx(1.5)
    assert result["primary_roof_vent_confidence"] == pytest.approx(0.8)
    assert result["primary_roof_solar_panel_present"] == "N"


class TestProcessChunk:
    @pytest.fixture
    def exporter(self, tmp_path, caplog):
        ex = NearmapAIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
            chunk_size=2,
            no_cache=True,
        )
        Path(ex.chunk_path).mkdir(parents=True, exist_ok=True)
        ex.logger.addHandler(caplog.handler)
        yield ex
        ex.logger.removeHandler(caplog.handler)

    @staticmethod
    def _aoi_gdf(tmp_path) -> gpd.GeoDataFrame:
        """Integer aoi ids and a system_version input column, read from an AOI file."""
        path = tmp_path / "aois.csv"
        pd.DataFrame(
            {
                AOI_ID_COLUMN_NAME: [1, 2],
                "system_version": ["mine-1", "mine-2"],
                "geometry": [box(0, 0, 0.0001, 0.0001).wkt, box(0.0001, 0, 0.0002, 0.0001).wkt],
            }
        ).to_csv(path, index=False)
        return read_from_file(str(path))

    @staticmethod
    def _api_response():
        features = _shared_roof_chunk()
        features[AOI_ID_COLUMN_NAME] = features[AOI_ID_COLUMN_NAME].map({"a": 1, "b": 2})
        features = features.drop(columns=["is_primary"]).set_index(AOI_ID_COLUMN_NAME)
        meta = pd.DataFrame(
            {"system_version": ["gen6-a", "gen6-b"], "survey_date": ["2024-01-01", "2024-01-01"]},
            index=pd.Index([1, 2], name=AOI_ID_COLUMN_NAME),
        )
        errors = pd.DataFrame(columns=["status_code", "message", "text", "request", "failure_type"])
        errors.index.name = AOI_ID_COLUMN_NAME
        return features, meta, errors

    def _run(self, exporter, tmp_path) -> Path:
        with (
            patch("nmaipy.exporter.FeatureApi") as mock_api,
            patch("nmaipy.exporter.collect_latency_stats_from_apis", return_value=None),
        ):
            mock_api.return_value.get_features_gdf_bulk.return_value = self._api_response()
            exporter.process_chunk(chunk_id="0000", aoi_gdf=self._aoi_gdf(tmp_path), classes_df=CLASSES)
        return Path(exporter.chunk_path)

    def test_unavailable_class_is_null_in_every_file(self, exporter, tmp_path):
        chunk_path = self._run(exporter, tmp_path)
        outputs = {
            "roof": ("roof_0000.parquet", "solar_panel_"),
            "building": ("building_0000.parquet", "primary_child_roof_solar_panel_"),
            "rollup": ("rollup_0000.parquet", "primary_roof_solar_panel_"),
        }
        for name, (file, prefix) in outputs.items():
            df = pd.read_parquet(chunk_path / file)
            if AOI_ID_COLUMN_NAME in df.columns:
                df = df.set_index(AOI_ID_COLUMN_NAME)
            assert df.loc[2, prefix + "count"] is pd.NA or pd.isna(df.loc[2, prefix + "count"]), name
            assert pd.isna(df.loc[2, prefix + "present"]), name
            assert df.loc[1, prefix + "count"] == 0, f"{name}: available at AOI 1's version, none found"
            assert df.loc[1, prefix.replace("solar_panel", "vent") + "count"] == 2, name

    def test_roof_object_failure_in_per_class_files_fails_the_chunk(self, exporter, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(exporter_mod, "roof_object_attributes", _raise)
        chunk_path = self._run(exporter, tmp_path)
        assert not (chunk_path / "rollup_0000.parquet").exists(), "the chunk must not be marked complete"
        assert "failed computing per-class files: ValueError: simulated" in caplog.text

    def test_roof_object_failure_in_rollup_raises(self, exporter, tmp_path, monkeypatch):
        monkeypatch.setattr(parcels_mod, "_roof_object_columns", _raise)
        with pytest.raises(ValueError, match="simulated"):
            self._run(exporter, tmp_path)


def _raise(*_args, **_kwargs):
    raise ValueError("simulated")
