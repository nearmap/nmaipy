"""Tests for dominant material/shape aggregate columns."""

import pytest

from nmaipy.feature_attributes import (
    _build_dominant_columns,
    _get_component_stats,
    flatten_roof_attributes,
)


def _make_roof(material_components, shape_components):
    """Build a minimal roof dict with material and shape attributes."""
    attributes = []
    if material_components:
        attributes.append({
            "classId": "89c7d478-58de-56bd-96d2-e71e27a36905",
            "description": "Roof material",
            "components": material_components,
        })
    if shape_components:
        attributes.append({
            "classId": "20a58db2-bc02-531d-98f5-451f88ce1fed",
            "description": "Roof types",
            "components": shape_components,
        })
    return {"attributes": attributes}


def _mat(desc, area, confidence, ratio, class_id="mat-class-id", dominant=False):
    return {
        "classId": class_id,
        "description": desc,
        "areaSqm": area,
        "areaSqft": area * 10.764,
        "confidence": confidence,
        "ratio": ratio,
        "dominant": dominant,
    }


def _shape(desc, area, confidence, ratio, class_id="shape-class-id"):
    return {
        "classId": class_id,
        "description": desc,
        "areaSqm": area,
        "areaSqft": area * 10.764,
        "confidence": confidence,
        "ratio": ratio,
    }


class TestGetComponentStats:
    def test_uses_raw_api_when_no_recalc(self):
        comp = _mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-id")
        stats = _get_component_stats(comp, recalc_attrs=None, country="us")
        assert stats["name"] == "tile_roof"
        assert stats["class_id"] == "tile-id"
        assert stats["ratio"] == 0.8
        assert stats["area"] == pytest.approx(100 * 10.764)
        assert stats["confidence"] == 0.9

    def test_uses_recalc_when_available(self):
        comp = _mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-id")
        recalc = {
            "tile_roof_class_id": "tile-id",
            "tile_roof_ratio": 0.6,
            "tile_roof_area_sqft": 500.0,
            "tile_roof_confidence": 0.85,
        }
        stats = _get_component_stats(comp, recalc_attrs=recalc, country="us")
        assert stats["ratio"] == 0.6
        assert stats["area"] == 500.0
        assert stats["confidence"] == 0.85
        assert stats["class_id"] == "tile-id"

    def test_metric_country_uses_sqm(self):
        comp = _mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-id")
        stats = _get_component_stats(comp, recalc_attrs=None, country="au")
        assert stats["area"] == 100

    def test_recalc_metric_country(self):
        comp = _mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-id")
        recalc = {
            "tile_roof_class_id": "tile-id",
            "tile_roof_ratio": 0.6,
            "tile_roof_area_sqm": 50.0,
            "tile_roof_confidence": 0.85,
        }
        stats = _get_component_stats(comp, recalc_attrs=recalc, country="au")
        assert stats["area"] == 50.0


class TestBuildDominantColumns:
    def test_selects_by_ratio(self):
        comps = [
            _mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-id"),
            _mat("Metal Roof", 20, 0.7, 0.2, class_id="metal-id"),
        ]
        cols = _build_dominant_columns(comps, None, "us", "roof_material")
        assert cols["dominant_roof_material_feature_class"] == "tile-id"
        assert cols["dominant_roof_material_description"] == "tile_roof"

    def test_highest_ratio_wins_not_area(self):
        """Highest ratio wins even if another component has larger area."""
        comps = [
            _mat("Tile Roof", 200, 0.9, 0.4, class_id="tile-id"),
            _mat("Shingle Roof", 80, 0.8, 0.6, class_id="shingle-id"),
        ]
        cols = _build_dominant_columns(comps, None, "us", "roof_material")
        assert cols["dominant_roof_material_description"] == "shingle_roof"

    def test_empty_returns_empty(self):
        assert _build_dominant_columns([], None, "us", "roof_material") == {}

    def test_shapes_select_by_ratio(self):
        comps = [
            _shape("Hip", 48, 0.75, 0.15, class_id="hip-id"),
            _shape("Gable", 75, 0.78, 0.24, class_id="gable-id"),
            _shape("Flat", 0, 1.0, 0.0, class_id="flat-id"),
        ]
        cols = _build_dominant_columns(comps, None, "us", "roof_types")
        assert cols["dominant_roof_types_description"] == "gable"

    def test_material_unknown_when_ratio_below_threshold(self):
        comps = [
            _mat("Tile Roof", 40, 0.9, 0.4, class_id="tile-id"),
            _mat("Metal Roof", 30, 0.7, 0.3, class_id="metal-id"),
        ]
        cols = _build_dominant_columns(comps, None, "us", "roof_material")
        assert cols["dominant_roof_material_feature_class"] is None
        assert cols["dominant_roof_material_description"] == "unknown"

    def test_shape_unknown_when_all_zero_area(self):
        comps = [
            _shape("Hip", 0, 0.75, 0.0, class_id="hip-id"),
            _shape("Gable", 0, 0.6, 0.0, class_id="gable-id"),
        ]
        cols = _build_dominant_columns(comps, None, "us", "roof_types")
        assert cols["dominant_roof_types_feature_class"] is None
        assert cols["dominant_roof_types_description"] == "unknown"

    def test_recalc_overrides_raw_winner(self):
        """When recalc_attrs reverses the ranking, the recalc winner is used."""
        comps = [
            _mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-id"),
            _mat("Metal Roof", 20, 0.7, 0.2, class_id="metal-id"),
        ]
        recalc = {
            "tile_roof_class_id": "tile-id",
            "tile_roof_ratio": 0.3,
            "tile_roof_area_sqft": 300.0,
            "tile_roof_confidence": 0.9,
            "metal_roof_class_id": "metal-id",
            "metal_roof_ratio": 0.7,
            "metal_roof_area_sqft": 700.0,
            "metal_roof_confidence": 0.85,
        }
        cols = _build_dominant_columns(comps, recalc, "us", "roof_material")
        assert cols["dominant_roof_material_feature_class"] == "metal-id"
        assert cols["dominant_roof_material_description"] == "metal_roof"
        assert cols["dominant_roof_material_ratio"] == 0.7
        assert cols["dominant_roof_material_area_sqft"] == 700.0
        assert cols["dominant_roof_material_confidence"] == 0.85

    def test_material_field_order(self):
        """Material fields: feature_class, description, area, ratio, confidence."""
        comps = [_mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-id")]
        cols = _build_dominant_columns(comps, None, "us", "roof_material")
        assert list(cols.keys()) == [
            "dominant_roof_material_feature_class",
            "dominant_roof_material_description",
            "dominant_roof_material_area_sqft",
            "dominant_roof_material_ratio",
            "dominant_roof_material_confidence",
        ]

    def test_shape_field_order(self):
        """Shape fields: feature_class, description, area, confidence (no ratio)."""
        comps = [_shape("Hip", 80, 0.75, 0.6, class_id="hip-id")]
        cols = _build_dominant_columns(comps, None, "us", "roof_types")
        assert list(cols.keys()) == [
            "dominant_roof_types_feature_class",
            "dominant_roof_types_description",
            "dominant_roof_types_area_sqft",
            "dominant_roof_types_confidence",
        ]


class TestFlattenRoofDominantColumns:
    def test_material_columns_emitted(self):
        roof = _make_roof(
            material_components=[
                _mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-uuid"),
                _mat("Metal Roof", 0, 1.0, 0.0, class_id="metal-uuid"),
            ],
            shape_components=[
                _shape("Hip", 80, 0.75, 0.6, class_id="hip-uuid"),
                _shape("Gable", 20, 0.6, 0.15, class_id="gable-uuid"),
            ],
        )
        result = flatten_roof_attributes([roof], country="us")
        assert result["dominant_roof_material_feature_class"] == "tile-uuid"
        assert result["dominant_roof_material_description"] == "tile_roof"
        assert result["dominant_roof_material_confidence"] == 0.9
        assert result["dominant_roof_material_area_sqft"] == pytest.approx(100 * 10.764)
        assert result["dominant_roof_material_ratio"] == 0.8
        assert result["dominant_roof_types_feature_class"] == "hip-uuid"
        assert result["dominant_roof_types_description"] == "hip"
        assert result["dominant_roof_types_confidence"] == 0.75
        # Shape should NOT have ratio
        assert "dominant_roof_types_ratio" not in result

    def test_material_unknown_when_ratio_below_threshold(self):
        roof = _make_roof(
            material_components=[
                _mat("Tile Roof", 40, 0.9, 0.4, class_id="tile-uuid"),
                _mat("Metal Roof", 30, 0.7, 0.3, class_id="metal-uuid"),
            ],
            shape_components=[],
        )
        result = flatten_roof_attributes([roof], country="us")
        assert result["dominant_roof_material_feature_class"] is None
        assert result["dominant_roof_material_description"] == "unknown"
        assert result["dominant_roof_material_confidence"] is None
        assert result["dominant_roof_material_ratio"] is None
        assert result["dominant_roof_material_area_sqft"] is None

    def test_dominant_columns_absent_when_no_material_or_shape_data(self):
        """Dominant columns are not emitted when the roof has no material/shape attributes."""
        roof = {"attributes": []}
        result = flatten_roof_attributes([roof], country="us")
        assert "dominant_roof_material_feature_class" not in result
        assert "dominant_roof_types_feature_class" not in result

    def test_metric_country_uses_sqm(self):
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-uuid")],
            shape_components=[_shape("Hip", 80, 0.75, 0.6, class_id="hip-uuid")],
        )
        result = flatten_roof_attributes([roof], country="au")
        assert result["dominant_roof_material_area_sqm"] == 100
        assert "dominant_roof_material_area_sqft" not in result
        assert result["dominant_roof_types_area_sqm"] == 80
        assert "dominant_roof_types_area_sqft" not in result

    def test_always_emitted_when_data_present(self):
        """Dominant columns are always emitted when material/shape data is present."""
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-uuid")],
            shape_components=[],
        )
        result = flatten_roof_attributes([roof], country="us")
        assert "dominant_roof_material_feature_class" in result

    def test_material_at_exact_threshold(self):
        """Ratio of exactly 0.5 should NOT be UNKNOWN."""
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 50, 0.9, 0.5, class_id="tile-uuid")],
            shape_components=[],
        )
        result = flatten_roof_attributes([roof], country="us")
        assert result["dominant_roof_material_feature_class"] == "tile-uuid"
        assert result["dominant_roof_material_description"] == "tile_roof"

    def test_shape_area_emitted(self):
        roof = _make_roof(
            material_components=[],
            shape_components=[_shape("Hip", 80, 0.75, 0.6, class_id="hip-uuid")],
        )
        result = flatten_roof_attributes([roof], country="us")
        assert result["dominant_roof_types_area_sqft"] == pytest.approx(80 * 10.764)

    def test_shape_unknown_when_all_zero_area(self):
        """If all shape components have zero area, dominant shape is UNKNOWN with nulled stats."""
        roof = _make_roof(
            material_components=[],
            shape_components=[
                _shape("Hip", 0, 0.75, 0.0, class_id="hip-uuid"),
                _shape("Gable", 0, 0.6, 0.0, class_id="gable-uuid"),
            ],
        )
        result = flatten_roof_attributes([roof], country="us")
        assert result["dominant_roof_types_feature_class"] is None
        assert result["dominant_roof_types_description"] == "unknown"
        assert result["dominant_roof_types_area_sqft"] is None
        assert result["dominant_roof_types_confidence"] is None

    def test_dominant_columns_before_constituents(self):
        """Dominant summary columns must appear before per-component columns."""
        roof = _make_roof(
            material_components=[
                _mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-uuid"),
                _mat("Metal Roof", 0, 1.0, 0.0, class_id="metal-uuid"),
            ],
            shape_components=[
                _shape("Hip", 80, 0.75, 0.6, class_id="hip-uuid"),
            ],
        )
        result = flatten_roof_attributes([roof], country="us")
        keys = list(result.keys())
        first_component = min(keys.index(k) for k in keys if k.startswith(("tile_", "metal_", "hip_")))
        dominant_keys = [k for k in keys if k.startswith("dominant_")]
        assert dominant_keys, "Expected dominant columns"
        last_dominant = max(keys.index(k) for k in dominant_keys)
        assert last_dominant < first_component

    def test_dominant_field_order_material(self):
        """Material fields: feature_class, description, area, ratio, confidence."""
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-uuid")],
            shape_components=[],
        )
        result = flatten_roof_attributes([roof], country="us")
        mat_keys = [k for k in result.keys() if k.startswith("dominant_roof_material_")]
        assert mat_keys == [
            "dominant_roof_material_feature_class",
            "dominant_roof_material_description",
            "dominant_roof_material_area_sqft",
            "dominant_roof_material_ratio",
            "dominant_roof_material_confidence",
        ]

    def test_dominant_field_order_shape(self):
        """Shape fields: feature_class, description, area, confidence (no ratio)."""
        roof = _make_roof(
            material_components=[],
            shape_components=[_shape("Hip", 80, 0.75, 0.6, class_id="hip-uuid")],
        )
        result = flatten_roof_attributes([roof], country="us")
        shape_keys = [k for k in result.keys() if k.startswith("dominant_roof_types_")]
        assert shape_keys == [
            "dominant_roof_types_feature_class",
            "dominant_roof_types_description",
            "dominant_roof_types_area_sqft",
            "dominant_roof_types_confidence",
        ]
