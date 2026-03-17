"""Tests for dominant material/shape aggregate columns."""

import pytest

from nmaipy.feature_attributes import (
    _clean_dominant_name,
    _select_dominant_material,
    _select_dominant_shape,
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


class TestCleanDominantName:
    def test_strips_roof_suffix(self):
        assert _clean_dominant_name("Tile Roof") == "tile"
        assert _clean_dominant_name("Shingle Roof") == "shingle"
        assert _clean_dominant_name("Metal Roof") == "metal"

    def test_override_flat_roof_material(self):
        assert _clean_dominant_name("Flat Roof Material") == "flat_material"

    def test_override_other_roof_shape(self):
        assert _clean_dominant_name("Other Roof Shape") == "other"

    def test_override_pvc_tpo(self):
        assert _clean_dominant_name("PVC/TPO") == "pvc_tpo"

    def test_override_mod_bit(self):
        assert _clean_dominant_name("Mod-Bit") == "mod_bit"

    def test_strips_deprecated_suffix(self):
        assert _clean_dominant_name("Flat (Deprecated)") == "flat"

    def test_simple_names_unchanged(self):
        assert _clean_dominant_name("Hip") == "hip"
        assert _clean_dominant_name("Gable") == "gable"
        assert _clean_dominant_name("TPO") == "tpo"
        assert _clean_dominant_name("EPDM") == "epdm"
        assert _clean_dominant_name("Slate") == "slate"

    def test_multi_word(self):
        assert _clean_dominant_name("Wood Shake") == "wood_shake"
        assert _clean_dominant_name("Clay Tile") == "clay_tile"
        assert _clean_dominant_name("Dutch Gable") == "dutch_gable"
        assert _clean_dominant_name("Built Up") == "built_up"
        assert _clean_dominant_name("Bowstring Truss") == "bowstring_truss"


class TestSelectDominantMaterial:
    def test_picks_highest_ratio(self):
        comps = [
            _mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-id", dominant=True),
            _mat("Metal Roof", 20, 0.7, 0.2, class_id="metal-id", dominant=False),
        ]
        winner = _select_dominant_material(comps)
        assert winner["description"] == "Tile Roof"

    def test_ignores_dominant_flag(self):
        """dominant flag is no longer used for selection — highest ratio wins."""
        comps = [
            _mat("Tile Roof", 60, 0.9, 0.4, class_id="tile-id", dominant=True),
            _mat("Shingle Roof", 80, 0.8, 0.6, class_id="shingle-id", dominant=False),
        ]
        winner = _select_dominant_material(comps)
        assert winner["description"] == "Shingle Roof"

    def test_empty_returns_none(self):
        assert _select_dominant_material([]) is None


class TestSelectDominantShape:
    def test_picks_highest_ratio(self):
        comps = [
            _shape("Hip", 48, 0.75, 0.15, class_id="hip-id"),
            _shape("Gable", 75, 0.78, 0.24, class_id="gable-id"),
            _shape("Flat", 0, 1.0, 0.0, class_id="flat-id"),
        ]
        winner = _select_dominant_shape(comps)
        assert winner["description"] == "Gable"

    def test_no_confidence_filter(self):
        """Low confidence components are still eligible."""
        comps = [
            _shape("Hip", 48, 0.3, 0.15, class_id="hip-id"),
            _shape("Gable", 75, 0.4, 0.24, class_id="gable-id"),
        ]
        winner = _select_dominant_shape(comps)
        assert winner["description"] == "Gable"

    def test_zero_area_still_eligible(self):
        """Zero-area components are still eligible."""
        comps = [
            _shape("Flat", 0, 1.0, 0.5, class_id="flat-id"),
            _shape("Hip", 48, 0.75, 0.15, class_id="hip-id"),
        ]
        winner = _select_dominant_shape(comps)
        assert winner["description"] == "Flat"

    def test_empty_returns_none(self):
        assert _select_dominant_shape([]) is None


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
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        assert result["dominant_material_feature_class"] == "tile-uuid"
        assert result["dominant_material_description"] == "tile"
        assert result["dominant_material_confidence"] == 0.9
        assert result["dominant_material_area_sqft"] == pytest.approx(100 * 10.764)
        assert result["dominant_material_ratio"] == 0.8
        assert result["dominant_shape_feature_class"] == "hip-uuid"
        assert result["dominant_shape_description"] == "hip"
        assert result["dominant_shape_confidence"] == 0.75
        # Shape should NOT have ratio
        assert "dominant_shape_ratio" not in result

    def test_material_unknown_when_ratio_below_threshold(self):
        roof = _make_roof(
            material_components=[
                _mat("Tile Roof", 40, 0.9, 0.4, class_id="tile-uuid"),
                _mat("Metal Roof", 30, 0.7, 0.3, class_id="metal-uuid"),
            ],
            shape_components=[],
        )
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        assert result["dominant_material_feature_class"] == "UNKNOWN"
        assert result["dominant_material_description"] == "unknown"
        # Confidence/ratio/area nulled out when UNKNOWN
        assert result["dominant_material_confidence"] is None
        assert result["dominant_material_ratio"] is None
        assert result["dominant_material_area_sqft"] is None

    def test_dominant_columns_absent_when_disabled(self):
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-uuid")],
            shape_components=[_shape("Hip", 80, 0.75, 0.6, class_id="hip-uuid")],
        )
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=False)
        assert "dominant_material_feature_class" not in result
        assert "dominant_shape_feature_class" not in result

    def test_metric_country_uses_sqm(self):
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-uuid")],
            shape_components=[_shape("Hip", 80, 0.75, 0.6, class_id="hip-uuid")],
        )
        result = flatten_roof_attributes([roof], country="au", include_dominant_summary=True)
        assert result["dominant_material_area_sqm"] == 100
        assert "dominant_material_area_sqft" not in result
        assert result["dominant_shape_area_sqm"] == 80
        assert "dominant_shape_area_sqft" not in result

    def test_default_is_disabled(self):
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 100, 0.9, 0.8, class_id="tile-uuid")],
            shape_components=[],
        )
        result = flatten_roof_attributes([roof], country="us")
        assert "dominant_material_feature_class" not in result

    def test_material_at_exact_threshold(self):
        """Ratio of exactly 0.5 should NOT be UNKNOWN."""
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 50, 0.9, 0.5, class_id="tile-uuid")],
            shape_components=[],
        )
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        assert result["dominant_material_feature_class"] == "tile-uuid"
        assert result["dominant_material_description"] == "tile"

    def test_shape_area_emitted(self):
        roof = _make_roof(
            material_components=[],
            shape_components=[_shape("Hip", 80, 0.75, 0.6, class_id="hip-uuid")],
        )
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        assert result["dominant_shape_area_sqft"] == pytest.approx(80 * 10.764)

    def test_shape_unknown_when_all_zero_area(self):
        """If all shape components have zero area, dominant shape is UNKNOWN with nulled stats."""
        roof = _make_roof(
            material_components=[],
            shape_components=[
                _shape("Hip", 0, 0.75, 0.0, class_id="hip-uuid"),
                _shape("Gable", 0, 0.6, 0.0, class_id="gable-uuid"),
            ],
        )
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        assert result["dominant_shape_feature_class"] == "UNKNOWN"
        assert result["dominant_shape_description"] == "unknown"
        assert result["dominant_shape_area_sqft"] is None
        assert result["dominant_shape_confidence"] is None

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
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        keys = list(result.keys())
        # All dominant_* keys must come before any per-component key
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
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        mat_keys = [k for k in result.keys() if k.startswith("dominant_material_")]
        assert mat_keys == [
            "dominant_material_feature_class",
            "dominant_material_description",
            "dominant_material_area_sqft",
            "dominant_material_ratio",
            "dominant_material_confidence",
        ]

    def test_dominant_field_order_shape(self):
        """Shape fields: feature_class, description, area, confidence (no ratio)."""
        roof = _make_roof(
            material_components=[],
            shape_components=[_shape("Hip", 80, 0.75, 0.6, class_id="hip-uuid")],
        )
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        shape_keys = [k for k in result.keys() if k.startswith("dominant_shape_")]
        assert shape_keys == [
            "dominant_shape_feature_class",
            "dominant_shape_description",
            "dominant_shape_area_sqft",
            "dominant_shape_confidence",
        ]
