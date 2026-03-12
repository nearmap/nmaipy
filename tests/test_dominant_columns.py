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


def _mat(desc, area, confidence, ratio, dominant=False):
    return {
        "description": desc,
        "areaSqm": area,
        "areaSqft": area * 10.764,
        "confidence": confidence,
        "ratio": ratio,
        "dominant": dominant,
    }


def _shape(desc, area, confidence, ratio):
    return {
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
    def test_single_dominant(self):
        comps = [
            _mat("Tile Roof", 100, 0.9, 0.8, dominant=True),
            _mat("Metal Roof", 20, 0.7, 0.2, dominant=False),
        ]
        winner = _select_dominant_material(comps)
        assert winner["description"] == "Tile Roof"

    def test_multiple_dominant_picks_highest_ratio(self):
        comps = [
            _mat("Tile Roof", 60, 0.9, 0.4, dominant=True),
            _mat("Shingle Roof", 80, 0.8, 0.6, dominant=True),
        ]
        winner = _select_dominant_material(comps)
        assert winner["description"] == "Shingle Roof"

    def test_none_dominant_returns_none(self):
        comps = [
            _mat("Tile Roof", 100, 0.9, 0.8, dominant=False),
            _mat("Metal Roof", 20, 0.7, 0.2, dominant=False),
        ]
        assert _select_dominant_material(comps) is None


class TestSelectDominantShape:
    def test_highest_ratio_above_threshold(self):
        comps = [
            _shape("Hip", 48, 0.75, 0.15),
            _shape("Gable", 75, 0.78, 0.24),
            _shape("Flat", 0, 1.0, 0.0),
        ]
        winner = _select_dominant_shape(comps)
        assert winner["description"] == "Gable"

    def test_all_below_confidence_returns_none(self):
        comps = [
            _shape("Hip", 48, 0.3, 0.15),
            _shape("Gable", 75, 0.4, 0.24),
        ]
        assert _select_dominant_shape(comps) is None

    def test_zero_area_excluded(self):
        comps = [
            _shape("Flat", 0, 1.0, 0.0),
            _shape("Hip", 48, 0.75, 0.15),
        ]
        winner = _select_dominant_shape(comps)
        assert winner["description"] == "Hip"


class TestFlattenRoofDominantColumns:
    def test_dominant_columns_emitted_when_enabled(self):
        roof = _make_roof(
            material_components=[
                _mat("Tile Roof", 100, 0.9, 0.8, dominant=True),
                _mat("Metal Roof", 0, 1.0, 0.0, dominant=False),
            ],
            shape_components=[
                _shape("Hip", 80, 0.75, 0.6),
                _shape("Gable", 20, 0.6, 0.15),
            ],
        )
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        assert result["dominant_material"] == "tile"
        assert result["dominant_material_confidence"] == 0.9
        assert result["dominant_material_area_sqft"] == pytest.approx(100 * 10.764)
        assert result["dominant_material_ratio"] == 0.8
        assert result["dominant_shape"] == "hip"
        assert result["dominant_shape_confidence"] == 0.75
        assert result["dominant_shape_ratio"] == 0.6

    def test_dominant_columns_absent_when_disabled(self):
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 100, 0.9, 0.8, dominant=True)],
            shape_components=[_shape("Hip", 80, 0.75, 0.6)],
        )
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=False)
        assert "dominant_material" not in result
        assert "dominant_shape" not in result

    def test_no_dominant_material_when_none_flagged(self):
        roof = _make_roof(
            material_components=[
                _mat("Tile Roof", 100, 0.9, 0.8, dominant=False),
            ],
            shape_components=[_shape("Hip", 80, 0.75, 0.6)],
        )
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        assert "dominant_material" not in result
        assert result["dominant_shape"] == "hip"

    def test_no_dominant_shape_when_all_low_confidence(self):
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 100, 0.9, 0.8, dominant=True)],
            shape_components=[_shape("Hip", 80, 0.3, 0.6)],
        )
        result = flatten_roof_attributes([roof], country="us", include_dominant_summary=True)
        assert result["dominant_material"] == "tile"
        assert "dominant_shape" not in result

    def test_metric_country_uses_sqm(self):
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 100, 0.9, 0.8, dominant=True)],
            shape_components=[_shape("Hip", 80, 0.75, 0.6)],
        )
        result = flatten_roof_attributes([roof], country="au", include_dominant_summary=True)
        assert result["dominant_material_area_sqm"] == 100
        assert "dominant_material_area_sqft" not in result
        assert result["dominant_shape_area_sqm"] == 80
        assert "dominant_shape_area_sqft" not in result

    def test_default_is_disabled(self):
        roof = _make_roof(
            material_components=[_mat("Tile Roof", 100, 0.9, 0.8, dominant=True)],
            shape_components=[],
        )
        result = flatten_roof_attributes([roof], country="us")
        assert "dominant_material" not in result
