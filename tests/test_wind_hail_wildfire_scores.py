#!/usr/bin/env python
"""Test that windScore, hailScore, wildfireScore, and windHailRiskScore are correctly handled."""

import json
from pathlib import Path

import pytest

from nmaipy.constants import ROOF_ID
from nmaipy.parcels import flatten_roof_attributes

data_directory = Path(__file__).parent / "data"


# Sample data matching API response structure
SAMPLE_WIND_SCORE = {
    "vulnerabilityScore": 5,
    "vulnerabilityProbability": 0.029,
    "vulnerabilityRateFactor": 0.56,
    "riskScore": 3,
    "riskRateFactor": 0.83,
    "femaAnnualWindFrequency": 5.125,
    "femaVersion": "1.18.0",
    "modelInputFeatures": {
        "pitchedRoofPresent": 1,
        "metalPresent": 0,
    }
}

SAMPLE_HAIL_SCORE = {
    "vulnerabilityScore": 5,
    "vulnerabilityProbability": 0.048,
    "vulnerabilityRateFactor": 0.63,
    "riskScore": 4,
    "riskRateFactor": 0.53,
    "femaAnnualHailFrequency": 4.125,
    "femaVersion": "1.18.0",
    "modelInputFeatures": {
        "flatPresent": 0,
        "shinglePresent": 1,
    }
}

SAMPLE_WILDFIRE_SCORE = {
    "vulnerabilityScore": 3,
    "vulnerabilityProbability": 0.919,
    "vulnerabilityRateFactor": 1.06,
    "femaAnnualWildfireFrequency": 0,
    "femaVersion": "1.18.0",
    "modelInputFeatures": {
        "fireResistantMaterialPresent": 0,
        "roofDebrisRatio": 0,
    }
}

SAMPLE_WIND_HAIL_RISK_SCORE = {
    "riskScore": 4,
    "riskRateFactor": 0.58,
    "modelInputFeatures": {
        "windRiskScore": 3,
        "hailRiskScore": 4
    }
}


class TestWindScore:
    """Tests for windScore flattening."""

    def test_wind_score_extraction(self):
        """Test that windScore fields are correctly extracted."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'windScore': SAMPLE_WIND_SCORE,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")

        assert result["wind_vulnerability_score"] == 5
        assert result["wind_vulnerability_probability"] == 0.029
        assert result["wind_vulnerability_rate_factor"] == 0.56
        assert result["wind_risk_score"] == 3
        assert result["wind_risk_rate_factor"] == 0.83
        assert result["wind_fema_annual_frequency"] == 5.125

    def test_wind_score_snake_case(self):
        """Test that snake_case field name works."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'wind_score': SAMPLE_WIND_SCORE,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")
        assert result["wind_vulnerability_score"] == 5

    def test_wind_score_missing(self):
        """Test that missing windScore is handled gracefully."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")
        assert "wind_vulnerability_score" not in result
        assert "wind_risk_score" not in result


class TestHailScore:
    """Tests for hailScore flattening."""

    def test_hail_score_extraction(self):
        """Test that hailScore fields are correctly extracted."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'hailScore': SAMPLE_HAIL_SCORE,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")

        assert result["hail_vulnerability_score"] == 5
        assert result["hail_vulnerability_probability"] == 0.048
        assert result["hail_vulnerability_rate_factor"] == 0.63
        assert result["hail_risk_score"] == 4
        assert result["hail_risk_rate_factor"] == 0.53
        assert result["hail_fema_annual_frequency"] == 4.125

    def test_hail_score_snake_case(self):
        """Test that snake_case field name works."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'hail_score': SAMPLE_HAIL_SCORE,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")
        assert result["hail_vulnerability_score"] == 5

    def test_hail_score_missing(self):
        """Test that missing hailScore is handled gracefully."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")
        assert "hail_vulnerability_score" not in result
        assert "hail_risk_score" not in result


class TestWildfireScore:
    """Tests for wildfireScore flattening."""

    def test_wildfire_score_extraction(self):
        """Test that wildfireScore fields are correctly extracted."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'wildfireScore': SAMPLE_WILDFIRE_SCORE,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")

        assert result["wildfire_vulnerability_score"] == 3
        assert result["wildfire_vulnerability_probability"] == 0.919
        assert result["wildfire_vulnerability_rate_factor"] == 1.06
        assert result["wildfire_fema_annual_frequency"] == 0

    def test_wildfire_score_snake_case(self):
        """Test that snake_case field name works."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'wildfire_score': SAMPLE_WILDFIRE_SCORE,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")
        assert result["wildfire_vulnerability_score"] == 3

    def test_wildfire_score_missing(self):
        """Test that missing wildfireScore is handled gracefully."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")
        assert "wildfire_vulnerability_score" not in result


class TestWindHailRiskScore:
    """Tests for windHailRiskScore flattening."""

    def test_wind_hail_risk_score_extraction(self):
        """Test that windHailRiskScore fields are correctly extracted."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'windHailRiskScore': SAMPLE_WIND_HAIL_RISK_SCORE,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")

        assert result["wind_hail_risk_score"] == 4
        assert result["wind_hail_risk_rate_factor"] == 0.58

    def test_wind_hail_risk_score_snake_case(self):
        """Test that snake_case field name works."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'wind_hail_risk_score': SAMPLE_WIND_HAIL_RISK_SCORE,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")
        assert result["wind_hail_risk_score"] == 4

    def test_wind_hail_risk_score_missing(self):
        """Test that missing windHailRiskScore is handled gracefully."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")
        assert "wind_hail_risk_score" not in result


class TestAllScoresTogether:
    """Tests for handling multiple scores on the same roof."""

    def test_all_scores_together(self):
        """Test that all score types can coexist on the same roof."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'windScore': SAMPLE_WIND_SCORE,
            'hailScore': SAMPLE_HAIL_SCORE,
            'wildfireScore': SAMPLE_WILDFIRE_SCORE,
            'windHailRiskScore': SAMPLE_WIND_HAIL_RISK_SCORE,
            'hurricaneScore': {
                "vulnerabilityScore": 4,
                "vulnerabilityProbability": 0.5,
                "vulnerabilityRateFactor": 1.0
            },
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")

        # Wind
        assert result["wind_vulnerability_score"] == 5
        assert result["wind_risk_score"] == 3

        # Hail
        assert result["hail_vulnerability_score"] == 5
        assert result["hail_risk_score"] == 4

        # Wildfire
        assert result["wildfire_vulnerability_score"] == 3

        # Wind+Hail combined
        assert result["wind_hail_risk_score"] == 4

        # Hurricane (existing)
        assert result["hurricane_vulnerability_score"] == 4

    def test_json_string_format(self):
        """Test that JSON string format (from parquet deserialization) works."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'windScore': json.dumps(SAMPLE_WIND_SCORE),
            'hailScore': json.dumps(SAMPLE_HAIL_SCORE),
            'attributes': []
        }

        result = flatten_roof_attributes([roof], country="us")

        assert result["wind_vulnerability_score"] == 5
        assert result["hail_vulnerability_score"] == 5

    def test_invalid_score_data(self):
        """Test that invalid score data is handled gracefully."""
        roof = {
            'feature_id': 'test_roof',
            'class_id': ROOF_ID,
            'windScore': "not a valid json",
            'hailScore': None,
            'wildfireScore': 123,  # Should be dict
            'attributes': []
        }

        # Should not crash
        result = flatten_roof_attributes([roof], country="us")

        assert "wind_vulnerability_score" not in result
        assert "hail_vulnerability_score" not in result
        assert "wildfire_vulnerability_score" not in result
