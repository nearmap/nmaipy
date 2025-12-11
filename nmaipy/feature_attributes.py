"""
Feature Attribute Flattening Utilities

This module provides functions to flatten nested feature attributes from various
Nearmap AI APIs into flat dictionaries suitable for tabular storage (CSV, Parquet).

Each feature class has its own flattening function that understands the specific
structure of that feature type's attributes.

Feature Classes:
- Roof (Feature API): Material components, 3D attributes, roof spotlight index, etc.
- Building (Feature API): 3D attributes, height, stories
- Building Lifecycle (Feature API): Damage scores and classifications
- Roof Instance (Roof Age API): Installation dates, trust scores, evidence types
"""
import json
from typing import Dict, List, Optional, Union

import pandas as pd

from nmaipy import log
from nmaipy.constants import (
    IMPERIAL_COUNTRIES,
    METERS_TO_FEET,
    ROOF_AGE_AFTER_INSTALLATION_CAPTURE_DATE_FIELD,
    ROOF_AGE_AREA_FIELD,
    ROOF_AGE_ASSESSOR_DATA_FIELD,
    ROOF_AGE_BEFORE_INSTALLATION_CAPTURE_DATE_FIELD,
    ROOF_AGE_EVIDENCE_TYPE_DESC_FIELD,
    ROOF_AGE_EVIDENCE_TYPE_FIELD,
    ROOF_AGE_INSTALLATION_DATE_FIELD,
    ROOF_AGE_KIND_FIELD,
    ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD,
    ROOF_AGE_MAX_CAPTURE_DATE_FIELD,
    ROOF_AGE_MIN_CAPTURE_DATE_FIELD,
    ROOF_AGE_NUM_CAPTURES_FIELD,
    ROOF_AGE_RELEVANT_PERMITS_FIELD,
    ROOF_AGE_TRUST_SCORE_FIELD,
    ROOF_AGE_UNTIL_DATE_FIELD,
)

logger = log.get_logger()

# String representations for boolean values in CSV outputs
TRUE_STRING = "Y"
FALSE_STRING = "N"


def flatten_building_attributes(buildings: List[dict], country: str) -> dict:
    """
    Flatten building attributes from Feature API.

    Args:
        buildings: List of building features with attributes
        country: Country code for units (e.g. "us" for imperial, "au" for metric)

    Returns:
        Flattened dictionary with building attributes
    """
    flattened = {}
    for building in buildings:
        attribute = building["attributes"]
        if "has3dAttributes" in attribute:
            flattened["has_3d_attributes"] = TRUE_STRING if attribute["has3dAttributes"] else FALSE_STRING
            if attribute["has3dAttributes"]:
                if country in IMPERIAL_COUNTRIES:
                    flattened["height_ft"] = round(attribute["height"] * METERS_TO_FEET, 1)
                else:
                    flattened["height_m"] = round(attribute["height"], 1)
                for k, v in attribute["numStories"].items():
                    flattened[f"num_storeys_{k}_confidence"] = v
        if "fidelity" in attribute:
            flattened["fidelity"] = attribute["fidelity"]
    return flattened


def flatten_roof_attributes(roofs: List[dict], country: str) -> dict:
    """
    Flatten roof attributes from Feature API.

    Args:
        roofs: List of roof features with attributes
        country: Country code for units (e.g. "us" for imperial, "au" for metric)

    Returns:
        Flattened dictionary with roof attributes including material components,
        3D attributes, roof spotlight index, hurricane scores, and defensible space data.
    """
    flattened = {}

    # Handle components and other attributes
    for roof in roofs:
        # Handle roofSpotlightIndex - check both camelCase and snake_case versions
        rsi_data = roof.get("roofSpotlightIndex") or roof.get("roof_spotlight_index")
        if rsi_data and isinstance(rsi_data, dict):
            if "value" in rsi_data:
                flattened["roof_spotlight_index"] = rsi_data["value"]
            if "confidence" in rsi_data:
                flattened["roof_spotlight_index_confidence"] = rsi_data["confidence"]
            if "modelVersion" in rsi_data:
                flattened["roof_spotlight_index_model_version"] = rsi_data["modelVersion"]

        # Handle hurricaneScore - check both camelCase and snake_case versions
        hurricane_score_data = roof.get("hurricaneScore") or roof.get("hurricane_score")
        if hurricane_score_data and isinstance(hurricane_score_data, dict):
            if "vulnerabilityScore" in hurricane_score_data:
                flattened["hurricane_vulnerability_score"] = hurricane_score_data["vulnerabilityScore"]
            if "vulnerabilityProbability" in hurricane_score_data:
                flattened["hurricane_vulnerability_probability"] = hurricane_score_data["vulnerabilityProbability"]
            if "vulnerabilityRateFactor" in hurricane_score_data:
                flattened["hurricane_vulnerability_rate_factor"] = hurricane_score_data["vulnerabilityRateFactor"]
            # Note: modelInputFeatures are not flattened as they are too detailed for typical use cases

        # Handle defensibleSpace - check both camelCase and snake_case versions
        defensible_space_data = roof.get("defensibleSpace") or roof.get("defensible_space")
        if defensible_space_data and isinstance(defensible_space_data, dict):
            zones = defensible_space_data.get("zones", [])
            for zone in zones:
                zone_id = zone.get("zoneId")
                if zone_id:
                    # Flatten key metrics for each zone
                    prefix = f"defensible_space_zone_{zone_id}"
                    if country in IMPERIAL_COUNTRIES:
                        if "zoneAreaSqft" in zone:
                            flattened[f"{prefix}_zone_area_sqft"] = zone["zoneAreaSqft"]
                        if "defensibleSpaceAreaSqft" in zone:
                            flattened[f"{prefix}_defensible_space_area_sqft"] = zone["defensibleSpaceAreaSqft"]
                        if "totalRiskObjectAreaSqft" in zone:
                            flattened[f"{prefix}_risk_object_area_sqft"] = zone["totalRiskObjectAreaSqft"]
                    else:
                        if "zoneAreaSqm" in zone:
                            flattened[f"{prefix}_zone_area_sqm"] = zone["zoneAreaSqm"]
                        if "defensibleSpaceAreaSqm" in zone:
                            flattened[f"{prefix}_defensible_space_area_sqm"] = zone["defensibleSpaceAreaSqm"]
                        if "totalRiskObjectAreaSqm" in zone:
                            flattened[f"{prefix}_risk_object_area_sqm"] = zone["totalRiskObjectAreaSqm"]

                    if "defensibleSpaceCoverageRatio" in zone:
                        flattened[f"{prefix}_coverage_ratio"] = zone["defensibleSpaceCoverageRatio"]
                    # Note: zoneGeometry and individual riskObjects are not flattened as they are too detailed

        for attribute in roof["attributes"]:
            if "components" in attribute:
                for component in attribute["components"]:
                    name = component["description"].lower().replace(" ", "_")
                    if "Low confidence" in attribute["description"]:
                        name = f"low_conf_{name}"
                    flattened[f"{name}_present"] = TRUE_STRING if component["areaSqm"] > 0 else FALSE_STRING
                    if country in IMPERIAL_COUNTRIES:
                        flattened[f"{name}_area_sqft"] = component["areaSqft"]
                    else:
                        flattened[f"{name}_area_sqm"] = component["areaSqm"]
                    flattened[f"{name}_confidence"] = component["confidence"]
                    if "dominant" in component:
                        flattened[f"{name}_dominant"] = TRUE_STRING if component["dominant"] else FALSE_STRING
                    # Handle ratio field if present
                    if "ratio" in component:
                        flattened[f"{name}_ratio"] = component["ratio"]

                    # Handle confidenceStats if present (from roofConditionConfidenceStats include parameter)
                    if "confidenceStats" in component:
                        confidence_stats = component["confidenceStats"]
                        # Flatten histogram bins
                        histograms = confidence_stats.get("histograms", [])
                        for histogram in histograms:
                            bin_type = histogram.get("binType", "unknown")
                            ratios = histogram.get("ratios", [])
                            # Create a column for each bin in the histogram
                            for bin_idx, ratio_value in enumerate(ratios):
                                flattened[f"{name}_confidence_stats_{bin_type}_bin_{bin_idx}"] = ratio_value
            elif "has3dAttributes" in attribute:
                flattened["has_3d_attributes"] = TRUE_STRING if attribute["has3dAttributes"] else FALSE_STRING
                if attribute["has3dAttributes"]:
                    flattened["pitch_degrees"] = attribute["pitch"]
    return flattened


def flatten_building_lifecycle_damage_attributes(building_lifecycles: List[dict]) -> dict:
    """
    Flatten building lifecycle damage attributes from Feature API.

    Args:
        building_lifecycles: List of building lifecycle features with damage field

    Returns:
        Dictionary with flattened damage attributes including confidence scores
        for damage classes and damage ratios.
    """
    flattened = {}

    for building_lifecycle in building_lifecycles:
        # Get damage data from top-level damage field
        damage_data = building_lifecycle.get("damage")

        if damage_data is None or not isinstance(damage_data, dict):
            continue

        # Extract confidences
        confidences = damage_data.get("confidences")
        if not isinstance(confidences, dict):
            continue

        # Process raw confidence scores (5 classes: Undamaged, Affected, Minor, Major, Destroyed)
        raw_confidences = confidences.get("raw")
        if isinstance(raw_confidences, dict) and len(raw_confidences) > 0:
            x = pd.Series(raw_confidences)
            flattened["damage_class"] = x.idxmax()
            flattened["damage_class_confidence"] = x.max()
            for damage_class, confidence in raw_confidences.items():
                flattened[f"damage_class_{damage_class}_confidence"] = confidence

        # Process 2tier confidences (UndamagedOrAffectedOrMinor vs MajorOrDestroyed)
        tier2_confidences = confidences.get("2tier")
        if isinstance(tier2_confidences, dict):
            for tier2_class, confidence in tier2_confidences.items():
                flattened[f"damage_2tier_{tier2_class}_confidence"] = confidence

        # Process damage ratios (specific damage indicators)
        ratios = damage_data.get("ratios")
        if isinstance(ratios, list):
            for ratio_item in ratios:
                if isinstance(ratio_item, dict):
                    description = ratio_item.get("description")
                    ratio_value = ratio_item.get("ratioAbove50PctConf")
                    if description is not None and ratio_value is not None:
                        # Normalize description to valid column name
                        normalized_desc = description.lower().replace(" ", "_")
                        flattened[f"damage_ratio_{normalized_desc}"] = ratio_value

    return flattened


def flatten_roof_instance_attributes(
    roof_instance: Union[dict, pd.Series],
    country: str,
    prefix: str = "",
) -> dict:
    """
    Flatten roof instance attributes from Roof Age API.

    Roof instances are temporal slices of roofs with installation date information.
    This function extracts Roof Age specific fields like installation date, trust score,
    and evidence type. It excludes internal fields (timeline, hilbertId, resourceId).

    Note: Area fields (area_sqm, area_sqft, etc.) are NOT added here because they're
    already handled by the standard Feature API column mapping in roof_age_api._parse_response()
    and exporter.py.

    Args:
        roof_instance: A roof instance feature (dict or pandas Series)
        country: Country code for units (e.g. "us" for imperial)
        prefix: Optional prefix for flattened keys (e.g., "primary_roof_instance_")

    Returns:
        Flattened dictionary with roof instance attributes

    Example:
        >>> instance = {"installationDate": "2019-06", "trustScore": 0.85, "evidenceType": 1}
        >>> attrs = flatten_roof_instance_attributes(instance, country="us")
        >>> print(attrs)
        {'installation_date': '2019-06', 'trust_score': 0.85, 'evidence_type': 1}
    """
    flattened = {}

    # Helper to get value from dict or Series, checking both camelCase and snake_case
    def get_value(camel_key, snake_key=None):
        """Get value checking both camelCase (API default) and snake_case (potential conversion)."""
        if isinstance(roof_instance, dict):
            val = roof_instance.get(camel_key)
            if val is None and snake_key:
                val = roof_instance.get(snake_key)
            return val
        elif isinstance(roof_instance, pd.Series):
            if camel_key in roof_instance.index:
                return roof_instance.get(camel_key)
            elif snake_key and snake_key in roof_instance.index:
                return roof_instance.get(snake_key)
        return None

    # Installation date
    installation_date = get_value(ROOF_AGE_INSTALLATION_DATE_FIELD, "installation_date")
    if installation_date is not None:
        flattened[f"{prefix}installation_date"] = installation_date

    # Until date (when this estimate is valid until)
    until_date = get_value(ROOF_AGE_UNTIL_DATE_FIELD, "until_date")
    if until_date is not None:
        flattened[f"{prefix}until_date"] = until_date

    # Trust score (confidence in the installation date)
    trust_score = get_value(ROOF_AGE_TRUST_SCORE_FIELD, "trust_score")
    if trust_score is not None:
        flattened[f"{prefix}trust_score"] = trust_score

    # Note: Area fields (area_sqm, area_sqft, etc.) are NOT added here because they're
    # already handled by the standard Feature API column mapping in roof_age_api._parse_response()
    # and exporter.py. Adding them here would create duplicate columns.

    # Evidence type and description
    evidence_type = get_value(ROOF_AGE_EVIDENCE_TYPE_FIELD, "evidence_type")
    if evidence_type is not None:
        flattened[f"{prefix}evidence_type"] = evidence_type

    evidence_desc = get_value(ROOF_AGE_EVIDENCE_TYPE_DESC_FIELD, "evidence_type_description")
    if evidence_desc is not None:
        flattened[f"{prefix}evidence_type_description"] = evidence_desc

    # Capture date information
    before_capture = get_value(ROOF_AGE_BEFORE_INSTALLATION_CAPTURE_DATE_FIELD, "before_installation_capture_date")
    if before_capture is not None:
        flattened[f"{prefix}before_installation_capture_date"] = before_capture

    after_capture = get_value(ROOF_AGE_AFTER_INSTALLATION_CAPTURE_DATE_FIELD, "after_installation_capture_date")
    if after_capture is not None:
        flattened[f"{prefix}after_installation_capture_date"] = after_capture

    min_capture = get_value(ROOF_AGE_MIN_CAPTURE_DATE_FIELD, "min_capture_date")
    if min_capture is not None:
        flattened[f"{prefix}min_capture_date"] = min_capture

    max_capture = get_value(ROOF_AGE_MAX_CAPTURE_DATE_FIELD, "max_capture_date")
    if max_capture is not None:
        flattened[f"{prefix}max_capture_date"] = max_capture

    num_captures = get_value(ROOF_AGE_NUM_CAPTURES_FIELD, "number_of_captures")
    if num_captures is not None:
        flattened[f"{prefix}number_of_captures"] = num_captures

    # Kind (roof type classification)
    kind = get_value(ROOF_AGE_KIND_FIELD, "kind")
    if kind is not None:
        flattened[f"{prefix}kind"] = kind

    # Relevant permits (JSON serialized for parquet compatibility)
    relevant_permits = get_value(ROOF_AGE_RELEVANT_PERMITS_FIELD, "relevant_permits")
    if relevant_permits is not None:
        flattened[f"{prefix}relevant_permits"] = json.dumps(relevant_permits) if isinstance(relevant_permits, (dict, list)) else relevant_permits

    # Assessor data (JSON serialized for parquet compatibility)
    assessor_data = get_value(ROOF_AGE_ASSESSOR_DATA_FIELD, "assessor_data")
    if assessor_data is not None:
        flattened[f"{prefix}assessor_data"] = json.dumps(assessor_data) if isinstance(assessor_data, (dict, list)) else assessor_data

    # Roof Age mapbrowser URL (shows before/after comparison view)
    mapbrowser_url = get_value(ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD, ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD)
    if mapbrowser_url is not None:
        flattened[f"{prefix}{ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD}"] = mapbrowser_url

    # Note: We intentionally exclude internal fields:
    # - timeline (detailed internal data, not public)
    # - hilbertId (internal spatial indexing)
    # - resourceId (internal reference)

    return flattened
