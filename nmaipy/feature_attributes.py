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
import ast
import json
from typing import Any, Dict, List, Optional, Union, overload

from dateutil.parser import parse as parse_date
import geopandas as gpd
import pandas as pd

from nmaipy import log
from nmaipy.constants import (
    IMPERIAL_COUNTRIES,
    METERS_TO_FEET,
    ROOF_AGE_AFTER_INSTALLATION_CAPTURE_DATE_FIELD,
    ROOF_AGE_AREA_FIELD,
    ROOF_AGE_AS_OF_DATE_FIELD,
    ROOF_AGE_ASSESSOR_DATA_FIELD,
    ROOF_AGE_BEFORE_INSTALLATION_CAPTURE_DATE_FIELD,
    ROOF_AGE_EVIDENCE_TYPE_DESC_FIELD,
    ROOF_AGE_EVIDENCE_TYPE_FIELD,
    ROOF_AGE_INSTALLATION_DATE_FIELD,
    ROOF_AGE_KIND_FIELD,
    ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD,
    ROOF_AGE_MAX_CAPTURE_DATE_FIELD,
    ROOF_AGE_MODEL_VERSION_OUTPUT_FIELD,
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


def _parse_include_param(val):
    """
    Parse an include parameter value, handling both dict and JSON string formats.

    Include parameters like roofSpotlightIndex, hurricaneScore, and defensibleSpace
    are returned as dicts from the API, but may be JSON-serialized to strings when
    stored in Parquet files. This function handles both cases.

    Args:
        val: The value to parse - can be dict, JSON string, None, or NaN

    Returns:
        dict if successfully parsed, None otherwise
    """
    if val is None:
        return None
    # Handle pandas NaN values
    if isinstance(val, float):
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _get_feature_value(feature, key):
    """
    Get a value from a feature, handling both dict and pandas Series.

    Args:
        feature: dict or pandas Series
        key: The key to look up

    Returns:
        The value, or None if not found
    """
    if isinstance(feature, dict):
        return feature.get(key)
    elif isinstance(feature, pd.Series):
        if key in feature.index:
            val = feature.get(key)
            # Handle NaN values
            if pd.isna(val):
                return None
            return val
    return None


@overload
def calculate_roof_age_years(
    installation_date: pd.Series,
    as_of_date: pd.Series,
) -> pd.Series: ...


@overload
def calculate_roof_age_years(
    installation_date: str,
    as_of_date: str,
) -> float: ...


@overload
def calculate_roof_age_years(
    installation_date: None,
    as_of_date: Any,
) -> None: ...


def calculate_roof_age_years(
    installation_date: Union[pd.Series, str, None],
    as_of_date: Union[pd.Series, str, None],
) -> Union[pd.Series, float, None]:
    """
    Calculate roof age in years from installation date to as-of date.

    Uses 365.25 days/year (accounting for leap years).
    Returns rounded to 1 decimal place.

    Handles both scalar values (str/datetime) and pandas Series.

    Args:
        installation_date: Installation date(s) - can be string, datetime, or pandas Series
        as_of_date: As-of date(s) for age calculation - can be string, datetime, or pandas Series

    Returns:
        Age in years (float for scalar input, Series for Series input), or None if calculation fails
    """
    if installation_date is None or as_of_date is None:
        return None

    try:
        if isinstance(installation_date, pd.Series):
            # Vectorized calculation for pandas Series
            install = pd.to_datetime(installation_date, errors="coerce")
            as_of = pd.to_datetime(as_of_date, errors="coerce")
            return ((as_of - install).dt.days / 365.25).round(1)
        else:
            # Scalar calculation
            install = parse_date(str(installation_date))
            as_of = parse_date(str(as_of_date))
            years = (as_of - install).days / 365.25
            return round(years, 1)
    except Exception:
        return None


def _reconstruct_attributes_from_dot_notation(feature) -> list:
    """
    Reconstruct attributes list from dot-notation columns.

    When features are processed through exporter.py's process_chunk(), the original
    'attributes' array is transformed into dot-notation columns (e.g., 'Roof material.components')
    and then the original 'attributes' column is dropped. This function reconstructs
    the attributes list from those dot-notation columns.

    This is fully data-driven - it dynamically discovers .components columns rather
    than using a hardcoded list.

    Args:
        feature: dict or pandas Series representing a feature row

    Returns:
        List of attribute dicts in the original format expected by flatten_roof_attributes
    """
    attributes = []

    # Get all column/key names from the feature
    if isinstance(feature, pd.Series):
        columns = feature.index.tolist()
    elif isinstance(feature, dict):
        columns = list(feature.keys())
    else:
        return attributes

    # Find all .components columns dynamically
    component_cols = [c for c in columns if isinstance(c, str) and c.endswith(".components")]

    for col in component_cols:
        description = col.rsplit(".", 1)[0]  # "Roof material.components" -> "Roof material"
        value = _get_feature_value(feature, col)
        if value:
            # Parse JSON string if needed
            if isinstance(value, str):
                try:
                    components = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    continue
            else:
                components = value
            if components:
                attributes.append({"description": description, "components": components})

    # Also handle 3D attributes (has3dAttributes, pitch)
    for col in columns:
        if not isinstance(col, str):
            continue
        if ".has3dAttributes" in col or ".pitch" in col:
            description = col.rsplit(".", 1)[0]
            key = col.rsplit(".", 1)[1]
            value = _get_feature_value(feature, col)
            if value is not None:
                # Find existing attribute with same description or create new
                existing = next((a for a in attributes if a.get("description") == description), None)
                if existing:
                    existing[key] = value
                else:
                    attributes.append({"description": description, key: value})

    return attributes


def _extract_building_3d_from_attribute_list(attribute_list: list) -> dict:
    """
    Extract the 3D attribute dict from the API's list-of-dicts format.

    The Feature API returns building attributes as a list like:
      [{"description": "Building 3d attributes", "has3dAttributes": True, "height": 4.7, ...},
       {"description": "Building pitch", "available": True, "value": 3.3},
       {"description": "Ground height", "available": True, "value": 105.0}]

    This extracts the "Building 3d attributes" entry into the flat dict format
    expected by the flattening logic.
    """
    result = {}
    for attr in attribute_list:
        if not isinstance(attr, dict):
            continue
        desc = attr.get("description", "")
        if desc == "Building 3d attributes":
            for key in ("has3dAttributes", "height", "numStories", "fidelity"):
                if key in attr:
                    result[key] = attr[key]
        elif desc == "Building pitch" and attr.get("available"):
            result["pitch"] = attr.get("value")
        elif desc == "Ground height" and attr.get("available"):
            result["ground_height"] = attr.get("value")
    return result


def flatten_building_attributes(buildings: List[dict], country: str) -> dict:
    """
    Flatten building attributes from Feature API.

    Handles three formats for the attributes field:
    1. List of dicts (raw API format from get_features_gdf_bulk)
    2. Single dict (pre-extracted, e.g. in tests)
    3. Missing/None (dropped by exporter — reconstructs from dot-notation columns)

    Args:
        buildings: List of building features with attributes
        country: Country code for units (e.g. "us" for imperial, "au" for metric)

    Returns:
        Flattened dictionary with building attributes
    """
    flattened = {}
    for building in buildings:
        # Get the raw attributes value
        try:
            raw_attributes = building["attributes"]
        except (KeyError, TypeError):
            raw_attributes = None

        # Parse string if needed (e.g. from parquet JSON or CSV repr)
        if isinstance(raw_attributes, str):
            try:
                raw_attributes = json.loads(raw_attributes)
            except (json.JSONDecodeError, TypeError):
                try:
                    raw_attributes = ast.literal_eval(raw_attributes)
                except (ValueError, SyntaxError):
                    raw_attributes = None

        # Normalise to a single dict with 3D fields
        if isinstance(raw_attributes, list):
            attribute = _extract_building_3d_from_attribute_list(raw_attributes)
        elif isinstance(raw_attributes, dict):
            attribute = raw_attributes
        else:
            attribute = {}

        if not attribute:
            continue

        if "has3dAttributes" in attribute:
            flattened["has_3d_attributes"] = TRUE_STRING if attribute["has3dAttributes"] else FALSE_STRING
            if attribute["has3dAttributes"]:
                if country in IMPERIAL_COUNTRIES:
                    flattened["height_ft"] = round(attribute["height"] * METERS_TO_FEET, 1)
                else:
                    flattened["height_m"] = round(attribute["height"], 1)
                for k, v in attribute["numStories"].items():
                    flattened[f"num_storeys_{k}_confidence"] = v
        if "pitch" in attribute:
            flattened["pitch_degrees"] = round(attribute["pitch"], 2)
        if "ground_height" in attribute:
            if country in IMPERIAL_COUNTRIES:
                flattened["ground_height_ft"] = round(attribute["ground_height"] * METERS_TO_FEET, 1)
            else:
                flattened["ground_height_m"] = round(attribute["ground_height"], 1)
        if "fidelity" in attribute:
            flattened["fidelity"] = attribute["fidelity"]
    return flattened


def flatten_roof_attributes(
    roofs: List[dict],
    country: str,
    child_features: gpd.GeoDataFrame = None,
) -> dict:
    """
    Flatten roof attributes from Feature API.

    For clipped roofs (where clipped_area < unclipped_area), component attributes
    are recalculated using spatial intersection with child features if available.
    The classIds from the roof's own components are used to find matching child
    features - this is fully data-driven with no hardcoded lists.

    Note: "Includes" (RSI, hurricane, defensible space) are NOT recalculated
    as they are already computed dynamically accounting for clipping.

    Args:
        roofs: List of roof features with attributes
        country: Country code for units (e.g. "us" for imperial, "au" for metric)
        child_features: Optional GeoDataFrame of child features for clipped roof
                       recalculation. If provided, component attributes will be
                       recalculated for clipped roofs using spatial intersection.

    Returns:
        Flattened dictionary with roof attributes including material components,
        3D attributes, roof spotlight index, hurricane scores, and defensible space data.
    """
    from nmaipy.parcels import calculate_child_feature_attributes

    flattened = {}

    # Handle components and other attributes
    for roof in roofs:
        # Handle roofSpotlightIndex - check both camelCase and snake_case versions
        # Use _parse_include_param to handle both dict and JSON string formats
        rsi_raw = roof.get("roofSpotlightIndex") or roof.get("roof_spotlight_index")
        rsi_data = _parse_include_param(rsi_raw)
        if rsi_data:
            if "value" in rsi_data:
                flattened["roof_spotlight_index"] = rsi_data["value"]
            if "confidence" in rsi_data:
                flattened["roof_spotlight_index_confidence"] = rsi_data["confidence"]
            if "modelVersion" in rsi_data:
                flattened["roof_spotlight_index_model_version"] = rsi_data["modelVersion"]

        # Handle hurricaneScore - check both camelCase and snake_case versions
        # Use _parse_include_param to handle both dict and JSON string formats
        hurricane_raw = roof.get("hurricaneScore") or roof.get("hurricane_score")
        hurricane_score_data = _parse_include_param(hurricane_raw)
        if hurricane_score_data:
            if "vulnerabilityScore" in hurricane_score_data:
                flattened["hurricane_vulnerability_score"] = hurricane_score_data["vulnerabilityScore"]
            if "vulnerabilityProbability" in hurricane_score_data:
                flattened["hurricane_vulnerability_probability"] = hurricane_score_data["vulnerabilityProbability"]
            if "vulnerabilityRateFactor" in hurricane_score_data:
                flattened["hurricane_vulnerability_rate_factor"] = hurricane_score_data["vulnerabilityRateFactor"]
            # Note: modelInputFeatures are not flattened as they are too detailed for typical use cases

        # Handle windScore - check both camelCase and snake_case versions
        wind_raw = roof.get("windScore") or roof.get("wind_score")
        wind_score_data = _parse_include_param(wind_raw)
        if wind_score_data:
            if "vulnerabilityScore" in wind_score_data:
                flattened["wind_vulnerability_score"] = wind_score_data["vulnerabilityScore"]
            if "vulnerabilityProbability" in wind_score_data:
                flattened["wind_vulnerability_probability"] = wind_score_data["vulnerabilityProbability"]
            if "vulnerabilityRateFactor" in wind_score_data:
                flattened["wind_vulnerability_rate_factor"] = wind_score_data["vulnerabilityRateFactor"]
            if "riskScore" in wind_score_data:
                flattened["wind_risk_score"] = wind_score_data["riskScore"]
            if "riskRateFactor" in wind_score_data:
                flattened["wind_risk_rate_factor"] = wind_score_data["riskRateFactor"]
            if "femaAnnualWindFrequency" in wind_score_data:
                flattened["wind_fema_annual_frequency"] = wind_score_data["femaAnnualWindFrequency"]

        # Handle hailScore - check both camelCase and snake_case versions
        hail_raw = roof.get("hailScore") or roof.get("hail_score")
        hail_score_data = _parse_include_param(hail_raw)
        if hail_score_data:
            if "vulnerabilityScore" in hail_score_data:
                flattened["hail_vulnerability_score"] = hail_score_data["vulnerabilityScore"]
            if "vulnerabilityProbability" in hail_score_data:
                flattened["hail_vulnerability_probability"] = hail_score_data["vulnerabilityProbability"]
            if "vulnerabilityRateFactor" in hail_score_data:
                flattened["hail_vulnerability_rate_factor"] = hail_score_data["vulnerabilityRateFactor"]
            if "riskScore" in hail_score_data:
                flattened["hail_risk_score"] = hail_score_data["riskScore"]
            if "riskRateFactor" in hail_score_data:
                flattened["hail_risk_rate_factor"] = hail_score_data["riskRateFactor"]
            if "femaAnnualHailFrequency" in hail_score_data:
                flattened["hail_fema_annual_frequency"] = hail_score_data["femaAnnualHailFrequency"]

        # Handle wildfireScore - check both camelCase and snake_case versions
        wildfire_raw = roof.get("wildfireScore") or roof.get("wildfire_score")
        wildfire_score_data = _parse_include_param(wildfire_raw)
        if wildfire_score_data:
            if "vulnerabilityScore" in wildfire_score_data:
                flattened["wildfire_vulnerability_score"] = wildfire_score_data["vulnerabilityScore"]
            if "vulnerabilityProbability" in wildfire_score_data:
                flattened["wildfire_vulnerability_probability"] = wildfire_score_data["vulnerabilityProbability"]
            if "vulnerabilityRateFactor" in wildfire_score_data:
                flattened["wildfire_vulnerability_rate_factor"] = wildfire_score_data["vulnerabilityRateFactor"]
            if "femaAnnualWildfireFrequency" in wildfire_score_data:
                flattened["wildfire_fema_annual_frequency"] = wildfire_score_data["femaAnnualWildfireFrequency"]

        # Handle windHailRiskScore - combined wind+hail risk score
        wind_hail_raw = roof.get("windHailRiskScore") or roof.get("wind_hail_risk_score")
        wind_hail_score_data = _parse_include_param(wind_hail_raw)
        if wind_hail_score_data:
            if "riskScore" in wind_hail_score_data:
                flattened["wind_hail_risk_score"] = wind_hail_score_data["riskScore"]
            if "riskRateFactor" in wind_hail_score_data:
                flattened["wind_hail_risk_rate_factor"] = wind_hail_score_data["riskRateFactor"]

        # Handle defensibleSpace - check both camelCase and snake_case versions
        # Use _parse_include_param to handle both dict and JSON string formats
        defensible_raw = roof.get("defensibleSpace") or roof.get("defensible_space")
        defensible_space_data = _parse_include_param(defensible_raw)
        if defensible_space_data:
            zones = defensible_space_data.get("zones", [])
            # Sort zones by zoneId to ensure consistent column ordering (zone 1, 2, 3, ...)
            zones_sorted = sorted(zones, key=lambda z: z.get("zoneId", 0))
            for zone in zones_sorted:
                zone_id = zone.get("zoneId")
                if zone_id:
                    # Flatten key metrics for each zone in a specific order:
                    # 1. zone_area, 2. defensible_space_area, 3. risk_object_area, 4. coverage_ratio
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

        # Safely access attributes - may not exist if dropped during process_chunk()
        # In that case, try to reconstruct from dot-notation columns
        attributes = roof.get("attributes")
        if attributes:
            if isinstance(attributes, str):
                try:
                    attributes = json.loads(attributes)
                except (json.JSONDecodeError, TypeError):
                    attributes = []
        if not attributes:
            # Try to reconstruct from dot-notation columns (created by exporter.py's process_chunk)
            attributes = _reconstruct_attributes_from_dot_notation(roof)

        # Detect if roof is clipped by comparing clipped vs unclipped area
        # Use _get_feature_value to handle both dict and Series formats
        clipped_area = _get_feature_value(roof, "clipped_area_sqm")
        unclipped_area = _get_feature_value(roof, "unclipped_area_sqm")
        is_clipped = (
            clipped_area is not None
            and unclipped_area is not None
            and clipped_area < unclipped_area * 0.99  # 1% tolerance
        )

        for attribute in (attributes or []):
            if "components" in attribute:
                components = attribute["components"]

                # For clipped roofs with child features, recalculate component attributes
                # using spatial intersection - this is data-driven by classIds in components
                # For clipped roofs, try to recalculate component attributes using child features
                recalc_attrs = None
                if is_clipped and child_features is not None:
                    geometry = _get_feature_value(roof, "geometry") or _get_feature_value(roof, "geometry_feature")
                    if isinstance(geometry, bytes):
                        from shapely import wkb
                        geometry = wkb.loads(geometry)
                    if geometry is not None:
                        recalc_attrs = calculate_child_feature_attributes(
                            geometry, components, child_features, country
                        )
                if recalc_attrs is not None:
                    flattened.update(recalc_attrs)

                for component in components:
                    name = component["description"].lower().replace(" ", "_")
                    if "Low confidence" in attribute.get("description", ""):
                        name = f"low_conf_{name}"

                    # If this component was recalculated, only add non-area attributes
                    if recalc_attrs is not None and f"{name}_present" in recalc_attrs:
                        if "dominant" in component:
                            flattened[f"{name}_dominant"] = TRUE_STRING if component["dominant"] else FALSE_STRING
                        if "confidenceStats" in component:
                            confidence_stats = component["confidenceStats"]
                            histograms = confidence_stats.get("histograms", [])
                            for histogram in histograms:
                                bin_type = histogram.get("binType", "unknown")
                                ratios = histogram.get("ratios", [])
                                for bin_idx, ratio_value in enumerate(ratios):
                                    flattened[f"{name}_confidence_stats_{bin_type}_bin_{bin_idx}"] = ratio_value
                        continue

                    # Use original component data (unclipped, or child features not available for this component)
                    flattened[f"{name}_present"] = TRUE_STRING if component["areaSqm"] > 0 else FALSE_STRING
                    if country in IMPERIAL_COUNTRIES:
                        flattened[f"{name}_area_sqft"] = component["areaSqft"]
                    else:
                        flattened[f"{name}_area_sqm"] = component["areaSqm"]
                    flattened[f"{name}_confidence"] = component["confidence"]
                    if "dominant" in component:
                        flattened[f"{name}_dominant"] = TRUE_STRING if component["dominant"] else FALSE_STRING
                    if "ratio" in component:
                        flattened[f"{name}_ratio"] = component["ratio"]
                    if "confidenceStats" in component:
                        confidence_stats = component["confidenceStats"]
                        histograms = confidence_stats.get("histograms", [])
                        for histogram in histograms:
                            bin_type = histogram.get("binType", "unknown")
                            ratios = histogram.get("ratios", [])
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
        {'roof_age_installation_date': '2019-06', 'roof_age_trust_score': 0.85, 'roof_age_evidence_type': 1}
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
        flattened[f"{prefix}roof_age_installation_date"] = installation_date

    # As-of date (when this estimate was computed)
    # Try new API field name first (asOfDate), fall back to legacy (untilDate)
    as_of_date = get_value(ROOF_AGE_AS_OF_DATE_FIELD, "as_of_date")
    if as_of_date is None:
        as_of_date = get_value(ROOF_AGE_UNTIL_DATE_FIELD, "until_date")
    if as_of_date is not None:
        flattened[f"{prefix}roof_age_as_of_date"] = as_of_date

    # Trust score (confidence in the installation date)
    trust_score = get_value(ROOF_AGE_TRUST_SCORE_FIELD, "trust_score")
    if trust_score is not None:
        flattened[f"{prefix}roof_age_trust_score"] = trust_score

    # Note: Area fields (area_sqm, area_sqft, etc.) are NOT added here because they're
    # already handled by the standard Feature API column mapping in roof_age_api._parse_response()
    # and exporter.py. Adding them here would create duplicate columns.

    # Evidence type and description
    evidence_type = get_value(ROOF_AGE_EVIDENCE_TYPE_FIELD, "evidence_type")
    if evidence_type is not None:
        flattened[f"{prefix}roof_age_evidence_type"] = evidence_type

    evidence_desc = get_value(ROOF_AGE_EVIDENCE_TYPE_DESC_FIELD, "evidence_type_description")
    if evidence_desc is not None:
        flattened[f"{prefix}roof_age_evidence_type_description"] = evidence_desc

    # Capture date information
    before_capture = get_value(ROOF_AGE_BEFORE_INSTALLATION_CAPTURE_DATE_FIELD, "before_installation_capture_date")
    if before_capture is not None:
        flattened[f"{prefix}roof_age_before_installation_capture_date"] = before_capture

    after_capture = get_value(ROOF_AGE_AFTER_INSTALLATION_CAPTURE_DATE_FIELD, "after_installation_capture_date")
    if after_capture is not None:
        flattened[f"{prefix}roof_age_after_installation_capture_date"] = after_capture

    min_capture = get_value(ROOF_AGE_MIN_CAPTURE_DATE_FIELD, "min_capture_date")
    if min_capture is not None:
        flattened[f"{prefix}roof_age_min_capture_date"] = min_capture

    max_capture = get_value(ROOF_AGE_MAX_CAPTURE_DATE_FIELD, "max_capture_date")
    if max_capture is not None:
        flattened[f"{prefix}roof_age_max_capture_date"] = max_capture

    num_captures = get_value(ROOF_AGE_NUM_CAPTURES_FIELD, "number_of_captures")
    if num_captures is not None:
        flattened[f"{prefix}roof_age_number_of_captures"] = num_captures

    # Kind (roof type classification)
    kind = get_value(ROOF_AGE_KIND_FIELD, "kind")
    if kind is not None:
        flattened[f"{prefix}roof_age_kind"] = kind

    # Relevant permits (JSON serialized for parquet compatibility)
    relevant_permits = get_value(ROOF_AGE_RELEVANT_PERMITS_FIELD, "relevant_permits")
    if relevant_permits is not None:
        flattened[f"{prefix}roof_age_relevant_permits"] = json.dumps(relevant_permits) if isinstance(relevant_permits, (dict, list)) else relevant_permits

    # Assessor data (JSON serialized for parquet compatibility)
    assessor_data = get_value(ROOF_AGE_ASSESSOR_DATA_FIELD, "assessor_data")
    if assessor_data is not None:
        flattened[f"{prefix}roof_age_assessor_data"] = json.dumps(assessor_data) if isinstance(assessor_data, (dict, list)) else assessor_data

    # Roof Age mapbrowser URL (shows before/after comparison view)
    mapbrowser_url = get_value(ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD, ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD)
    if mapbrowser_url is not None:
        flattened[f"{prefix}{ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD}"] = mapbrowser_url

    # Roof Age model version (top-level response metadata propagated to each feature)
    model_version = get_value(ROOF_AGE_MODEL_VERSION_OUTPUT_FIELD, ROOF_AGE_MODEL_VERSION_OUTPUT_FIELD)
    if model_version is not None:
        flattened[f"{prefix}{ROOF_AGE_MODEL_VERSION_OUTPUT_FIELD}"] = model_version

    # Calculate roof age in years as of as_of_date
    age_years = calculate_roof_age_years(installation_date, as_of_date)
    if age_years is not None:
        flattened[f"{prefix}roof_age_years_as_of_date"] = age_years
    return flattened
