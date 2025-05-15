# Reference code for various algorithms dealing with data from the AI Feature API
# Simplified to only keep the building size function

from shapely.geometry import Polygon

# Constants
BUILDING_SMALL_MAX_AREA_SQM = 30

def is_building_small(building_poly: Polygon) -> bool:
    """
    Check if a building polygon is small based on its area
    WARNING: This requires a polygon in an area-based coordinate system measured in metres.

    Args:
        building_poly: Building polygon in a metric coordinate system

    Returns:
        True if the building is considered small (< 30 sqm)
    """
    return building_poly.area < BUILDING_SMALL_MAX_AREA_SQM
