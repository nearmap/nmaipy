# Reference code for various algorithms dealing with data from the AI Feature API

from shapely.geometry import MultiPolygon, Polygon, Point
from typing import Union

BUILDING_SMALL_MAX_AREA_SQM = 30
BUILDING_MIN_WIDTH_M = 3
EROSION_M = -0.5


def get_width(r):
    """
    Get the width of the minimum bounding rectangle of a polygon or multipolygon.
    """
    if r:
        x, y = r.exterior.coords.xy
        l1, l2 = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
        width = min(l1, l2)
        return width
    else:
        return 0


def get_widths_from_multipolygon(poly):
    """
    Get the widths of all the minimum bounding rectangles within the multipolygon or polygon, returning as a list.
    """
    if isinstance(poly, MultiPolygon):
        return [get_width(x.minimum_rotated_rectangle) for x in poly.geoms]
    else:
        return [get_width(poly.minimum_rotated_rectangle)]


def is_building_small(building_poly: Polygon) -> bool:
    """
    Check if a building polygon is small based on its area (assumes polygon in an area based coordinate system measured in metres).
    # WARNING: This requires a polygon in an area based coordinate system measured in metres.
    """
    return building_poly.area < BUILDING_SMALL_MAX_AREA_SQM


def check_in_out_belongingness_of_building(building_poly: Polygon, parcel_poly: Union[Polygon, MultiPolygon]) -> bool:
    """
    Check if a building polygon meaningfully falls within the parcel polygon, and separately whether it belongs exterior to the parcel.
    Assumes building and parcel are in an area based coordinate system measured in metres.
    """
    building_parts_within = parcel_poly.intersection(building_poly)
    building_parts_without = building_poly.difference(parcel_poly)

    max_width_in = max(get_widths_from_multipolygon(building_parts_within.buffer(EROSION_M)))
    max_width_out = max(get_widths_from_multipolygon(building_parts_without.buffer(EROSION_M)))

    belongs_in = max_width_in >= BUILDING_MIN_WIDTH_M
    belongs_out = max_width_out >= BUILDING_MIN_WIDTH_M
    return (belongs_in, belongs_out)


def is_building_multiparcel(building_poly: Polygon, parcel_poly: Union[Polygon, MultiPolygon]) -> bool:
    """
    Check if a building polygon meaningfully falls bothi within the parcel polygon, and without it.
    """
    belongs_in, belongs_out = check_in_out_belongingness_of_building(building_poly, parcel_poly)
    building_is_small = is_building_small(building_poly)
    building_is_multiparcel = (belongs_in and belongs_out) and not building_is_small
    return (building_is_multiparcel, belongs_in, belongs_out, building_is_small)

def get_building_status(building_poly: Polygon, parcel_poly: Union[Polygon, MultiPolygon]) -> str:
    """
    Get the status of a building polygon based on its relationship to the parcel polygon.
    """
    building_is_multiparcel, belongs_in, belongs_out, building_is_small = is_building_multiparcel(building_poly, parcel_poly)
    is_keep = building_is_small or (building_is_multiparcel and belongs_in) or belongs_in

    return {
        "building_keep": is_keep,
        "building_small": building_is_small,
        "building_multiparcel": building_is_multiparcel,
    }
