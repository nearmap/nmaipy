from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, shape


FEATURES_URL = "https://api.nearmap.com/ai/features/v4/features.json"
CHAR_LIMIT = 3800

class ApiError(Enum):
    AOI_NOT_FOUND = "AOI not found"
    AOI_EXCEEDS_MAX_SIZE = "AOI exceeds maximum size"
    

def polygon2coordstring(poly):
    """
    Turn a shapely polygon into the format required by the API for a query polygon.
    """
    coords = poly.boundary.coords[:]
    flat_coords = np.array(coords).flatten()
    coordstring = ",".join(flat_coords.astype(str))
    return coordstring


def geometry2coordstring(geometry):
    """
    Turn a shapely polygon or multipolygon into a single coord sting to be used in API requests.
    To meet the contraints on the API URL the following changes may be applied:
     - Multipolygons are simplified to single polygons by taking the convex hull
     - Polygons that have too many coordinates (resulting in strings that are too long) are
       simplified by taking the convex hull.
     - Polygons that have a convex hull with too many coordinates are simplified to a box.
    If the coord string return does not represent the polygon exactly, the exact flag is set to False.
    """

    if isinstance(geometry, MultiPolygon):
        coordstring = polygon2coordstring(geometry.convex_hull)
        exact = False
    else:
        coordstring = polygon2coordstring(geometry)
        exact = True
    if len(coordstring) > CHAR_LIMIT:
        exact = False
        coordstring = polygon2coordstring(geometry.convex_hull)
    if len(coordstring) > CHAR_LIMIT:
        exact = False
        coordstring = polygon2coordstring(geometry.envelope)
    return coordstring, exact

def get_session():
    """
    Return a request session with retrying configured.
    """
    session = requests.Session()
    retries = Retry(
        total=10, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def get_features(geometry, packs, api_key, since=None, until=None):
    """
    Get data for a AOI
    """
    # Create request string
    coordstring, exact = geometry2coordstring(geometry)
    request_string = f"{FEATURES_URL}?polygon={coordstring}&packs={packs}&apikey={api_key}"
    
    # Add dates if given
    if since:
        request_string += f"&since={since}"
    if until:
        request_string += f"&until={until}"
        
    # Request data
    response = get_session().get(request_string)

    # Check for errors
    if response.status_code == 404:
        return None, ApiError.AOI_NOT_FOUND
    elif (
        response.status_code == 400
        and response.json()["code"] == "AOI_EXCEEDS_MAX_SIZE"
    ):
        return None, ApiError.AOI_EXCEEDS_MAX_SIZE
    elif not response.ok:
        # Fail hard for unexpected errors
        raise RuntimeError(
            f"\n{parcel_id=}\n\n{request_string=}\n\n{response.status_code=}\n\n{response.text}\n\n"
        )
    data = response.json()
    # If the AOI was altered for the API request, we need to filter features in the response
    if not exact:
        data["features"] = [
            f for f in data["features"] if shape(f["geometry"]).intersects(geometry)
        ]
    return data, None

