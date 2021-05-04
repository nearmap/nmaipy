from pathlib import Path

import pytest
from shapely.geometry import Polygon
from shapely.wkt import loads


@pytest.fixture(scope="session")
def cache_directory() -> Path:
    return Path(__file__).parent.absolute() / "cache"


@pytest.fixture(scope="session")
def sydney_aoi() -> Polygon:
    return loads(
        """
        POLYGON ((
            151.2770527676921 -33.79276755642528,
            151.2774527676922 -33.79276755642528,
            151.2774527676922 -33.79236755642528,
            151.2770527676921 -33.79236755642528,
            151.2770527676921 -33.79276755642528
        ))
    """
    )
