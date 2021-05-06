from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon
from shapely.wkt import loads

from nearmap_ai import parcels
from nearmap_ai.constants import LAT_LONG_CRS


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


@pytest.fixture(scope="session")
def parcels_gdf() -> gpd.GeoDataFrame:
    return parcels.read_from_file(Path(__file__).parent.absolute() / "data" / "test_parcels.csv")


@pytest.fixture(scope="session")
def features_gdf() -> gpd.GeoDataFrame:
    df = pd.read_csv(Path(__file__).parent.absolute() / "data" / "test_features.csv")
    return gpd.GeoDataFrame(
        df.drop("geometry", axis=1),
        geometry=df.geometry.apply(loads),
        crs=LAT_LONG_CRS,
    )
