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
def large_adelaide_aoi() -> Polygon:
    """
    MB_CODE_2021
    40215247200
    """
    return loads(
        """
    POLYGON ((
        138.58494782597298922 -35.0916037690875342, 138.58494023597955902 -35.09151746908665359,
        138.58116021657605188 -35.09172645893046649, 138.58108491658788353 -35.09173058892735497,
        138.5811483565279616 -35.09253722893505056, 138.58095789655803287 -35.09254724892718258,
        138.5809769565215106 -35.09308979893137348, 138.58058428655178318 -35.0936245989183746,
        138.57899145659069973 -35.097158618874154, 138.57914994644420403 -35.09912024889305826,
        138.57638729688110857 -35.09925901877887355, 138.57623720701226944 -35.09752446876174758,
        138.57277963755808514 -35.09771424861899192, 138.57266973757572259 -35.09771598861443209,
        138.57265323759003195 -35.09752646861255698, 138.56935622811258213 -35.09767446847625649,
        138.56868422821455056 -35.09777847844893728, 138.56863122822099399 -35.09781245844695263,
        138.57053262760541656 -35.10282668855750643, 138.57143654745979688 -35.10282444859510065,
        138.57308389719975139 -35.10273262866309807, 138.58576742519736058 -35.10203206918703955,
        138.58558022536499266 -35.09980492916529471, 138.58535924558324837 -35.09684546913752712,
        138.58527102567015277 -35.09566727912645945, 138.58494782597298922 -35.0916037690875342
    ))
    """
    )


@pytest.fixture(scope="session")
def data_directory() -> Path:
    return Path(__file__).parent.absolute() / "data"


@pytest.fixture(scope="session")
def parcels_gdf(data_directory: Path) -> gpd.GeoDataFrame:
    return parcels.read_from_file(data_directory / "test_parcels.csv")


@pytest.fixture(scope="session")
def features_gdf(data_directory: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(data_directory / "test_features.csv")
    return gpd.GeoDataFrame(
        df.drop("geometry", axis=1),
        geometry=df.geometry.apply(loads),
        crs=LAT_LONG_CRS,
    )
