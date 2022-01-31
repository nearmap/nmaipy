import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon
import warnings


from nearmap_ai.feature_api import FeatureApi
from nearmap_ai.constants import BUILDING_ID, LAWN_GRASS_ID, POOL_ID, AOI_ID_COLUMN_NAME, SURVEY_RESOURCE_ID_COL_NAME

# sys.path.append(Path(__file__).parent.parent.absolute() / "scripts")
sys.path.append("/home/jovyan/nearmap-ai-user-guides/scripts")
import ai_offline_parcel


class TestAIOfflineParcel:
    @pytest.mark.filterwarnings("ignore:.*initial implementation of Parquet.*")
    def test_process_chunk_au(
        self, parcel_gdf_au_tests: gpd.GeoDataFrame, cache_directory: Path, processed_output_directory: Path
    ):
        tag = "tests_au"
        chunk_id = 0

        output_dir = Path("/home/jovyan/data/tmp") / tag
        packs = ["building", "vegetation"]
        country = "au"
        final_path = output_dir / "final"  # Permanent path for later visual inspection
        final_path.mkdir(parents=True, exist_ok=True)

        chunk_path = output_dir / "chunks"
        chunk_path.mkdir(parents=True, exist_ok=True)

        cache_path = output_dir / "cache"

        feature_api = FeatureApi()
        classes_df = feature_api.get_feature_classes(packs)

        ai_offline_parcel.process_chunk(
            chunk_id=chunk_id,
            parcel_gdf=parcel_gdf_au_tests,
            classes_df=classes_df,
            output_dir=output_dir,
            key_file=None,
            config=None,
            country=country,
            packs=packs,
            include_parcel_geometry=True,
            save_features=True,
        )

        assert chunk_path.exists()
        assert (chunk_path / f"rollup_{chunk_id}.parquet").exists()
        assert cache_path.exists()

        data = []
        data_features = []
        errors = []

        for cp in chunk_path.glob(f"rollup_*.parquet"):
            data.append(pd.read_parquet(cp))

        outpath = final_path / f"{tag}.csv"
        outpath_features = final_path / f"{tag}_features.gpkg"
        data = pd.concat(data)
        data.to_csv(outpath, index=True)

        outpath_errors = final_path / f"{tag}_errors.csv"
        for cp in chunk_path.glob(f"errors_*.parquet"):
            errors.append(pd.read_parquet(cp))
        errors = pd.concat(errors)
        errors.to_csv(outpath_errors, index=True)

        for cp in [p for p in chunk_path.glob(f"features_*.parquet")]:
            data_features.append(gpd.read_parquet(cp))
        data_features = pd.concat(data_features)
        if len(data_features) > 0:
            data_features.to_file(outpath_features, driver="GPKG")

        assert outpath.exists()
        assert outpath_errors.exists()
        assert outpath_features.exists()

        assert len(data) == len(parcel_gdf_au_tests)  # Assert got a result for every parcel.
