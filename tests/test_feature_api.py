from pathlib import Path

import geopandas as gpd
import pytest
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon
from shapely.wkt import loads

from nearmap_ai.constants import BUILDING_ID, SOLAR_ID, VEG_MEDHIGH_ID, ASPHALT_ID
from nearmap_ai.feature_api import FeatureApi
import nearmap_ai.log
import logging


class TestFeatureAPI:
    def test_get_features_gdf(self, sydney_aoi: Polygon, cache_directory: Path):
        date_1 = "2020-01-01"
        date_2 = "2020-06-01"
        region = "au"
        packs = ["building"]
        aoi_id = "123"

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(sydney_aoi, region, packs, aoi_id, date_1, date_2)
        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["date"] <= date_2
        # We get 3 buildings
        assert len(features_gdf) == 3
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 3
        # The AOI ID has been assigned
        assert len(features_gdf[features_gdf.aoi_id == aoi_id]) == 3

    def test_split_geometry_into_grid(self, sydney_aoi: Polygon):
        aoi = sydney_aoi
        b = aoi.bounds
        width = b[2] - b[0]
        height = b[3] - b[1]
        d = max(width, height)

        for cell_size in [d / 5, width, height, d, 2 * d]:
            df_gridded = FeatureApi.split_geometry_into_grid(aoi, cell_size)
            geom_recombined = df_gridded.geometry.unary_union
            assert geom_recombined.difference(aoi).area == pytest.approx(0)
            assert aoi.difference(geom_recombined).area == pytest.approx(0)

    def test_combine_features_gdf_from_grid(
        self,
    ):
        pass
        # TODO: Write a bunch of tests here for different nuanced scenarios.

    def test_large_aoi(self, cache_directory: Path, large_adelaide_aoi: Polygon):
        survey_resource_id = "fe48a583-da45-5cd3-9fee-8321354bdf7a"  # 2011-03-03
        country = "au"
        packs = ["building", "vegetation"]
        aoi_id = "0"

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(
            large_adelaide_aoi, region=country, packs=packs, aoi_id=aoi_id, survey_resource_id=survey_resource_id
        )
        # No error
        assert error is None
        # We get 3 buildings
        assert len(features_gdf.query("class_id == @BUILDING_ID")) == 6  # Guessed
        assert len(features_gdf.query("class_id == @VEG_MEDHIGH_ID")) == 213  # Guessed

        # Assert that buildings aren't overhanging the edge of the parcel. If this fails, the clipped/unclipped hasn't been managed correctly during the grid merge.
        assert features_gdf.query("class_id == @BUILDING_ID").clipped_area_sqm.sum() == pytest.approx(
            features_gdf.query("class_id == @BUILDING_ID").unclipped_area_sqm.sum(), 0.1
        )

        assert len(features_gdf.query("class_id == @VEG_MEDHIGH_ID")) == 213  # Guessed

        # The AOI ID has been assigned to all features
        assert len(features_gdf[features_gdf.aoi_id == aoi_id]) == len(features_gdf)

    def test_not_found(self, cache_directory: Path):
        # Somewhere in the Pacific
        aoi = loads(
            """
            POLYGON((
                -145.5385490968286 -28.74031798505824, 
                -145.5381490968286 -28.74031798505824,
                -145.5381490968286 -28.73991798505824, 
                -145.5385490968286 -28.73991798505824,
                -145.5385490968286 -28.74031798505824
                ))
        """
        )
        country = "au"
        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(aoi, region=country)
        # No data
        assert features_gdf is None
        assert metadata is None
        # There is a error message
        assert isinstance(error["message"], str)

    def test_get_cache(self, cache_directory: Path, sydney_aoi: Polygon):
        date_1 = "2020-06-01"
        date_2 = "2020-12-01"
        country = "au"
        packs = ["building"]
        aoi_id = "123"
        # First do a standard pull to ensure the file is populated in the cache.
        feature_api = FeatureApi(cache_dir=cache_directory, compress_cache=False)
        features_gdf, metadata, error = feature_api.get_features_gdf(sydney_aoi, country, packs, aoi_id, date_1, date_2)
        assert error is None

        # Then re-request using invalid API key to ensure data is not being pulled from the API but read from the cache.
        api_key = "not an api key"
        # Run
        feature_api = FeatureApi(api_key, cache_dir=cache_directory, compress_cache=False)
        features_gdf, metadata, error = feature_api.get_features_gdf(sydney_aoi, country, packs, aoi_id, date_1, date_2)
        # Check output
        assert error is None
        assert date_1 <= metadata["date"] <= date_2
        assert len(features_gdf) == 3

    def test_get_compressed_cache(self, cache_directory: Path, sydney_aoi: Polygon):
        date_1 = "2020-06-01"
        date_2 = "2020-12-01"
        country = "au"
        packs = ["building"]
        aoi_id = "123"
        # First do a standard pull to ensure the file is populated in the cache.
        feature_api = FeatureApi(cache_dir=cache_directory, compress_cache=True)
        features_gdf, metadata, error = feature_api.get_features_gdf(sydney_aoi, country, packs, aoi_id, date_1, date_2)
        assert error is None

        # Then re-request using invalid API key to ensure data is not being pulled from the API but read from the cache.
        api_key = "not an api key"
        # Run
        feature_api = FeatureApi(api_key, cache_dir=cache_directory, compress_cache=True)
        features_gdf, metadata, error = feature_api.get_features_gdf(sydney_aoi, country, packs, aoi_id, date_1, date_2)
        # Check output
        assert error is None
        assert date_1 <= metadata["date"] <= date_2
        assert len(features_gdf) == 3

    def test_get_bulk(self, cache_directory: Path, sydney_aoi: Polygon):
        aois = []
        for i in range(4):
            for j in range(4):
                aois.append({"aoi_id": f"{i}_{j}", "geometry": translate(sydney_aoi, 0.001 * i, 0.001 * j)})
        # Add an AOI with an invalid type to test an error case - multipolygon of two separate chunks

        aoi_gdf = gpd.GeoDataFrame(aois)
        date_1 = "2020-01-01"
        date_2 = "2020-12-01"
        country = "au"
        packs = ["building"]

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(aoi_gdf, country, packs, date_1, date_2)
        # Check metadata
        assert len(metadata_df) == 16
        assert len(metadata_df.merge(aoi_gdf, on="aoi_id", how="inner")) == 16
        # Check error
        assert len(errors_df) == 0
        # We get only buildings
        assert len(features_gdf) == 69
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 69

    def test_get_bulk_with_data_dates(self, cache_directory: Path, sydney_aoi: Polygon):
        aois = []
        for i in range(4):
            for j in range(4):
                aois.append(
                    {
                        "aoi_id": f"{i}_{j}",
                        "since": "2020-01-01",
                        "until": "2020-03-01",
                        "geometry": translate(sydney_aoi, 0.001 * i, 0.001 * j),
                    }
                )

        aoi_gdf = gpd.GeoDataFrame(aois)
        country = "au"
        packs = ["building"]

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(aoi_gdf, country, packs)
        print(metadata_df.iloc[0].T)

        # Check metadata
        assert len(metadata_df) == 16
        assert len(metadata_df.merge(aoi_gdf, on="aoi_id", how="inner")) == 16

        # We get only buildings
        assert len(features_gdf) == 69
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 69
        # The dates are within range
        for row in features_gdf.itertuples():
            assert "2020-01-01" <= row.survey_date <= "2020-03-01"

    def test_multipolygon_1(self, cache_directory: Path, sydney_aoi: Polygon):
        aoi = sydney_aoi.union(translate(sydney_aoi, 0.002, 0.01))
        print(f"Multipolygon 1 (use QuickWKT in QGIS to visualise): {aoi}")
        date_1 = "2020-01-01"
        date_2 = "2020-06-01"
        country = "au"
        packs = ["building"]
        aoi_id = "123"
        # Run
        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(aoi, country, packs, aoi_id, date_1, date_2)
        print(metadata)
        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["date"] <= date_2
        # We get 3 buildings
        assert len(features_gdf) == 6
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 6
        # The AOI ID has been assigned
        assert len(features_gdf[features_gdf.aoi_id == aoi_id]) == 6
        # All buildings intersect the AOI
        assert len(features_gdf[features_gdf.intersects(aoi)]) == 6

    def test_multipolygon_2(self, cache_directory: Path, sydney_aoi: Polygon):
        aoi = MultiPolygon([translate(sydney_aoi, 0.001, 0.001), translate(sydney_aoi, 0.003, 0.003)])
        print(f"Multipolygon 2 (use QuickWKT in QGIS to visualise): {aoi}")
        date_1 = "2020-01-01"
        date_2 = "2020-06-01"
        country = "au"
        packs = ["building"]
        aoi_id = "123"
        # Run
        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(aoi, country, packs, aoi_id, date_1, date_2)
        print(metadata)

        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["date"] <= date_2
        # We get 3 buildings
        assert len(features_gdf) == 11
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 11
        # The AOI ID has been assigned
        assert len(features_gdf[features_gdf.aoi_id == aoi_id]) == 11
        # All buildings intersect the AOI
        assert len(features_gdf[features_gdf.intersects(aoi)]) == 11

    def test_multipolygon_3(self, cache_directory: Path):
        """
        Multipolygon with two nearby parts, and an overlapping discrete class (building)
        """
        aoi = loads(
            "MultiPolygon (((-88.40618111505668253 43.06538384370446693, -88.40618111505668253 43.06557268197261834, -88.40601285961312783 43.06557268197261834, -88.40601285961312783 43.06538384370446693, -88.40618111505668253 43.06538384370446693)),((-88.40590800477149003 43.06555664855734022, -88.40590394063035262 43.06538028071267377, -88.40578689336528839 43.06537434239258033, -88.40577307528538142 43.06555189791498606, -88.40577307528538142 43.06555189791498606, -88.40590800477149003 43.06555664855734022)))")
        print(f"Multipolygon 2 (use QuickWKT in QGIS to visualise): {aoi}")
        date_1 = "2020-01-01"
        date_2 = "2022-07-01"
        country = "us"
        packs = ["building"]
        aoi_id = "3"
        # Run
        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(aoi, country, packs, aoi_id, date_1, date_2)
        print(metadata)

        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["date"] <= date_2
        # We get 3 buildings
        assert len(features_gdf) == 1
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 1
        # The AOI ID has been assigned
        assert len(features_gdf[features_gdf.aoi_id == aoi_id]) == 1
        # All buildings intersect the AOI
        assert len(features_gdf[features_gdf.intersects(aoi)]) == 1

        assert features_gdf["unclipped_area_sqm"].sum() == pytest.approx(154, rel=0.02)
        assert features_gdf["area_sqm"].sum() == pytest.approx(154, rel=0.02)
        assert features_gdf["clipped_area_sqm"].sum() == pytest.approx(70, rel=0.02)


    def test_polygon_with_hole_1(self, cache_directory: Path):
        # This one should have a building in the middle which gets pulled then discarded, and clear space around it.
        aoi = loads("Polygon((-87.98409445082069169 42.9844739669082827, -87.98409445082069169 42.98497943053578041, -87.98334642812285722 42.98497943053578041, -87.98334642812285722 42.9844739669082827, -87.98409445082069169 42.9844739669082827), (-87.98398389681051412 42.98492725383756152, -87.9834560905684242 42.98492725383756152, -87.98343201832427951 42.98454440595978809, -87.98402490878204674 42.98453201391029666, -87.98398389681051412 42.98492725383756152))")
        country = "us"
        date_1 = "2022-04-21"
        date_2 = "2022-04-21"
        packs = ["building"]
        aoi_id = 11

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(aoi, country, packs, aoi_id, date_1, date_2)
        print(metadata)

        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["date"] <= date_2
        # We get no buildings (inner gets discarded)
        assert len(features_gdf) == 0
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 0

    def test_polygon_with_hole_2(self, cache_directory: Path):
        """
        Test correct behaviour of a connected class, with complex many holed query AOI.
        """
        aoi = loads("Polygon ((-87.98869354480743254 42.98720736071164339, -87.98869354480743254 42.98745790451570059, -87.98832791466347203 42.98745790451570059, -87.98832791466347203 42.98720736071164339, -87.98869354480743254 42.98720736071164339),(-87.98862916890175256 42.9874149280962996, -87.98856291796968776 42.98741401370408255, -87.98856229296089282 42.9873628077175951, -87.98862916890175256 42.9874149280962996),(-87.98843666619346493 42.98739206828654602, -87.98848416686175256 42.98736737968242494, -87.98842104097366246 42.98729559979562254, -87.9883935405867561 42.98732120282217295, -87.98834416489208365 42.98737743800379718, -87.98837854037572015 42.98741355650798823, -87.98840979081536773 42.98735457818008854, -87.98840979081536773 42.98735457818008854, -87.98843666619346493 42.98739206828654602),(-87.9883354147689829 42.9873874963235707, -87.98833916482176676 42.9874492177950458, -87.98868416967553685 42.98745013218676547, -87.98868604470192167 42.98742315762577704, -87.9883697902526194 42.98741858566508256, -87.9883354147689829 42.9873874963235707),(-87.98868104463157636 42.98741721407682803, -87.98868104463157636 42.98721467587366618, -87.98833228972502241 42.98721238988563442, -87.98864729415674901 42.9874149280962996, -87.98864729415674901 42.9874149280962996, -87.98868104463157636 42.98741721407682803),(-87.9885416676707024 42.98740761295811552, -87.98855229282020218 42.98736509370006331, -87.98845979151882091 42.98731343047596454, -87.98850041709034997 42.98737469482540519, -87.98841666591209787 42.98740212660388949, -87.98843104111432467 42.98741172772346175, -87.98843104111432467 42.98741172772346175, -87.9885416676707024 42.98740761295811552))")
        country = "us"
        date_1 = "2022-04-21"
        date_2 = "2022-04-21"
        packs = ["surfaces"]
        aoi_id = 12

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(aoi, country, packs, aoi_id, date_1, date_2)
        print(metadata)

        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["date"] <= date_2
        # We get one piece of hollowed out asphalt
        assert len(features_gdf) == 1
        assert len(features_gdf[features_gdf.class_id == ASPHALT_ID]) == 1
        assert features_gdf["clipped_area_sqm"].sum() == features_gdf["area_sqm"].sum()
        assert features_gdf["clipped_area_sqm"].sum() == pytest.approx(259.2, rel=0.02)

    def test_classes(self, cache_directory: Path):
        feature_api = FeatureApi(cache_dir=cache_directory)
        classes_df = feature_api.get_feature_classes()
        assert classes_df.loc[BUILDING_ID].description == "Building"

    def test_classes_filtered(self, cache_directory: Path):
        feature_api = FeatureApi(cache_dir=cache_directory)
        classes_df = feature_api.get_feature_classes(packs=["solar", "building"])
        assert classes_df.loc[BUILDING_ID].description == "Building"
        assert classes_df.loc[SOLAR_ID].description == "Solar Panel"
        assert len(classes_df) == 2

    def test_unknown_pack(self, cache_directory: Path):
        feature_api = FeatureApi(cache_dir=cache_directory)
        with pytest.raises(ValueError) as excinfo:
            feature_api.get_feature_classes(packs=["solar", "foobar"])
        assert "foobar" in str(excinfo.value)

    def test_packs(self, cache_directory: Path):
        """
        Test that this set of packs are all valid. Does not check whether additional packs have been made available.
        """
        feature_api = FeatureApi(cache_dir=cache_directory)
        packs = feature_api.get_packs()
        expected_subset = {
            "building",
            "building_char",
            "roof_char",
            "roof_cond",
            "surfaces",
            "vegetation",
            "poles",
            "construction",
            "pool",
            "solar",
            "trampoline",
        }
        assert not expected_subset.difference(packs.keys())
