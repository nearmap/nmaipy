from pathlib import Path

import pandas as pd
import geopandas as gpd
import pytest
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon
from shapely.wkt import loads

from nearmap_ai.constants import BUILDING_ID, SOLAR_ID, VEG_MEDHIGH_ID, ASPHALT_ID, SOLAR_HW_ID, API_CRS
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

    def test_trim_features_to_aoi(self, cache_directory: Path):
        aoi = loads(
            "MULTIPOLYGON (((-81.96101015253903 29.05631578134146, -81.96097715235663 29.05643178114233, -81.96088515183891 29.05676078057977, -81.96081015166226 29.05688178032226, -81.96071615151028 29.05699178005651, -81.96050315132298 29.05714677958319, -81.96030215120676 29.05725677918626, -81.96002115116396 29.05733877872941, -81.95975115121745 29.05736077836785, -81.95966415128315 29.0573387782909, -81.9592571518955 29.05705277817911, -81.95915015219606 29.05689377826324, -81.95908815262986 29.05664577852331, -81.95908115301701 29.05641477882801, -81.95915615341346 29.05616177926368, -81.95941915372902 29.05591977991694, -81.95949115377999 29.055874780067, -81.9598821540514 29.05563378087792, -81.96037115415523 29.05547378170073, -81.96061515411904 29.05544678203954, -81.96077815401978 29.05547378220442, -81.9608651538808 29.05553978222199, -81.96098515356586 29.05570478214509, -81.96103515330084 29.05585378200341, -81.96101015253903 29.05631578134146)), ((-81.965573157064 29.05269479199695, -81.96551115685583 29.05283179172925, -81.96541715672163 29.05293079147429, -81.96469115613426 29.05342678988017, -81.96450915619464 29.05342678965309, -81.96423415640598 29.05335478940948, -81.96400815645254 29.05337178910396, -81.96389515631647 29.05347578881938, -81.96363915561045 29.05394978784645, -81.96358915545353 29.05405378764083, -81.96360115525603 29.05416978749583, -81.96363915516162 29.05421878747553, -81.96372015504298 29.05427378750045, -81.96410215486105 29.05430678792986, -81.96424015465836 29.05440078797153, -81.96424615447276 29.05451078782699, -81.96420915438496 29.05457078769814, -81.96409615424729 29.05467578741273, -81.96392115414024 29.05477478705886, -81.96355815400395 29.0549287863961, -81.96329515397292 29.05499978597212, -81.96305715400537 29.05502778563849, -81.96281315415156 29.05498878538958, -81.96260015443251 29.0548627852986, -81.96218015526925 29.05444478535088, -81.96194215557841 29.05430678524426, -81.961836155797 29.05419678526315, -81.96179815600141 29.05408178537342, -81.96196115625911 29.05389478583257, -81.96219215657767 29.05365778644532, -81.9625221568685 29.05341778718632, -81.96287515717678 29.05316278797737, -81.96310015757098 29.05288178864514, -81.96318115798456 29.05261778910983, -81.96318815851441 29.05229878955775, -81.9632691590098 29.05198579009027, -81.96335715918424 29.05186379036877, -81.96410215975392 29.05137479197944, -81.96416415973346 29.05137479205743, -81.96422715961252 29.05143479205378, -81.96421415949663 29.05150679193795, -81.96361315825364 29.05237078999111, -81.96361315801505 29.05251378979393, -81.96368915791651 29.05255778982842, -81.96417115776693 29.05255179044018, -81.96456515752637 29.05261779084218, -81.96479715747789 29.0526007911561, -81.96501615763911 29.05246079162424, -81.96525415781238 29.05230979213174, -81.96541715775842 29.05230979233601, -81.96552915758446 29.05239179236256, -81.96556715744335 29.05246879230323, -81.965573157064 29.05269479199695)))                                                                                                                                                                                                                           ERROR:nearmap_ai:Geometry: MULTIPOLYGON (((-81.96101015253903 29.05631578134146, -81.96097715235663 29.05643178114233, -81.96088515183891 29.05676078057977, -81.96081015166226 29.05688178032226, -81.96071615151028 29.05699178005651, -81.96050315132298 29.05714677958319, -81.96030215120676 29.05725677918626, -81.96002115116396 29.05733877872941, -81.95975115121745 29.05736077836785, -81.95966415128315 29.0573387782909, -81.9592571518955 29.05705277817911, -81.95915015219606 29.05689377826324, -81.95908815262986 29.05664577852331, -81.95908115301701 29.05641477882801, -81.95915615341346 29.05616177926368, -81.95941915372902 29.05591977991694, -81.95949115377999 29.055874780067, -81.9598821540514 29.05563378087792, -81.96037115415523 29.05547378170073, -81.96061515411904 29.05544678203954, -81.96077815401978 29.05547378220442, -81.9608651538808 29.05553978222199, -81.96098515356586 29.05570478214509, -81.96103515330084 29.05585378200341, -81.96101015253903 29.05631578134146)), ((-81.965573157064 29.05269479199695, -81.96551115685583 29.05283179172925, -81.96541715672163 29.05293079147429, -81.96469115613426 29.05342678988017, -81.96450915619464 29.05342678965309, -81.96423415640598 29.05335478940948, -81.96400815645254 29.05337178910396, -81.96389515631647 29.05347578881938, -81.96363915561045 29.05394978784645, -81.96358915545353 29.05405378764083, -81.96360115525603 29.05416978749583, -81.96363915516162 29.05421878747553, -81.96372015504298 29.05427378750045, -81.96410215486105 29.05430678792986, -81.96424015465836 29.05440078797153, -81.96424615447276 29.05451078782699, -81.96420915438496 29.05457078769814, -81.96409615424729 29.05467578741273, -81.96392115414024 29.05477478705886, -81.96355815400395 29.0549287863961, -81.96329515397292 29.05499978597212, -81.96305715400537 29.05502778563849, -81.96281315415156 29.05498878538958, -81.96260015443251 29.0548627852986, -81.96218015526925 29.05444478535088, -81.96194215557841 29.05430678524426, -81.961836155797 29.05419678526315, -81.96179815600141 29.05408178537342, -81.96196115625911 29.05389478583257, -81.96219215657767 29.05365778644532, -81.9625221568685 29.05341778718632, -81.96287515717678 29.05316278797737, -81.96310015757098 29.05288178864514, -81.96318115798456 29.05261778910983, -81.96318815851441 29.05229878955775, -81.9632691590098 29.05198579009027, -81.96335715918424 29.05186379036877, -81.96410215975392 29.05137479197944, -81.96416415973346 29.05137479205743, -81.96422715961252 29.05143479205378, -81.96421415949663 29.05150679193795, -81.96361315825364 29.05237078999111, -81.96361315801505 29.05251378979393, -81.96368915791651 29.05255778982842, -81.96417115776693 29.05255179044018, -81.96456515752637 29.05261779084218, -81.96479715747789 29.0526007911561, -81.96501615763911 29.05246079162424, -81.96525415781238 29.05230979213174, -81.96541715775842 29.05230979233601, -81.96552915758446 29.05239179236256, -81.96556715744335 29.05246879230323, -81.965573157064 29.05269479199695)))"
        )
        country = "us"
        date_1 = "2010-01-01"
        date_2 = "2022-03-31"
        packs = ["building", "vegetation"]
        aoi_id = 47

        parcels_gdf = gpd.GeoDataFrame(
            [{"geometry": aoi, "aoi_id": aoi_id, "since": date_1, "until": date_2}], crs=API_CRS
        )

        classes_df = pd.DataFrame(
            [
                {"id": BUILDING_ID, "description": "building"},
                {"id": VEG_MEDHIGH_ID, "description": "Medium and High Vegetation (>2m)"},
            ]
        ).set_index("id")

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(aoi, country, packs, aoi_id, date_1, date_2)

        print(metadata)
        print(features_gdf.T)

        assert len(features_gdf.feature_id.unique()) == len(features_gdf)
        #TODO: Make this test richer/quantitative info, not just that we end up with unique feature IDs.

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
        features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(
            aoi_gdf, country, packs, date_1, date_2
        )
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
            "MultiPolygon (((-88.40618111505668253 43.06538384370446693, -88.40618111505668253 43.06557268197261834, -88.40601285961312783 43.06557268197261834, -88.40601285961312783 43.06538384370446693, -88.40618111505668253 43.06538384370446693)),((-88.40590800477149003 43.06555664855734022, -88.40590394063035262 43.06538028071267377, -88.40578689336528839 43.06537434239258033, -88.40577307528538142 43.06555189791498606, -88.40577307528538142 43.06555189791498606, -88.40590800477149003 43.06555664855734022)))"
        )
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
        aoi = loads(
            "Polygon((-87.98409445082069169 42.9844739669082827, -87.98409445082069169 42.98497943053578041, -87.98334642812285722 42.98497943053578041, -87.98334642812285722 42.9844739669082827, -87.98409445082069169 42.9844739669082827), (-87.98398389681051412 42.98492725383756152, -87.9834560905684242 42.98492725383756152, -87.98343201832427951 42.98454440595978809, -87.98402490878204674 42.98453201391029666, -87.98398389681051412 42.98492725383756152))"
        )
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
        aoi = loads(
            "Polygon ((-87.98869354480743254 42.98720736071164339, -87.98869354480743254 42.98745790451570059, -87.98832791466347203 42.98745790451570059, -87.98832791466347203 42.98720736071164339, -87.98869354480743254 42.98720736071164339),(-87.98862916890175256 42.9874149280962996, -87.98856291796968776 42.98741401370408255, -87.98856229296089282 42.9873628077175951, -87.98862916890175256 42.9874149280962996),(-87.98843666619346493 42.98739206828654602, -87.98848416686175256 42.98736737968242494, -87.98842104097366246 42.98729559979562254, -87.9883935405867561 42.98732120282217295, -87.98834416489208365 42.98737743800379718, -87.98837854037572015 42.98741355650798823, -87.98840979081536773 42.98735457818008854, -87.98840979081536773 42.98735457818008854, -87.98843666619346493 42.98739206828654602),(-87.9883354147689829 42.9873874963235707, -87.98833916482176676 42.9874492177950458, -87.98868416967553685 42.98745013218676547, -87.98868604470192167 42.98742315762577704, -87.9883697902526194 42.98741858566508256, -87.9883354147689829 42.9873874963235707),(-87.98868104463157636 42.98741721407682803, -87.98868104463157636 42.98721467587366618, -87.98833228972502241 42.98721238988563442, -87.98864729415674901 42.9874149280962996, -87.98864729415674901 42.9874149280962996, -87.98868104463157636 42.98741721407682803),(-87.9885416676707024 42.98740761295811552, -87.98855229282020218 42.98736509370006331, -87.98845979151882091 42.98731343047596454, -87.98850041709034997 42.98737469482540519, -87.98841666591209787 42.98740212660388949, -87.98843104111432467 42.98741172772346175, -87.98843104111432467 42.98741172772346175, -87.9885416676707024 42.98740761295811552))"
        )
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
        assert classes_df.loc[SOLAR_HW_ID].description == "Solar Hot Water"
        assert len(classes_df) == 3

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
