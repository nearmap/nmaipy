from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import pytest
from shapely.wkt import loads

from nmaipy import parcels
from nmaipy.constants import (
    BUILDING_LIFECYCLE_ID,
    BUILDING_ID,
    BUILDING_NEW_ID,
    ROOF_ID,
    LAWN_GRASS_ID,
    POOL_ID,
    VEG_MEDHIGH_ID,
    AOI_ID_COLUMN_NAME,
    API_CRS,
    WATER_BODY_ID,
    AREA_CRS,
    SQUARED_METERS_TO_SQUARED_FEET,
)
from nmaipy.feature_api import FeatureApi


@pytest.mark.skip("Comment out this line if you wish to regen the test data")
def test_gen_data(parcels_gdf, data_directory: Path, cache_directory: Path):
    """
    Generate the test data for the parcels tests. Uses a specific date to ensure the data is consistent.
    """
    outfname = data_directory / "test_features.csv"
    from nmaipy.feature_api import FeatureApi

    packs = ["building", "building_char", "roof_char", "roof_cond", "surfaces", "vegetation"]
    features_gdf, _, _ = FeatureApi(cache_dir=cache_directory, alpha=True, beta=True).get_features_gdf_bulk(
        parcels_gdf, packs=packs, region="au", since_bulk="2021-10-04", until_bulk="2021-10-04"
    )
    features_gdf.to_csv(outfname, index=False)


@pytest.mark.skip("Comment out this line if you wish to regen the test data")
def test_gen_data_2(parcels_2_gdf, data_directory: Path, cache_directory: Path):
    """
    Generate secondary test data set.
    """
    outfname = data_directory / "test_features_2.csv"
    from nmaipy.feature_api import FeatureApi

    packs = ["building", "building_char", "roof_char", "roof_cond", "surfaces", "vegetation"]
    features_gdf, _, _ = FeatureApi(cache_dir=cache_directory, workers=1).get_features_gdf_bulk(
        parcels_2_gdf, packs=packs, region="us", since_bulk="2022-06-29", until_bulk="2022-06-29"
    )
    features_gdf.to_csv(outfname, index=False)


class TestParcels:
    def test_filter(self, features_2_gdf, parcels_2_gdf):
        assert len(features_2_gdf) == 1409
        f_gdf = features_2_gdf[features_2_gdf.class_id == ROOF_ID]
        assert len(f_gdf) == 161
        country = "us"
        config = {
            "min_size": {
                ROOF_ID: 4,
            },
            "min_confidence": {
                ROOF_ID: 0.65,
            },
            "min_fidelity": {
                ROOF_ID: 0.15,
            },
            "min_area_in_parcel": {
                ROOF_ID: 4,
            },
            "min_ratio_in_parcel": {
                ROOF_ID: 0,
            },
            "building_style_filtering": {
                BUILDING_LIFECYCLE_ID: True,
                BUILDING_ID: True,
                BUILDING_NEW_ID: True,
                ROOF_ID: True,
            },
        }
        filtered_gdf = parcels.filter_features_in_parcels(f_gdf, region=country, config=config, aoi_gdf=parcels_2_gdf)
        assert len(filtered_gdf) == 127 # Manually checked that four buildings should be removed from the 131, as they visually don't belong in the parcel.
        assert not (filtered_gdf.confidence < 0.65).any()
        assert not (filtered_gdf.unclipped_area_sqm < 4).any()
        assert not (filtered_gdf.fidelity < 0.15).any()

    def test_flatten_building(self):
        country = "au"
        attributes = [
            {
                "classId": "19e49dad-4228-554e-9f5e-c2e37b2e11d9",
                "description": "Building 3d attributes",
                "has3dAttributes": True,
                "height": 8.887635612487793,
                "numStories": {"1": 0.057618971750252275, "2": 0.8145058300927666, "3+": 0.1278751981569811},
            }
        ]
        expected = {
            "has_3d_attributes": "Y",
            "height_m": 8.9,
            "num_storeys_1_confidence": 0.057618971750252275,
            "num_storeys_2_confidence": 0.8145058300927666,
            "num_storeys_3+_confidence": 0.1278751981569811,
        }
        assert expected == parcels.flatten_building_attributes(attributes, country)

    def test_flatten_roof(self):
        attributes = [
            {
                "classId": "39072960-5582-52af-9051-4bc8625ff9ba",
                "description": "Roof 3d attributes",
                "has3dAttributes": True,
                "pitch": 26.21,
            },
            {
                "classId": "3065525d-3f14-5b9d-8c4c-077f1ad5c694",
                "components": [
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "f907e625-26b3-59db-a806-d41f62ce1f1b",
                        "confidence": 1,
                        "description": "Structurally Damaged Roof",
                        "ratio": 0,
                    },
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "abb1f304-ce01-527b-b799-cbfd07551b2c",
                        "confidence": 1,
                        "description": "Roof With Temporary Repair",
                        "ratio": 0,
                    },
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "f41e02b0-adc0-5b46-ac95-8c59aa9fe317",
                        "confidence": 1,
                        "description": "Roof Ponding",
                        "ratio": 0,
                    },
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "526496bf-7344-5024-82d7-77ceb671feb4",
                        "confidence": 1,
                        "description": "Roof Rusting",
                        "ratio": 0,
                    },
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "cfa8951a-4c29-54de-ae98-e5f804c305e3",
                        "confidence": 1,
                        "description": "Roof Tile/Shingle Discolouration",
                        "ratio": 0,
                    },
                ],
                "description": "Roof condition",
            },
            {
                "classId": "89c7d478-58de-56bd-96d2-e71e27a36905",
                "components": [
                    {
                        "areaSqft": 3059,
                        "areaSqm": 284.2,
                        "classId": "516fdfd5-0be9-59fe-b849-92faef8ef26e",
                        "confidence": 0.9904731109239588,
                        "description": "Tile Roof",
                        "dominant": True,
                        "ratio": 0.9113204992491568,
                    },
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "4bbf8dbd-cc81-5773-961f-0121101422be",
                        "confidence": 1,
                        "description": "Shingle Roof",
                        "dominant": False,
                        "ratio": 0,
                    },
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "4424186a-0b42-5608-a5a0-d4432695c260",
                        "confidence": 1,
                        "description": "Metal Roof",
                        "dominant": False,
                        "ratio": 0,
                    },
                ],
                "description": "Roof material",
            },
            {
                "classId": "20a58db2-bc02-531d-98f5-451f88ce1fed",
                "components": [
                    {
                        "areaSqft": 517,
                        "areaSqm": 48,
                        "classId": "ac0a5f75-d8aa-554c-8a43-cee9684ef9e9",
                        "confidence": 0.7464101831606086,
                        "description": "Hip",
                        "ratio": 0.15403978237857266,
                    },
                    {
                        "areaSqft": 805,
                        "areaSqm": 74.8,
                        "classId": "59c6e27e-6ef2-5b5c-90e7-31cfca78c0c2",
                        "confidence": 0.7787800614990344,
                        "description": "Gable",
                        "ratio": 0.23987592624504567,
                    },
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "3719eb40-d6d1-5071-bbe6-379a551bb65f",
                        "confidence": 1,
                        "description": "Dutch Gable",
                        "ratio": 0,
                    },
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "224f98d3-b853-542a-8b18-e1e46e3a8200",
                        "confidence": 1,
                        "description": "Flat",
                        "ratio": 0,
                    },
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "89582082-e5b8-5853-bc94-3a0392cab98a",
                        "confidence": 1,
                        "description": "Turret",
                        "ratio": 0,
                    },
                    {
                        "areaSqft": 66,
                        "areaSqm": 6.1,
                        "classId": "6e78c065-ecd9-59e3-8b62-cdef9a310dde",
                        "confidence": 0.5372738248389376,
                        "description": "Other Roof Shape",
                        "ratio": 0.019589621131926836,
                    },
                ],
                "description": "Roof types",
            },
            {
                "classId": "7ab56e15-d5d4-51bb-92bd-69e910e82e56",
                "components": [
                    {
                        "areaSqft": 0,
                        "areaSqm": 0,
                        "classId": "8e9448bd-4669-5f46-b8f0-840fee25c34c",
                        "confidence": 1,
                        "description": "Tree Overhang",
                        "ratio": 0,
                    }
                ],
                "description": "Roof tree overhang",
            },
        ]
        expected = {
            "has_3d_attributes": "Y",
            "pitch_degrees": 26.21,
            "structurally_damaged_roof_present": "N",
            "structurally_damaged_roof_area_sqm": 0,
            "structurally_damaged_roof_confidence": 1,
            "roof_with_temporary_repair_present": "N",
            "roof_with_temporary_repair_area_sqm": 0,
            "roof_with_temporary_repair_confidence": 1,
            "roof_ponding_present": "N",
            "roof_ponding_area_sqm": 0,
            "roof_ponding_confidence": 1,
            "roof_rusting_present": "N",
            "roof_rusting_area_sqm": 0,
            "roof_rusting_confidence": 1,
            "roof_tile/shingle_discolouration_present": "N",
            "roof_tile/shingle_discolouration_area_sqm": 0,
            "roof_tile/shingle_discolouration_confidence": 1,
            "tile_roof_present": "Y",
            "tile_roof_area_sqm": 284.2,
            "tile_roof_confidence": 0.9904731109239588,
            "tile_roof_dominant": "Y",
            "shingle_roof_present": "N",
            "shingle_roof_area_sqm": 0,
            "shingle_roof_confidence": 1,
            "shingle_roof_dominant": "N",
            "metal_roof_present": "N",
            "metal_roof_area_sqm": 0,
            "metal_roof_confidence": 1,
            "metal_roof_dominant": "N",
            "hip_present": "Y",
            "hip_area_sqm": 48,
            "hip_confidence": 0.7464101831606086,
            "gable_present": "Y",
            "gable_area_sqm": 74.8,
            "gable_confidence": 0.7787800614990344,
            "dutch_gable_present": "N",
            "dutch_gable_area_sqm": 0,
            "dutch_gable_confidence": 1,
            "flat_present": "N",
            "flat_area_sqm": 0,
            "flat_confidence": 1,
            "turret_present": "N",
            "turret_area_sqm": 0,
            "turret_confidence": 1,
            "other_roof_shape_present": "Y",
            "other_roof_shape_area_sqm": 6.1,
            "other_roof_shape_confidence": 0.5372738248389376,
            "tree_overhang_present": "N",
            "tree_overhang_area_sqm": 0,
            "tree_overhang_confidence": 1,
        }
        assert expected == parcels.flatten_roof_attributes(attributes, "au")

    def test_rollup(self, parcels_gdf, features_gdf):
        classes_df = pd.DataFrame(
            {"id": BUILDING_ID, "description": "building"},
            {"id": POOL_ID, "description": "pool"},
            {"id": LAWN_GRASS_ID, "description": "lawn"},
        ).set_index("id")
        country = "au"
        features_gdf = parcels.filter_features_in_parcels(features_gdf, aoi_gdf=parcels_gdf, region=country)
        df = parcels.parcel_rollup(
            parcels_gdf,
            features_gdf,
            classes_df,
            country=country,
            calc_buffers=False,
            primary_decision="largest_intersection",
        )

        expected = pd.DataFrame.from_dict(
            {
                "building_present": {
                    0: "Y",
                    1: "Y",
                    2: "Y",
                    3: "N",
                    4: "Y",
                    5: "Y",
                    6: "Y",
                    7: "N",
                    8: "Y",
                    9: "Y",
                    10: "Y",
                    11: "Y",
                    12: "Y",
                    13: "Y",
                    14: "Y",
                    15: "Y",
                },
                "building_count": {
                    0: 3,
                    1: 6,
                    2: 2,
                    3: 0,
                    4: 5,
                    5: 9,
                    6: 5,
                    7: 0,
                    8: 7,
                    9: 6,
                    10: 4,
                    11: 1,
                    12: 4,
                    13: 5,
                    14: 7,
                    15: 2,
                },
                "building_total_area_sqm": {
                    0: 459,
                    1: 787,
                    2: 13.1,
                    3: 0,
                    4: 1029,
                    5: 706,
                    6: 947.5,
                    7: 0.0,
                    8: 873,
                    9: 529,
                    10: 689,
                    11: 3717.3,
                    12: 1589,
                    13: 1281.3,
                    14: 1048.7,
                    15: 306.2,
                },
                "building_total_clipped_area_sqm": {
                    0: 453.8,
                    1: 573.3,
                    2: 13.0,
                    3: 0.0,
                    4: 633.1,
                    5: 664.5,
                    6: 640.0,
                    7: 0.0,
                    8: 746.6,
                    9: 513.6,
                    10: 454.9,
                    11: 637.0,
                    12: 636.8,
                    13: 646.1,
                    14: 556.1,
                    15: 76.9,
                },
                "building_total_unclipped_area_sqm": {
                    0: 459,
                    1: 787,
                    2: 13.1,
                    3: 0,
                    4: 1029,
                    5: 706,
                    6: 947.5,
                    7: 0.0,
                    8: 873,
                    9: 529,
                    10: 689,
                    11: 3717.3,
                    12: 1589,
                    13: 1281.3,
                    14: 1048.7,
                    15: 306.2,
                },
                "building_confidence": {
                    0: 0.999995194375515,
                    1: 0.9999999999999569,
                    2: 0.9995231628417969,
                    3: None,
                    4: 0.999999999987466,
                    5: 1.0,
                    6: 0.9999999999788258,
                    7: None,
                    8: 1.0,
                    9: 0.9999999999999964,
                    10: 0.9999999993451638,
                    11: 0.998046875,
                    12: 0.9999999978899723,
                    13: 0.9999999999893419,
                    14: 0.9999999999999998,
                    15: 0.9997901916503906,
                },
                "primary_building_area_sqm": {
                    0: 459.2,
                    1: 412.7,
                    2: 13.1,
                    3: 0.0,
                    4: 211.7,
                    5: 324.8,
                    6: 419.5,
                    7: 0.0,
                    8: 314.0,
                    9: 178.9,
                    10: 508.6,
                    11: 3717.3,
                    12: 1089.8,
                    13: 457.5,
                    14: 250.2,
                    15: 306.2,
                },
                "primary_building_clipped_area_sqm": {
                    0: 438.5,
                    1: 335.7,
                    2: 13.0,
                    3: 0.0,
                    4: 211.8,
                    5: 308.2,
                    6: 403.4,
                    7: 0.0,
                    8: 304.8,
                    9: 177.7,
                    10: 329.9,
                    11: 637.0,
                    12: 316.9,
                    13: 346.3,
                    14: 240.9,
                    15: 70.3,
                },
                "primary_building_unclipped_area_sqm": {
                    0: 459.2,
                    1: 412.7,
                    2: 13.1,
                    3: 0.0,
                    4: 211.7,
                    5: 324.8,
                    6: 419.5,
                    7: 0.0,
                    8: 314.0,
                    9: 178.9,
                    10: 508.6,
                    11: 3717.3,
                    12: 1089.8,
                    13: 457.5,
                    14: 250.2,
                    15: 306.2,
                },
                "primary_building_confidence": {
                    0: 0.990234375,
                    1: 0.998046875,
                    2: 0.755859375,
                    3: None,
                    4: 0.994140625,
                    5: 0.994140625,
                    6: 0.998046875,
                    7: None,
                    8: 0.994140625,
                    9: 0.998046875,
                    10: 0.998046875,
                    11: 0.998046875,
                    12: 0.990234375,
                    13: 0.990234375,
                    14: 0.998046875,
                    15: 0.990234375,
                },
                "primary_building_fidelity": {
                    0: 0.8072493587202652,
                    1: 0.9288912999298852,
                    2: 0.8367290593213565,
                    3: None,
                    4: 0.9380670718691608,
                    5: 0.8300419354124584,
                    6: 0.9363495102842524,
                    7: None,
                    8: 0.8059515328388869,
                    9: 0.906370564960434,
                    10: 0.9061553858042126,
                    11: 0.9088252238622468,
                    12: 0.6737367716860018,
                    13: 0.7497063672869632,
                    14: 0.8260399932417153,
                    15: 0.8458701178311958,
                },
                "primary_building_has_3d_attributes": {
                    0: "Y",
                    1: "Y",
                    2: "N",
                    3: None,
                    4: "Y",
                    5: "Y",
                    6: "Y",
                    7: None,
                    8: "Y",
                    9: "Y",
                    10: "Y",
                    11: "Y",
                    12: "Y",
                    13: "Y",
                    14: "Y",
                    15: "Y",
                },
                "primary_building_height_m": {
                    0: 8.6,
                    1: 13.0,
                    2: None,
                    3: None,
                    4: 12.9,
                    5: 7.0,
                    6: 8.6,
                    7: None,
                    8: 7.6,
                    9: 9.1,
                    10: 9.2,
                    11: 12.0,
                    12: 16.2,
                    13: 8.5,
                    14: 9.1,
                    15: 7.4,
                },
                "primary_building_num_storeys_1_confidence": {
                    0: 0.021999877253298276,
                    1: 0.012903212313960154,
                    2: None,
                    3: None,
                    4: 0.0002661113620101014,
                    5: 0.4549840465642416,
                    6: 0.21207449253675276,
                    7: None,
                    8: 0.44725765925037214,
                    9: 0.030071088843366463,
                    10: 0.1980398075081962,
                    11: 0.0912941739535858,
                    12: 0.09,
                    13: 0.2793941032102138,
                    14: 0.039923396691618575,
                    15: 0.24804083595198403,
                },
                "primary_building_num_storeys_2_confidence": {
                    0: 0.7493167390107146,
                    1: 0.4345346900569799,
                    2: None,
                    3: None,
                    4: 0.22250085557225888,
                    5: 0.5381568182470936,
                    6: 0.6622280308540118,
                    7: None,
                    8: 0.5442314440618939,
                    9: 0.6074534411436349,
                    10: 0.5867560387781604,
                    11: 0.30264025367209835,
                    12: 0.36354650405280986,
                    13: 0.6197106473080908,
                    14: 0.8415278350378601,
                    15: 0.7504207470420801,
                },
                "primary_building_num_storeys_3+_confidence": {
                    0: 0.22868338373598704,
                    1: 0.55256209762906,
                    2: None,
                    3: None,
                    4: 0.777233033065731,
                    5: 0.006859135188664742,
                    6: 0.12569747660923533,
                    7: None,
                    8: 0.008510896687734013,
                    9: 0.3624754700129988,
                    10: 0.2152041537136434,
                    11: 0.6060655723743159,
                    12: 0.5464534959471902,
                    13: 0.1008952494816952,
                    14: 0.11854876827052133,
                    15: 0.001538417005935797,
                },
                "aoi_id": {
                    0: "0_0",
                    1: "0_1",
                    2: "0_2",
                    3: "0_3",
                    4: "1_0",
                    5: "1_1",
                    6: "1_2",
                    7: "1_3",
                    8: "2_0",
                    9: "2_1",
                    10: "2_2",
                    11: "2_3",
                    12: "3_0",
                    13: "3_1",
                    14: "3_2",
                    15: "3_3",
                },
                "mesh_date": {
                    0: "2021-01-23",
                    1: "2021-01-23",
                    2: "2021-01-23",
                    3: "2021-01-23",
                    4: "2021-01-23",
                    5: "2021-01-23",
                    6: "2021-01-23",
                    7: "2021-01-23",
                    8: "2021-01-23",
                    9: "2021-01-23",
                    10: "2021-01-23",
                    11: "2021-01-23",
                    12: "2021-01-23",
                    13: "2021-01-23",
                    14: "2021-01-23",
                    15: "2021-01-23",
                },
            }
        )
        pd.testing.assert_frame_equal(df, expected, rtol=0.8)

    def test_nearest_primary(self):
        parcels_gdf = gpd.GeoDataFrame(
            [
                {
                    "aoi_id": 0,
                    "lat": 42.0005,
                    "lon": -114.9997,
                    "geometry": loads("POLYGON ((-114.999 42, -114.999 42.001, -115 42.001, -115 42, -114.999 42))"),
                }
            ],
            geometry="geometry",
        )
        parcels_gdf = parcels_gdf.set_crs("EPSG:4326")
        country = "us"
        features_gdf = gpd.GeoDataFrame(
            [
                # This should be the primary
                {
                    "feature_id": 0,
                    "aoi_id": 0,
                    "confidence": 0.94,
                    "fidelity": 0.7,  # Made up
                    "class_id": "0339726f-081e-5a6e-b9a9-42d95c1b5c8a",
                    "mesh_date": "2021-10-10",
                    "geometry": loads(
                        "POLYGON ((-114.9996 42.0001, -114.9996 42.00040000000001, -114.9999 42.00040000000001, -114.9999 42.0001, -114.9996 42.0001))"
                    ),
                },
                # Larger, but further away
                {
                    "feature_id": 1,
                    "aoi_id": 0,
                    "confidence": 0.92,
                    "fidelity": 0.7,  # Made up
                    "class_id": "0339726f-081e-5a6e-b9a9-42d95c1b5c8a",
                    "mesh_date": "2021-10-10",
                    "geometry": loads(
                        "POLYGON ((-114.9991 42.0001, -114.9991 42.00040000000001, -114.9995 42.00040000000001, -114.9995 42.0001, -114.9991 42.0001))"
                    ),
                },
                # Closer, but low confidence
                {
                    "feature_id": 2,
                    "aoi_id": 0,
                    "confidence": 0.85,
                    "fidelity": 0.6,
                    "class_id": "0339726f-081e-5a6e-b9a9-42d95c1b5c8a",
                    "mesh_date": "2021-10-10",
                    "geometry": loads(
                        "POLYGON ((-114.9996 42.0005, -114.9996 42.00056, -114.9999 42.00056, -114.9999 42.0005, -114.9996 42.0005))"
                    ),
                },
            ]
        )
        features_gdf = features_gdf.set_crs("EPSG:4326")

        # calculate areas
        parcels_gdf = parcels_gdf.to_crs("esri:102003")
        features_gdf = features_gdf.to_crs("esri:102003")
        features_gdf["area_sqm"] = features_gdf.area
        features_gdf["unclipped_area_sqm"] = features_gdf["area_sqm"]
        gdf = features_gdf.merge(parcels_gdf, on="aoi_id", how="left", suffixes=["_feature", "_aoi"])
        gdf["clipped_area_sqm"] = gdf.apply(
            lambda row: row.geometry_feature.intersection(row.geometry_aoi).area, axis=1
        )
        features_gdf = features_gdf.merge(gdf[["feature_id", "clipped_area_sqm"]])
        for col in ["area_sqm", "clipped_area_sqm", "unclipped_area_sqm"]:
            features_gdf[col.replace("sqm", "sqft")] = features_gdf[col] * 3.28084
        parcels_gdf = parcels_gdf.to_crs("EPSG:4326")
        features_gdf = features_gdf.to_crs("EPSG:4326")

        classes_df = pd.DataFrame([["Pool"]], columns=["description"], index=["0339726f-081e-5a6e-b9a9-42d95c1b5c8a"])

        features_gdf = parcels.filter_features_in_parcels(features_gdf, aoi_gdf=parcels_gdf, region=country)

        rollup_df = parcels.parcel_rollup(
            parcels_gdf, features_gdf, classes_df, country=country, calc_buffers=False, primary_decision="nearest"
        )
        expected = pd.DataFrame(
            [
                {
                    "pool_present": "Y",
                    "pool_count": 3,
                    "pool_total_area_sqft": 6883.735654712847,
                    "pool_total_clipped_area_sqft": 6883.7,
                    "pool_total_unclipped_area_sqft": 6883.7,
                    "pool_confidence": 0.99928,
                    "primary_pool_area_sqft": 2717.2650041320935,
                    "primary_pool_clipped_area_sqft": 2717.3,
                    "primary_pool_unclipped_area_sqft": 2717.3,
                    "primary_pool_confidence": 0.94,
                    "aoi_id": 0,
                    "mesh_date": "2021-10-10",
                }
            ]
        )
        pd.testing.assert_frame_equal(rollup_df, expected, atol=1e-3)

    def test_filter_and_rollup_gridded(self, cache_directory: Path, parcel_gdf_au_tests: gpd.GeoDataFrame):
        packs = ["building", "vegetation"]
        country = "au"
        parcel_gdf = parcel_gdf_au_tests

        feature_api = FeatureApi(cache_dir=cache_directory)
        classes_df = feature_api.get_feature_classes(packs)

        features_gdf, metadata_df, error_df = feature_api.get_features_gdf_bulk(parcel_gdf, region=country, packs=packs)

        # No error
        assert len(error_df) == 0

        # Had a bug that didn't do AOI IDs properly
        assert features_gdf[AOI_ID_COLUMN_NAME].isna().sum() == 0
        assert metadata_df[AOI_ID_COLUMN_NAME].isna().sum() == 0

        # We get results in all AOIs
        assert len(features_gdf.aoi_id.unique()) == len(parcel_gdf)

        features_gdf_filtered = parcels.filter_features_in_parcels(
            features_gdf, config=None, aoi_gdf=parcel_gdf, region=country
        )
        assert (
            len(features_gdf) > len(features_gdf_filtered) * 0.95
        )  # Very little should have been filtered out in these examples.
        assert len(features_gdf_filtered.aoi_id.unique()) == len(parcel_gdf)

        # Check rollup matches what's expected
        rollup_df = parcels.parcel_rollup(
            parcel_gdf,
            features_gdf,
            classes_df,
            country=country,
            calc_buffers=False,
            primary_decision="largest_intersection",
        )
        assert len(rollup_df) == len(parcel_gdf)
        final_df = metadata_df.merge(rollup_df, on=AOI_ID_COLUMN_NAME).merge(parcel_gdf, on=AOI_ID_COLUMN_NAME)
        assert len(final_df) == len(parcel_gdf)

    def test_tree_buffers_null(self, features_gdf, parcels_gdf):
        """
        Test a scenario where we know a building always at least partially intersects the boundary, so buffers are never
        valid to create.

        Args:
            features_gdf:
            parcels_gdf:

        Returns:

        """
        classes_df = pd.DataFrame(
            {"id": BUILDING_ID, "description": "building"},
            {"id": POOL_ID, "description": "pool"},
            {"id": LAWN_GRASS_ID, "description": "lawn"},
        ).set_index("id")
        country = "au"
        features_gdf = parcels.filter_features_in_parcels(features_gdf, aoi_gdf=parcels_gdf, region=country)
        df = parcels.parcel_rollup(
            parcels_gdf,
            features_gdf,
            classes_df,
            country=country,
            calc_buffers=True,
            primary_decision="largest_intersection",
        )
        # For this test, every result should be NaN for buffers, as there's always a building overlapping the parcel boundary.
        assert df.filter(like="buffer").head().isna().all().all()

    def test_tree_buffers_real(self, features_2_gdf, parcels_2_gdf):
        """
        Test a scenario where we know a building always at least partially intersects the boundary, so buffers are never
        valid to create.

        Args:
            features_2_gdf: Stored features generated from parcels_2_gdf queries.
            parcels_2_gdf: Realistic set of 100 parcels from New York.

        Returns:

        """
        country = "us"
        classes_df = pd.DataFrame(
            {"id": BUILDING_ID, "description": "building"},
            {"id": POOL_ID, "description": "pool"},
            {"id": LAWN_GRASS_ID, "description": "lawn"},
        ).set_index("id")
        country = "us"
        features_gdf = parcels.filter_features_in_parcels(features_2_gdf, aoi_gdf=parcels_2_gdf, region=country)
        df = parcels.parcel_rollup(
            parcels_2_gdf,
            features_gdf,
            classes_df,
            country=country,
            calc_buffers=True,
            primary_decision="largest_intersection",
        )

        # Test that all buffers fail when they're too large for the parcels.
        assert df.filter(like="30ft_tree_zone").isna().all().all()
        assert df.filter(like="100ft_tree_zone").isna().all().all()

        # Test values checked off a correct result with Gen 5 data (checked in at same time as this comment).
        np.testing.assert_allclose(df.filter(like="tree_zone").sum().values, [278, 18, 188, 3, 0, 0, 0, 0], rtol=0.12)
        np.testing.assert_allclose(df.filter(like="building_count").sum().values, [127, 16, 3, 0, 0], rtol=0.05)

    def test_building_fidelity_filter_scenario(self, cache_directory: Path):
        """
        Test a particular area of water in New York, which had a false positive building with large area, high
        confidence, and fidelity of 0.16 in Gen 4.

        Returns:

        """
        aoi = loads(
            "Polygon ((-74.06416616471615555 40.65346976328397233, -74.06416616471615555 40.6752980952800911, -74.02222033693037417 40.6752980952800911, -74.02222033693037417 40.65346976328397233, -74.06416616471615555 40.65346976328397233))"
        )
        country = "us"
        date_1 = "2022-02-27"
        date_2 = "2022-02-27"
        packs = ["building", "surfaces"]
        aoi_id = 42

        parcels_gdf = gpd.GeoDataFrame(
            [{"geometry": aoi, "aoi_id": aoi_id, "since": date_1, "until": date_2}], crs=API_CRS
        )

        classes_df = pd.DataFrame(
            {"id": BUILDING_ID, "description": "building"}, {"id": WATER_BODY_ID, "description": "water_body"}
        ).set_index("id")

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, errors = feature_api.get_features_gdf(aoi, country, packs, aoi_id, date_1, date_2)
        print(metadata)
        config = {
            "min_size": {
                BUILDING_ID: 4,
            },
            "min_confidence": {
                BUILDING_ID: 0.65,
            },
            "min_fidelity": {
                BUILDING_ID: 0.4, # This is not the default, but serves the purpose of the test
            },
            "min_area_in_parcel": {
                BUILDING_ID: 4,
            },
            "min_ratio_in_parcel": {
                BUILDING_ID: 0.0,
            },
        }
        features_gdf = parcels.filter_features_in_parcels(
            features_gdf, config=config, region=country, aoi_gdf=parcels_gdf
        )
        print(features_gdf)
        df = parcels.parcel_rollup(
            parcels_gdf,
            features_gdf,
            classes_df,
            country="us",
            calc_buffers=False,
            primary_decision="largest_intersection",
        )
        print(df.T)
        assert df.loc[0, "building_count"] < 1

    def test_rollup_snake_geometry(self, cache_directory: Path):
        """
        There has been a particular challenge in past for clipped areas for connected classes working correctly with
        very thin snake like geometries. Test an example here.
        Args:
            cache_directory:

        Returns:

        """
        aoi = loads(
            "MULTIPOLYGON (((-94.27029850497952 35.32242744039677, -94.26993050494337 35.32247943893384, -94.26985050491517 35.32250343860449, -94.26973350489644 35.32252443813537, -94.26940150487778 35.32256243682338, -94.2693645048664 35.32257243667198, -94.26907050482663 35.32262043549726, -94.26876050479999 35.32266143426713, -94.26845050476024 35.32271043302973, -94.26833750475838 35.32272043258573, -94.26823150473301 35.32274443215611, -94.26806850471318 35.32276943150611, -94.26800550469527 35.32278543124922, -94.26793350468655 35.32279643096209, -94.26775150464475 35.32283643022545, -94.26756550460976 35.32287242947696, -94.26746450456737 35.3229064290575, -94.26733450447178 35.32297542849495, -94.26722150433019 35.32307142797362, -94.2671205041643 35.32318142748579, -94.26689850372237 35.32347042637097, -94.26675250344596 35.32365142564574, -94.26664650330591 35.32374542515277, -94.26658350323537 35.32379342486687, -94.26649250315184 35.32385142446399, -94.26640850309995 35.3238894241061, -94.26634250307846 35.32390742383556, -94.26629850304623 35.32393042364529, -94.26623150301994 35.32395142336817, -94.26604750291777 35.32402742259055, -94.26586250279436 35.32411642179731, -94.26576250271789 35.32417042136314, -94.2651745022106 35.32452241877866, -94.26507450211074 35.32459041833163, -94.26503650208066 35.32461141816614, -94.26473050175652 35.32483041678813, -94.2646595016726 35.32488641646361, -94.26451250148071 35.32501341578173, -94.26442950134877 35.32509941538372, -94.26437850125428 35.32516041513171, -94.26429550106964 35.32527841470466, -94.2642875010262 35.32530541464933, -94.26426150097322 35.32533941451818, -94.26425250092983 35.32536641445898, -94.26419450074398 35.32548341412922, -94.26415750058365 35.32558341389578, -94.2641385004017 35.32569541372093, -94.26411150023205 35.32580041352156, -94.26405150012658 35.32586841322834, -94.26396850006792 35.32590941287084, -94.26385850004486 35.32593041242726, -94.26371649999051 35.32597241184104, -94.26364149995553 35.32599841152801, -94.26357749990453 35.32603341124918, -94.26351549983193 35.32608141096627, -94.2634734997572 35.32612941076054, -94.26346149970074 35.32616441068244, -94.26338549944325 35.32632541024276, -94.26335149907419 35.3265514099061, -94.26330549869799 35.32678240951849, -94.26324749846628 35.32692640916361, -94.26320549832195 35.32701640891956, -94.26316349822538 35.32707740870182, -94.26312149814365 35.32712940849233, -94.26302949802211 35.32720840806505, -94.26297049797158 35.32724240780622, -94.2628614979143 35.32728340734784, -94.26278649788669 35.32730440703897, -94.26269449787227 35.32731840667076, -94.26256849787261 35.32732540617759, -94.26241749789688 35.3273194055997, -94.26223249800712 35.32726340493608, -94.26218549806786 35.3272294047855, -94.26212349817652 35.32716740460254, -94.2620814983396 35.32707140452791, -94.26204749860467 35.32691340454078, -94.26202649902103 35.32666340468783, -94.26201449916644 35.32657640472087, -94.26200149921743 35.32654640469809, -94.26197249942093 35.32642540469652, -94.26194749951463 35.32637040465023, -94.26190449962014 35.3263094045399, -94.26182049976468 35.32622740429053, -94.26163550003591 35.32607540371527, -94.26151850025228 35.32595240337606, -94.26148450032389 35.32591140328237, -94.2614095005145 35.32580140309349, -94.26140050056199 35.32577340308438, -94.26137550064441 35.32572540303174, -94.26130050095981 35.32554040291156, -94.26127050110767 35.32545340287538, -94.26122350140206 35.32527940285329, -94.26119050171327 35.32509440289522, -94.26116450181925 35.32503240285172, -94.26112350191693 35.3249764027448, -94.26098550213696 35.32485340232522, -94.26093150220628 35.32481540215176, -94.26075350243278 35.32469140157892, -94.26061950255625 35.32462640112187, -94.26053550258742 35.32461340080988, -94.26026950260253 35.32462239977599, -94.26013250265646 35.32459939926887, -94.25997350276641 35.32454439870636, -94.25988150285714 35.32449639839575, -94.25973050305673 35.32438739791378, -94.2595875031887 35.32431839742596, -94.25951250320908 35.3243113971433, -94.25938650320182 35.32432439664559, -94.25928550316655 35.32435239623047, -94.25920950311658 35.3243873959053, -94.25913450303969 35.3244383955692, -94.25910950300229 35.3244623954507, -94.25904250287238 35.3245443951169, -94.25897550270547 35.32464839476279, -94.2588205020143 35.32507039377614, -94.2587915018143 35.32519139355279, -94.25876950173104 35.32524239342089, -94.25874450157251 35.32533839323594, -94.25870350123083 35.32554439288782, -94.25869950106325 35.32564439278022, -94.25872450084218 35.32577439275684, -94.25872450078337 35.32580939272454, -94.25877050045605 35.32600139272509, -94.2588175003017 35.32609039282446, -94.2588465001778 35.32616239287003, -94.25885050011357 35.32620039285047, -94.25882549998852 35.32627639268389, -94.25878349990711 35.32632739247472, -94.25862949972114 35.32644739176953, -94.25858649967839 35.32647539157772, -94.25851549957113 35.32654339124085, -94.25845249933865 35.32668539086656, -94.25844049920863 35.32676339074827, -94.25843949915154 35.326797390713, -94.25846549886981 35.32696339066023, -94.25844849870998 35.32705939050602, -94.25838249861211 35.32712139019382, -94.25828949852843 35.32717638978384, -94.25812149845189 35.32723138908404, -94.25784449834005 35.32731338793817, -94.2577854982984 35.32734138768433, -94.25753749822056 35.3274013866707, -94.25739249818525 35.32743038608363, -94.25729549815377 35.32745438568661, -94.25722449814184 35.32746538540214, -94.25708949810206 35.32749638485178, -94.25655349787927 35.32765738263132, -94.2563764978142 35.32770538190276, -94.25621149776151 35.32774538122798, -94.25610049773263 35.32776838077763, -94.25586449762679 35.32784337979581, -94.25572249754616 35.32789837919587, -94.25565749749087 35.32793437891115, -94.25555049739843 35.32799437844169, -94.25548749732067 35.32804337815254, -94.25535349712321 35.32816637752003, -94.25526149693204 35.32828337705536, -94.25522449677851 35.32837537682658, -94.25521149659733 35.32848237667661, -94.25521149646781 35.32855837660576, -94.25522849629249 35.32866037657648, -94.25524049625059 35.32868437660051, -94.25526249610049 35.32877137660454, -94.25537549549486 35.32912137671552, -94.25542149530378 35.32923137679103, -94.25548949506502 35.32936837692653, -94.25557649479427 35.32952337711887, -94.25559749472794 35.32956137716478, -94.25561849462929 35.32961837719296, -94.25562749441761 35.3297423771124, -94.25561449404945 35.32995937686007, -94.25563749371254 35.33015637676577, -94.25570749333141 35.33037737683113, -94.25573349313051 35.33049437682293, -94.255771492701 35.33074537673655, -94.25577949245046 35.33089237663081, -94.25575949222912 35.33102337643147, -94.25569949198832 35.33116737606503, -94.25555949151459 35.33145137525827, -94.25554649144912 35.33149037517154, -94.25553649139698 35.33152137510396, -94.25553049135313 35.3315473750565, -94.25545549135467 35.33154937476399, -94.25547949145864 35.33148737491471, -94.25568049232815 35.33096837617659, -94.25569349268611 35.33075737642332, -94.25554849386388 35.33007137650033, -94.25553149400643 35.32998837651184, -94.25553949466614 35.32960037690404, -94.25529649539232 35.32918537635029, -94.25521249582356 35.32893637625738, -94.25513849648429 35.32855237632909, -94.25519949700737 35.32824237685386, -94.25549749750019 35.32793737828994, -94.25566149761298 35.32786237899358, -94.25585949774795 35.32777237984253, -94.25713049826395 35.32739838510101, -94.25818649859218 35.32714438941547, -94.25836149874871 35.32704139018658, -94.25837249939198 35.32665839058256, -94.2584224996141 35.32652339090021, -94.25873350015245 35.32618439241356, -94.25862350103935 35.32566339246944, -94.25869250179821 35.32520739315612, -94.25890850273485 35.3246353945164, -94.25913650321053 35.32433639567085, -94.259373503329 35.32424939666451, -94.25959250328687 35.32425939749946, -94.26006750281586 35.32450839910181, -94.26039750269649 35.32455740032918, -94.2606165026566 35.32456640116528, -94.26077150253072 35.32463140170334, -94.26093150228284 35.32476940219389, -94.26121250184555 35.32501340305422, -94.26147250052449 35.32579140334572, -94.26198949954023 35.3263524048288, -94.26213749839712 35.32703340477887, -94.26221749818242 35.32715840497389, -94.26239349800174 35.32725740556349, -94.26274049796541 35.32725940690224, -94.26300349814177 35.32713740802922, -94.26314049842721 35.32695640872305, -94.26337649982482 35.32609441041796, -94.26352750002006 35.32596641111706, -94.26394250014627 35.32586341281223, -94.26403450025825 35.32578941323435, -94.26422950111785 35.32525341447267, -94.26467550185441 35.32477441662669, -94.2657895028502 35.32408742154212, -94.2665385033513 35.32372542475466, -94.26674650362054 35.32354442571889, -94.26712950434647 35.32306842762195, -94.26738950461565 35.32288242879023, -94.26784150475311 35.32276243063833, -94.26930350495296 35.32252343648086, -94.27029750507889 35.32236544044803, -94.27029850497952 35.32242744039677)))"
        )
        country = "us"
        packs = ["building", "vegetation"]
        aoi_id = 51310013081068
        survey_resource_id = "6d1db358-f051-567c-b255-0428f66a9e08"

        parcels_gdf = gpd.GeoDataFrame(
            [{"geometry": aoi, "aoi_id": aoi_id, "survey_resource_id": survey_resource_id}], crs=API_CRS
        )

        classes_df = pd.DataFrame(
            [
                {"id": BUILDING_ID, "description": "building"},
                {"id": VEG_MEDHIGH_ID, "description": "Medium and High Vegetation (>2m)"},
            ]
        ).set_index("id")

        feature_api = FeatureApi(cache_dir=cache_directory, compress_cache=True, workers=4)
        features_gdf, metadata, error = feature_api.get_features_gdf(
            aoi, country, packs, aoi_id, survey_resource_id=survey_resource_id
        )
        features_gdf = parcels.filter_features_in_parcels(features_gdf, aoi_gdf=parcels_gdf, region=country)

        df = parcels.parcel_rollup(
            parcels_gdf,
            features_gdf,
            classes_df,
            country="us",
            calc_buffers=False,
            primary_decision="largest_intersection",
        )
        df["parcel_area_sqm"] = parcels_gdf.to_crs(AREA_CRS["us"]).area
        df["pct_tree_cover"] = (
            df["medium_and_high_vegetation_(>2m)_total_clipped_area_sqft"] / SQUARED_METERS_TO_SQUARED_FEET
        ) / df["parcel_area_sqm"]
        print(metadata)
        print(df.T)
        features_gdf.drop(columns="attributes").to_file(cache_directory / "snake_test_features.geojson")
        np.testing.assert_approx_equal(df.loc[0, "parcel_area_sqm"], 17236.0, significant=5)
        assert df.loc[0, "pct_tree_cover"] < 1
        np.testing.assert_approx_equal(df.loc[0, "pct_tree_cover"], 0.81, significant=2)

    def test_large_query_geoid_11179800001006(self, cache_directory: Path):
        """
        Test a particular US census block.

        Returns:

        """
        aoi = loads(
            "MULTIPOLYGON (((-86.77140591336757 33.29757306989932, -86.77125291297651 33.29815506834493, -86.77127391294874 33.29817906831292, -86.77133991283127 33.29829306812129, -86.77136091277683 33.29835106800906, -86.77137591271322 33.29842406785431, -86.77137791262065 33.29854106758032, -86.77136391254038 33.29865106730399, -86.77134891251312 33.29869406718468, -86.77126391240365 33.29888006664464, -86.77111191223987 33.29917206577411, -86.77103491217103 33.29930206537514, -86.77086891203777 33.29956306455985, -86.77079891198875 33.29966406423726, -86.7706449118897 33.29987506355348, -86.77053091183855 33.30000306311373, -86.77038291174422 33.30020506245734, -86.77029991167046 33.30034506202551, -86.77020591157259 33.30052206149237, -86.77006491139581 33.30082606060045, -86.76999091132221 33.30096106019033, -86.76989291125363 33.30110305973439, -86.76977991120874 33.30122305931226, -86.76969191117885 33.3013100589987, -86.76930091104921 33.30169305761192, -86.76909491099502 33.30187705692242, -86.76894091096747 33.301998056446, -86.76882291095707 33.3020770561134, -86.7684559109412 33.30230205512728, -86.76823591090997 33.30246505446789, -86.76810291087756 33.30258105402701, -86.76797791083399 33.30270705357164, -86.76788691079354 33.30281005321288, -86.76781691075333 33.3029010529085, -86.76768491069971 33.30304405240281, -86.76738691063242 33.30329805142528, -86.76665091052273 33.30385404917609, -86.76642391048573 33.30403004846968, -86.76632091045259 33.30413104809772, -86.76625191041781 33.30421504780895, -86.76613091033387 33.30439204723039, -86.76597691021576 33.30463204645773, -86.76587691014909 33.30477504598663, -86.76571591006903 33.30497004531295, -86.76559291002422 33.30509804484863, -86.76538090999931 33.30525104421173, -86.76528190996487 33.30535204384233, -86.76512990986507 33.30556804312661, -86.76482090959641 33.30609304146041, -86.76473490954342 33.30621104106454, -86.76467490952675 33.30626704085228, -86.76458490950949 33.30634104055812, -86.76447190947547 33.3064500401494, -86.76439490942099 33.30656503977143, -86.76430890938278 33.30666403942091, -86.76423090936724 33.30672903916329, -86.76410290932006 33.30686403867112, -86.76406290929236 33.30692303847615, -86.7640379092619 33.3069770383125, -86.76397390914791 33.30716203777911, -86.76391190911001 33.30724703749238, -86.76386290909946 33.30728903732722, -86.76353290906457 33.30752503632835, -86.76345590905117 33.30758703607809, -86.76337890903088 33.30765803580564, -86.76327890898774 33.30777203539826, -86.76322690895638 33.30784303515759, -86.76313690887204 33.30800503464495, -86.76310090881665 33.30809803437064, -86.76307690876439 33.30818003413871, -86.76305090868694 33.30829603382072, -86.76303890862937 33.30837803360416, -86.76302590850311 33.3085500331655, -86.76300990826691 33.30886703236715, -86.76299490815313 33.30902403196255, -86.76299290804805 33.30916203162137, -86.76299990797358 33.30925503140227, -86.76302090791582 33.309318031275, -86.76305990784226 33.30939103114657, -86.76313690774712 33.30947003105289, -86.76335090753901 33.30961603097353, -86.76373590722788 33.30979603103457, -86.76387790710648 33.3098710310364, -86.76406390693752 33.30998203100803, -86.76414190687632 33.31001603102692, -86.76420390683226 33.31003703105666, -86.76429390677795 33.31005503113029, -86.76435790674535 33.31006003120176, -86.76450290669024 33.31004703142288, -86.76465290664669 33.31001603169439, -86.76478890662491 33.30996503199611, -86.7651359066126 33.30977903290118, -86.76531190661231 33.30967703337831, -86.76550890660737 33.30956903389689, -86.76601590660664 33.30927603526585, -86.76615090662666 33.30917203569264, -86.76643290668854 33.30892903664543, -86.76661490675045 33.30874403732758, -86.76675090678421 33.30862203779767, -86.76682490678995 33.30857203801379, -86.7668809067781 33.30855503812703, -86.76694090675754 33.30854703822369, -86.76703490671323 33.30855003833778, -86.76707690668906 33.30855703837508, -86.767138906642 33.30858203839487, -86.76724090655142 33.30864003838692, -86.76728090650951 33.30867103836393, -86.76734090642607 33.30874403826578, -86.76739490633209 33.30883403811909, -86.76744190621713 33.30895503788891, -86.76746490610903 33.30908103761568, -86.76746690604291 33.3091650374163, -86.76744490596505 33.30927803711602, -86.76741090589881 33.30938303681936, -86.76736590584757 33.30947503653957, -86.76727190580282 33.3095870361478, -86.76719790579872 33.30963503593591, -86.76711390580076 33.30968103571576, -86.76700990581097 33.30972803546703, -86.76693590580852 33.30977403525973, -86.76690290580005 33.30980403514437, -86.76686590577948 33.30985203498039, -86.76685290575429 33.30989203486698, -86.76684790571548 33.3099450347327, -86.76685090568158 33.30998703463528, -86.76686390563849 33.31003503453658, -86.76688790560522 33.31006403449801, -86.76691490556057 33.31010603443205, -86.76695490551386 33.31014303439513, -86.76700390546706 33.31017503438211, -86.76708990540509 33.31020503442232, -86.76713690537858 33.31021203446694, -86.76718190535989 33.31021003453063, -86.76724290533785 33.31020303462731, -86.76731690532323 33.31017903478186, -86.76735590532041 33.31016003487859, -86.76740190533076 33.31012003503494, -86.76756690540805 33.30992503571937, -86.76761490542064 33.30988103588774, -86.76767590541962 33.30984703604897, -86.76773990540799 33.30982503618529, -86.76778890539302 33.30981603627082, -86.76783590537271 33.30981503633451, -86.76791990532951 33.30982203642724, -86.76798890528765 33.30983603648365, -86.76805190523605 33.30986603649386, -86.76810790517983 33.30990603647102, -86.76818190508591 33.30998403638061, -86.76826490494908 33.31011203618235, -86.7683109048825 33.31017103610117, -86.76836290482488 33.31021503606382, -86.76878590442824 33.31048003598383, -86.76888090432159 33.31056203591252, -86.76892690425801 33.31061703584148, -86.76896690419169 33.31067903574594, -86.76899290411293 33.31076503557474, -86.76901390400529 33.31089103530152, -86.76899990394769 33.31097303508726, -86.76898090391339 33.31102803493093, -86.76894690388201 33.3110880347428, -86.76890790385519 33.31114503455527, -86.76885190383298 33.31120603433569, -86.7687659038134 33.31128103404295, -86.76845890376063 33.31152703304884, -86.76832590374376 33.31162603263577, -86.76828390373863 33.31165703250595, -86.76819790371462 33.31173803219792, -86.76816290369777 33.31178003205088, -86.76814790368509 33.31180503197105, -86.76808690360703 33.31194103156408, -86.76807590357465 33.31198903143438, -86.76807490353156 33.31204503129879, -86.76808690348263 33.31210103118043, -86.76811090342974 33.3121550310829, -86.76815290336172 33.31221803098768, -86.76823790324845 33.31231403087077, -86.76834190312269 33.3124150307673, -86.7684949029575 33.31253803067681, -86.76867890275167 33.31269503054679, -86.76880490259082 33.31282803039721, -86.76887290248823 33.3129200302683, -86.76891590240554 33.31300103013231, -86.76894890233284 33.31307502999979, -86.76907990196611 33.31346902923481, -86.7691239018712 33.31356502906483, -86.76916290181209 33.31361802899086, -86.76922990173786 33.31367402894741, -86.76930990165685 33.31373102891921, -86.76942390155254 33.31379802891308, -86.76945790151129 33.31383102888024, -86.76946490148312 33.31386302881346, -86.76945190146094 33.31389902871012, -86.76912390141517 33.31415002766875, -86.76899790137928 33.31427002721198, -86.76888090132452 33.31440902672161, -86.76870590121931 33.31464702591553, -86.76859090115913 33.31479202541264, -86.76832590110874 33.31501302452334, -86.76804490106673 33.31523302361328, -86.76791190105712 33.3153240232136, -86.76769090105384 33.31545902258782, -86.76751590103027 33.31559302202668, -86.76743590100115 33.3156780217129, -86.76730290093511 33.31584202113628, -86.76718890081541 33.31606402044563, -86.76709990073313 33.31622301994053, -86.7669889005998 33.31646101921449, -86.76695790055444 33.31653801898622, -86.76691190047183 33.31667201859983, -86.76680890025126 33.31701801762336, -86.76676890014809 33.31717501718934, -86.76667189984364 33.31762601596666, -86.76664989975393 33.31775501562471, -86.76662689970959 33.3178260154214, -86.76659689967863 33.31788401523959, -86.76654689964683 33.31795501499854, -86.76648689961978 33.31802601474346, -86.76642889961187 33.31807101455391, -86.76632789961552 33.31812701427776, -86.76623389962903 33.31816601405236, -86.76619989963635 33.31817701397829, -86.76609389965935 33.31821101374813, -86.76595389969589 33.3182480134631, -86.765583899753 33.31839701258518, -86.76542989975596 33.31848601215393, -86.76534189975062 33.31854601188519, -86.76524989973791 33.31861801158155, -86.7651758997214 33.31868401131759, -86.7649128996227 33.31897101025096, -86.76482889960947 33.31903900996754, -86.76476789961018 33.31907500979424, -86.76470289961661 33.3191060096274, -86.76460189963761 33.31914000940264, -86.76449389967188 33.31916100919963, -86.764291899757 33.31917300888627, -86.76422989978133 33.31917900878449, -86.76409589981623 33.3192150085081, -86.76403189982695 33.31924000835701, -86.76395989982052 33.31929200812871, -86.76392789979472 33.31934500795416, -86.7639028997564 33.31941000776013, -86.76387489967182 33.31953700741033, -86.76386089960064 33.31963800714371, -86.76385189958103 33.31966900705523, -86.76381789956233 33.31971400689723, -86.76376989955249 33.31975600672676, -86.76371989954595 33.31979500656076, -86.76363489955037 33.3198410063281, -86.76355589956431 33.319871006143, -86.7633058996246 33.31994500560824, -86.76324289963266 33.31997300545047, -86.76317789963548 33.32000900527027, -86.76312389963005 33.32004900509571, -86.7630728996179 33.32009600490822, -86.76302389959483 33.32015600469165, -86.76298289956033 33.32022600446179, -86.76296389952471 33.32028400429255, -86.7629098993613 33.32053000361217, -86.76285689924804 33.32071000309497, -86.76282889919831 33.32079200285374, -86.76275789914656 33.32090300248002, -86.76269089910599 33.32099700215358, -86.76261289907143 33.32109000181382, -86.76256289905582 33.32114100161707, -86.76230489900726 33.32136300070236, -86.76208389895829 33.32156299989359, -86.7619968989266 33.32165799953484, -86.76196889891609 33.32168899941828, -86.76189189885679 33.32181399899972, -86.7618348987957 33.3219289986343, -86.76176189871619 33.32207799816182, -86.76152789842531 33.32260299652886, -86.76132789821824 33.32299799526375, -86.76126089815986 33.32311599487502, -86.76119589811356 33.32321699453099, -86.76107289804516 33.3233829939418, -86.76099189801651 33.32347099360634, -86.760913897994 33.32354899329986, -86.76082089797949 33.32362599297385, -86.76070589797393 33.32370499261075, -86.76060389798279 33.32375699233329, -86.76045789800658 33.3238169919718, -86.76035489803577 33.32384299175718, -86.76018289809411 33.32387399142947, -86.75982689823246 33.32391499080853, -86.75969089827596 33.32394299054049, -86.75954289831374 33.32398599021753, -86.75944589833108 33.32402398998132, -86.75939289833583 33.32405098983666, -86.759291898333 33.32411798952214, -86.75918989831325 33.32420798914863, -86.75900889823852 33.32441998835471, -86.75885089819242 33.32457998772371, -86.75878789817773 33.32463898748388, -86.75868289816789 33.32471798713242, -86.75860889816761 33.3247649869063, -86.75848489818819 33.32481598659668, -86.75837689821401 33.32484998635302, -86.75812089829252 33.32490798583169, -86.75753089850134 33.32500498472118, -86.7571698986015 33.32510098394887, -86.7570428986313 33.32514198365879, -86.75622489878462 33.32545798165636, -86.75587789889143 33.32553698094452, -86.75541789906313 33.32560198010076, -86.75528489910511 33.32563097983076, -86.75516489913637 33.32566597956468, -86.75509489914741 33.32569597938505, -86.7549828991598 33.32575097907984, -86.75490989916096 33.32579597885754, -86.75483889915071 33.32585497860251, -86.75478089912824 33.32592197834629, -86.75472489908147 33.32601997801415, -86.75452089887233 33.32642897667062, -86.75446289883045 33.3265219763476, -86.75440189879671 33.32660597604295, -86.7541948987066 33.32685897508993, -86.75386389858502 33.3272339736391, -86.7534688984522 33.32766597194395, -86.75335689843223 33.3277649715227, -86.75319189841331 33.32789697093716, -86.753086898414 33.32796397060784, -86.75299289842064 33.32801597033333, -86.75289589843176 33.32806397006447, -86.75273589846519 33.32812296967255, -86.75255189850878 33.32818396923917, -86.75223089859492 33.328276968517, -86.75203789864005 33.32834196805912, -86.75185989866597 33.32842296758232, -86.7517408986775 33.32848496724321, -86.75160189866821 33.32858796676806, -86.75149689865135 33.32867896637489, -86.75141589862662 33.32876496603063, -86.75133289858952 33.3288689656368, -86.75122589852835 33.32902096508251, -86.75105089841249 33.32929096411986, -86.75093189831793 33.32949596340934, -86.75075789812605 33.32986796218252, -86.75073889809144 33.32992696200094, -86.75072389804599 33.32999796179435, -86.75069789790398 33.33020596121641, -86.75069989785466 33.33027096105126, -86.75071289780892 33.330323960934, -86.75074089774688 33.33038896080861, -86.75078189766808 33.33046796066689, -86.75084689756785 33.33055996052839, -86.75093389746426 33.3306419604496, -86.75101689737528 33.33070696040875, -86.75122089717709 33.33083896038052, -86.75133689705173 33.330930960321, -86.75148689684553 33.33110896009212, -86.7515148967916 33.33116295999586, -86.75154789670459 33.33125795980162, -86.75155889664258 33.33133395962261, -86.75155789659911 33.33139295946891, -86.75153989653414 33.33149195918593, -86.75151489648609 33.33157295893847, -86.75138289632895 33.33187095796595, -86.75125489621765 33.33210495716407, -86.75120689616534 33.33220695682635, -86.75117089611605 33.33229695653814, -86.75115389607012 33.33236995632329, -86.75114989601398 33.33244795611559, -86.7511598959458 33.33253295591157, -86.75119889576203 33.33275395540141, -86.75121889564866 33.33289295507357, -86.75122589556999 33.33299395482364, -86.75121589552202 33.33306495462471, -86.75119189547507 33.33314395438332, -86.75114389542884 33.33323795406568, -86.75108489539851 33.33331795376697, -86.75075789526726 33.33371195223712, -86.7506068952142 33.3338839515555, -86.75037989514921 33.33412295058037, -86.75027789512559 33.33422295016103, -86.75000389507906 33.33446894909213, -86.74991889530685 33.33455594871266, -86.7498898953756 33.33460494853266, -86.74985389543932 33.33469494823357, -86.74983489545033 33.33477294799659, -86.74981989543353 33.33486894772028, -86.74981689533399 33.33501794732826, -86.74982889523935 33.33508694717109, -86.74985089510099 33.33516594700616, -86.74990289482561 33.33528294679707, -86.75023089400298 33.33576594608615, -86.75025289397269 33.33579194605372, -86.74930389635246 33.33615794344264, -86.74924989653316 33.33617894329054, -86.74902989725346 33.3362859426155, -86.74887889773883 33.33637194211995, -86.7487298982082 33.33646994159723, -86.74832389946532 33.33676794009401, -86.74813190007484 33.33688893943519, -86.7477549012992 33.33708993823704, -86.7476889015072 33.3371339380047, -86.74760590176031 33.33720093768252, -86.7472419028412 33.33753593616468, -86.74714990312454 33.33760693581652, -86.74699890360969 33.33769593531613, -86.74669690462321 33.33781493446755, -86.74637090570283 33.33796393349949, -86.74625390608431 33.33802593313038, -86.74587290731623 33.33824293189044, -86.74573090777984 33.33831793144375, -86.74548390859688 33.33843393070409, -86.74525290936596 33.33853593002933, -86.74503591009278 33.3386259294107, -86.74464391141171 33.33878092831334, -86.74447091200857 33.3388289278814, -86.74437891232992 33.33884892766586, -86.74429991261024 33.33885992749656, -86.7441839130355 33.3388569272968, -86.7440999133504 33.33884492717716, -86.74399091376993 33.33881392706123, -86.74385391431068 33.33875592696388, -86.74357691542687 33.33860592685002, -86.74336091627865 33.33851492669468, -86.74327891659659 33.33848792661646, -86.74317291700342 33.33845892650032, -86.74299191768293 33.33843092624721, -86.74288191809583 33.33841392609329, -86.74264191897981 33.33840092569623, -86.74232392012492 33.33842092507576, -86.74215192073027 33.33845192468911, -86.74194092145687 33.33851292415667, -86.74186392171812 33.33854092394796, -86.74181092190092 33.33855592381506, -86.74173892214388 33.33858392361535, -86.74142692316072 33.338757922617, -86.7412599237018 33.33885592207074, -86.74094292471356 33.33906492097649, -86.74074992532461 33.33919992029125, -86.74054992595249 33.33934791956114, -86.7403389266057 33.33951791875653, -86.74018692706177 33.33966191812302, -86.74000492758985 33.3398609172981, -86.73979392817162 33.34013691622861, -86.73967092850347 33.34030891557761, -86.73960692866834 33.34040991521016, -86.73954792879925 33.34053391479384, -86.7394949289029 33.34066591436823, -86.73947092890391 33.3407929140067, -86.73946692885296 33.34088891375849, -86.73946992860557 33.34123491289477, -86.73945892856314 33.34135591257145, -86.73940592860835 33.34157391193027, -86.73928592883983 33.34187891095272, -86.73920792895451 33.34212991018524, -86.73918392892458 33.34230290970885, -86.73916192883758 33.34254890905287, -86.73913492881057 33.34273390854122, -86.73910092883925 33.34287490812774, -86.73904892890756 33.34305490758493, -86.73900992894249 33.34321390711773, -86.73898892894542 33.34332290680757, -86.73897992892132 33.34340690658117, -86.73897692886838 33.34350090634018, -86.73898592870803 33.34368790588697, -86.73898692868804 33.34371190582853, -86.73901392848805 33.34385990550462, -86.7391189277935 33.34431290455169, -86.73914892754073 33.34452190407963, -86.73915292738566 33.34472790356972, -86.73914992738574 33.34474390352437, -86.73914192737841 33.34479790337499, -86.73912792739655 33.34484690322767, -86.73905192757827 33.34499090273419, -86.73899792773426 33.34505390248228, -86.73889392805599 33.34514390207582, -86.73882692827078 33.34519090184156, -86.7383489298729 33.34542490042531, -86.73812593060667 33.34555489971314, -86.73802893091755 33.34562389937234, -86.73794893116582 33.34569289906114, -86.73784293147963 33.34580689859253, -86.73759893218813 33.3460908974607, -86.73735393292118 33.34634489640305, -86.73718193346525 33.34647989576915, -86.73707293381391 33.34655989538149, -86.73693993424523 33.34664889493011, -86.73640693600419 33.34696189323164, -86.7361919367237 33.34707389258271, -86.73596793746279 33.34720689186651, -86.73582193794057 33.34729989138427, -86.73567093842234 33.34741489083908, -86.73554593881416 33.34752089036113, -86.73541293922317 33.34764588982262, -86.73523793973824 33.34784588902638, -86.73503994028147 33.34813288797584, -86.73497194046261 33.34823988759432, -86.73453694157064 33.34900488495773, -86.73445594178746 33.34913188450556, -86.73427794229652 33.3493618836337, -86.73418594257083 33.34946388322503, -86.73398694317802 33.34966388239245, -86.73388694349016 33.34975388200021, -86.73371294404757 33.34988888137131, -86.73357694448832 33.34998688089853, -86.7332279456359 33.3502138797466, -86.73310194605888 33.3502828793629, -86.73292794664906 33.35036887885611, -86.73278794713518 33.35042087849077, -86.73264194764877 33.35046487813504, -86.732510948121 33.35048687785888, -86.73229694890006 33.35051087743715, -86.73209494964813 33.35051387708734, -86.7317139510909 33.35046987654925, -86.73155095171843 33.35043487635861, -86.73141395223981 33.35041487617521, -86.7312839527294 33.35040387598167, -86.7311099533715 33.35040987567186, -86.73102395368001 33.35042687548447, -86.73095695391473 33.35044887531708, -86.73076695457188 33.35052487480926, -86.73066195492621 33.35058087449453, -86.73027495622426 33.35080087330228, -86.73010195680195 33.35090387275842, -86.72952795871973 33.35124587095528, -86.72875896130324 33.35168686858621, -86.72817996324598 33.35202686678674, -86.72786796429017 33.35221586580438, -86.72730296617497 33.35257086399708, -86.72684196770452 33.35287686248537, -86.72671696811518 33.35296686205893, -86.72652596872746 33.35312986134635, -86.72643896900047 33.35321386099839, -86.72637396919522 33.35329186070187, -86.72627596947837 33.35342686021305, -86.72605597010416 33.35374685907571, -86.72600897024365 33.35380585885541, -86.72586997067854 33.35394385829179, -86.72576997100198 33.35402585792797, -86.72566197136086 33.35409885757264, -86.72550497189248 33.35418885709495, -86.72531897253376 33.354276856574, -86.72469197474344 33.35449585500564, -86.72447697550699 33.35456185449011, -86.72420797647239 33.35462785388506, -86.72403697710465 33.35463885357454, -86.72394697744565 33.354630853444, -86.7238349778742 33.35461385329843, -86.72375797817202 33.35459685321097, -86.72367397850304 33.35456785314044, -86.72347097931244 33.35448185300795, -86.72322798028993 33.35436385288475, -86.72314198062884 33.35433385281306, -86.72305998094753 33.35431285272654, -86.72297698126216 33.35430485260742, -86.72289798155896 33.35430185248303, -86.72282998180874 33.35430885235314, -86.7227519820923 33.35432185219238, -86.72197398488811 33.35450885045351, -86.72178098558388 33.35455185003057, -86.72167698596418 33.35456584982439, -86.72154198646498 33.35457184958573, -86.72144298683483 33.35457184942111, -86.72136298713831 33.35456384930698, -86.72118498783001 33.35451784911981, -86.72109798817412 33.35448484905318, -86.72099098860159 33.35443684898867, -86.7208639891176 33.35436484894745, -86.72072098971121 33.35426184895276, -86.72065598998974 33.35419984899092, -86.72056199040594 33.35408684910111, -86.72039599114835 33.35387384932704, -86.72022599188348 33.35369884945607, -86.72010799238006 33.35360084949006, -86.71995099303513 33.35347984951281, -86.71982399357009 33.35337284955249, -86.71974799389361 33.35330284959019, -86.71967499421098 33.35322384965404, -86.71960799452023 33.35311984978662, -86.71956299474 33.35302884992539, -86.71953999486404 33.35296185004449, -86.71951299503039 33.35284685026978, -86.71949799514574 33.35274285048926, -86.71947599527283 33.35266385063822, -86.71945199539937 33.35259885075081, -86.719395995653 33.35251985084261, -86.71935599582422 33.35248085086712, -86.71930699602495 33.35244885086001, -86.71925399623485 33.3524268508227, -86.71914899663835 33.35240485069799, -86.71910599680017 33.35240185063279, -86.71904499702271 33.35240985051156, -86.71876999800982 33.35247484989701, -86.71825399983763 33.35264084864148, -86.71800500071613 33.35272784801998, -86.71791800102278 33.35275884780147, -86.71775100161632 33.35280984740209, -86.71762600205676 33.35285484708729, -86.71751500244518 33.35289984679605, -86.71742500275256 33.35294984652845, -86.71734000303576 33.35300984624585, -86.71725600331079 33.35307784594638, -86.71683900465692 33.35345184437617, -86.71674600496647 33.35351884406455, -86.71668300518259 33.35355284388002, -86.71641600611751 33.35366284317782, -86.71616300701541 33.35374584256226, -86.71579800830344 33.35387984164165, -86.71553600923342 33.35396684100263, -86.71509401079446 33.35412883989011, -86.71467101227556 33.35430883876873, -86.71428701360104 33.35450883766723, -86.71396301471401 33.35468883671282, -86.71364501579065 33.35489583570703, -86.71335001678473 33.35509783475198, -86.71308501766671 33.35530083384528, -86.71289401830616 33.35544083320681, -86.71250701958925 33.35574983185633, -86.71220502058041 33.35601183075591, -86.712053021088 33.3561278302389, -86.71194602144895 33.35620282989038, -86.711851021773 33.35626282959618, -86.71169802230112 33.35634782914917, -86.71156402276877 33.3564128287794, -86.71142802325032 33.35646582843403, -86.71134502354074 33.35650482820807, -86.71120302404493 33.35655782785294, -86.71106402454372 33.35659982752803, -86.71093802500063 33.3566288272543, -86.7107950255229 33.35665482695949, -86.71054702643555 33.35668682647825, -86.71040702695501 33.35669682622499, -86.71023702759049 33.35669982593834, -86.71006402824094 33.35669582566273, -86.70993702872099 33.35668782547192, -86.70937903085684 33.35660082475108, -86.70901503225254 33.35653882429234, -86.70874703329241 33.35646882400957, -86.70865403365715 33.35643682392886, -86.70859503389092 33.35641182388822, -86.70807703596375 33.35615082362467, -86.70796503640899 33.35609982355515, -86.70786303680966 33.35606282347036, -86.70775703722367 33.35602882337213, -86.70764403766037 33.35600182324651, -86.70752203812467 33.35598682307892, -86.70731503889803 33.35598982273027, -86.70711803961699 33.3560268223215, -86.707015039984 33.35606382206807, -86.70689904039178 33.35611682175719, -86.70669004110589 33.35625382110394, -86.7065740414919 33.35635082069447, -86.70627204248727 33.35662381958315, -86.70592704366211 33.35686181848097, -86.70562504471314 33.35702581761669, -86.70549304517459 33.3570938172474, -86.70506504667351 33.35730981606102, -86.70486104738335 33.35742281547362, -86.7047420477941 33.35749581511536, -86.70461704822213 33.35757981472274, -86.70449004865237 33.35767481430248, -86.70434104914587 33.35780981375715, -86.70392405051027 33.35822481214984, -86.70384105077605 33.35831981180261, -86.70375205106545 33.35841281145017, -86.70363205144875 33.35855281094267, -86.70351905180353 33.35869781043577, -86.70337805223633 33.35889980975673, -86.70316705287027 33.35923180867541, -86.70297205345204 33.35954880765446, -86.70267905432043 33.36003980608972, -86.70247505491734 33.3603998049614, -86.70193105646437 33.3614628017304, -86.70189205657663 33.36153680150399, -86.70169605715417 33.36188080042855, -86.70163305734117 33.36198880008881, -86.70147605782689 33.36221679933344, -86.70135005822456 33.36238379876295, -86.7012130586635 33.36255179817294, -86.7010210592864 33.36277179738079, -86.70085405983714 33.3629447967328, -86.70075706016763 33.36302279640611, -86.70065706051938 33.36307979612108, -86.7005860607745 33.36310879594397, -86.70045406125625 33.36314679564985, -86.69996306306595 33.36325079463847, -86.69965506419742 33.36332479398524, -86.69951906469052 33.36337179366556, -86.69944806494503 33.36340279348449, -86.69932606537564 33.36347079314118, -86.69918006587292 33.36359179264372, -86.6989620666038 33.36379879184383, -86.6987350673693 33.36400579103044, -86.69861806777074 33.36409779064358, -86.69847806825449 33.36420079019635, -86.69828706892807 33.36431178965106, -86.69770207100916 33.36461478806365, -86.69739907207882 33.36479178719889, -86.69734007228335 33.36483478701204, -86.6972200726903 33.36494278658741, -86.69716707286325 33.36500578636669, -86.69697807358965 33.36498478611375, -86.69683107416117 33.36495378594886, -86.69679507429906 33.36495078589849, -86.69671507458661 33.36498678569391, -86.69665507481334 33.36498878559477, -86.69655907518377 33.36497478547351, -86.69632307610205 33.36492278521333, -86.69620907653893 33.36491278505487, -86.69610007695243 33.36491278488261, -86.69588607775636 33.36493078450545, -86.69572407836218 33.3649507842062, -86.69565907859651 33.36497878404293, -86.69559007882897 33.36504578378909, -86.69556807888883 33.3650997836376, -86.69557107885512 33.36515078353213, -86.69559207875575 33.365195783468, -86.69567207842888 33.36524878347963, -86.69568207837915 33.36527578343701, -86.69565607846513 33.36530478333327, -86.69557207877389 33.36532778315107, -86.6954600791951 33.36533678295501, -86.69534907962166 33.36532478280586, -86.6952500800123 33.36529078272315, -86.69520608019018 33.36526578270772, -86.69516508034623 33.36526478264521, -86.69512008050094 33.36530178249432, -86.69507608066404 33.36531078240554, -86.69502408087533 33.36527878239252, -86.69496908108709 33.36527178232087, -86.69487408143208 33.36530778209341, -86.69483208157547 33.36534478194737, -86.69484408149223 33.36543178177875, -86.69482008156126 33.36548278163097, -86.69476208176368 33.36552378145122, -86.69469408202187 33.36552378134408, -86.69461408231913 33.36553878118575, -86.6945140826829 33.3655757809485, -86.69442108301192 33.36563178068151, -86.69432808335777 33.3656487804985, -86.69423908370351 33.36563078039707, -86.69415108405659 33.36558678035311, -86.69410808425548 33.36550378046381, -86.69405108448817 33.36546578045571, -86.69385908525308 33.36538178033354, -86.69378008556619 33.3653507802755, -86.69371608581298 33.36534178019392, -86.6936320861259 33.36535578003144, -86.69354508644081 33.36539177981697, -86.69348908663763 33.36542877964935, -86.69344508679785 33.36544477954569, -86.69340508695402 33.36543477950405, -86.69337508707424 33.36541977948891, -86.69334108720335 33.36541977943531, -86.69331508729441 33.36543777935575, -86.69331308728587 33.36547577927116, -86.69333108719923 33.36551877920736, -86.69334108714555 33.36555577914381, -86.69332508719653 33.3655787790693, -86.69330408727414 33.36558377902549, -86.69324308751807 33.36555477899152, -86.69316908779314 33.36556877884498, -86.69307708812555 33.36560877861441, -86.6929990884133 33.36562877844875, -86.6929060887724 33.36561477833224, -86.69272008953094 33.36549077830458, -86.69267108972625 33.3654687782744, -86.69262508989837 33.36547477818907, -86.69258409003427 33.36552177802398, -86.69254809014576 33.36558177783902, -86.69249709032299 33.36562077767531, -86.69243209057107 33.36561777757937, -86.69237909078915 33.36557777758139, -86.6922720912189 33.3655217775324, -86.69213809176408 33.36543477750694, -86.69206209205555 33.3654277774021, -86.69199909230389 33.36540577734972, -86.6919460925168 33.3653777773259, -86.69185409290186 33.36529177736418, -86.69172509341976 33.3652237773056, -86.69170609349645 33.36521277729903, -86.69163509378502 33.36516677728498, -86.69156809406252 33.36511077729851, -86.69151909427295 33.36505177734676, -86.69149209444173 33.36489177764477, -86.69147509452942 33.36483577773715, -86.69142009477196 33.36475377782471, -86.69136009502563 33.36469077786386, -86.69123409556707 33.36453677799206, -86.69119109574744 33.36449477801328, -86.69103509635534 33.36445477785128, -86.69093009676288 33.36443177773382, -86.69080709722913 33.36443177753895, -86.6906980976497 33.36441377740447, -86.69058009811668 33.36436577731936, -86.69137209511462 33.36436577857451, -86.69138209511027 33.36428477876272, -86.69162509419056 33.36428177915437, -86.69157909684164 33.35831479178968, -86.69158209924159 33.35250580416646, -86.69159010011386 33.35033280880801, -86.69160610325585 33.34262282525723, -86.69190410215505 33.34264082574279, -86.7002830713497 33.34277984016732, -86.70041307240612 33.33947884765852, -86.70038907262284 33.33920084822734, -86.70475205665936 33.33928385583166, -86.704767059212 33.33395086777718, -86.70477206016255 33.33196987221374, -86.70504905916137 33.33197287271755, -86.70894204505647 33.33207987964987, -86.71019404055845 33.3320388820497, -86.71015304260114 33.32838889028832, -86.71108403926448 33.32837989205133, -86.71347403063584 33.32847489630583, -86.7179050146319 33.32865090418415, -86.71791001484287 33.32824390514577, -86.71793901595477 33.32608291025667, -86.71794301619487 33.32563091132189, -86.71794901636756 33.32528591214057, -86.71973300998663 33.32530191548605, -86.71990500936829 33.32530891579569, -86.72011700861277 33.32530591620477, -86.72015200863501 33.32504991687463, -86.72058700707211 33.32506691766025, -86.72232800081728 33.32513292080813, -86.72235200161444 33.32362992442346, -86.72238300268447 33.32162092925478, -86.72238500272437 33.32154092944869, -86.72249900231255 33.32155292963974, -86.72309500019392 33.32155693077818, -86.72666798748639 33.32159093757824, -86.72667698963851 33.31801994620024, -86.7312519734801 33.31802695512404, -86.73138097536717 33.31435496436192, -86.73337596833474 33.31439996820822, -86.73562196043065 33.31442897259051, -86.73603195898588 33.31443697338362, -86.73668695666689 33.31446597461019, -86.73670295644982 33.31470697404195, -86.73695595556518 33.31470097455804, -86.73839695048494 33.31472897734222, -86.7386319496554 33.3147349777926, -86.73863294982364 33.31448197842835, -86.73956794652908 33.3144989802393, -86.73984494766876 33.3114169885384, -86.73990494751192 33.31133998885245, -86.73994394740176 33.31130198902636, -86.74437393182481 33.31144499755645, -86.74848291736991 33.3115780054585, -86.74854391988532 33.30786101518541, -86.74855192019886 33.30739601640325, -86.74836892083488 33.30739501603269, -86.74836292235042 33.30535602128622, -86.74835892317208 33.30425402412389, -86.74863692220015 33.3042700246563, -86.75247191630935 33.30440803018784, -86.75248191662324 33.30398203129443, -86.75251691849068 33.30146203780926, -86.75252791899327 33.30078303956621, -86.75288291884684 33.30077404001712, -86.75290091918461 33.30031204122333, -86.75290591928641 33.30017304158565, -86.75291292181434 33.2967890502696, -86.75909491896283 33.29709605668421, -86.76764791514586 33.29734006606171, -86.76793291502021 33.29734606638007, -86.77149891346606 33.29739707042359, -86.77140591336757 33.29757306989932)))"
        )
        country = "us"
        date_1 = "2021-11-12"
        date_2 = "2021-11-12"
        packs = ["building", "vegetation"]
        aoi_id = 11179800001006

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
        features_gdf = parcels.filter_features_in_parcels(features_gdf, aoi_gdf=parcels_gdf, region=country)

        df = parcels.parcel_rollup(
            parcels_gdf,
            features_gdf,
            classes_df,
            country="us",
            calc_buffers=False,
            primary_decision="largest_intersection",
        )
        print(metadata)
        print(df.T)
        assert (
            df.loc[0, "building_count"] == 25
        )  # Includes some lower confidence ones that had been filtered by the previous thresholds.
