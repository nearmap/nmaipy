from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.wkt import loads

from nearmap_ai import parcels
from nearmap_ai.constants import BUILDING_ID, LAWN_GRASS_ID, POOL_ID, VEG_MEDHIGH_ID, AOI_ID_COLUMN_NAME
from nearmap_ai.feature_api import FeatureApi


@pytest.mark.skip("Comment out this line if you wish to regen the test data")
def test_gen_data(parcels_gdf, data_directory):
    outfname = data_directory / "test_features.csv"
    from nearmap_ai.feature_api import FeatureApi

    packs = ["building", "building_char", "roof_char", "roof_cond", "surfaces", "vegetation"]
    features_gdf, _, _ = FeatureApi().get_features_gdf_bulk(parcels_gdf, packs=packs)
    features_gdf.to_csv(outfname, index=False)


class TestParcels:
    def test_filter(self, features_gdf):
        assert len(features_gdf) == 602
        features_gdf = features_gdf[features_gdf.class_id == BUILDING_ID]
        assert len(features_gdf) == 68

        config = {
            "min_size": {
                BUILDING_ID: 25,
            },
            "min_confidence": {
                BUILDING_ID: 0.8,
            },
            "min_area_in_parcel": {
                BUILDING_ID: 25,
            },
            "min_ratio_in_parcel": {
                BUILDING_ID: 0.5,
            },
        }
        filtered_gdf = parcels.filter_features_in_parcels(features_gdf, config=config)
        assert len(filtered_gdf) == 44
        assert not (filtered_gdf.confidence < 0.8).any()
        assert not (filtered_gdf.unclipped_area_sqm < 25).any()
        intersection_mask = (filtered_gdf.intersection_ratio < 0.5) & (filtered_gdf.clipped_area_sqm < 25)
        assert not intersection_mask.any()

    def test_flatten_building(self):
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
        assert expected == parcels.flatten_building_attributes(attributes, "au")

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
        features_gdf = parcels.filter_features_in_parcels(features_gdf)
        df = parcels.parcel_rollup(parcels_gdf, features_gdf, classes_df, "au", "largest_intersection")

        expected = pd.DataFrame(
            [
                {
                    "building_present": "Y",
                    "building_count": 1,
                    "building_total_area_sqm": 459.2,
                    "building_total_clipped_area_sqm": 438.5,
                    "building_total_unclipped_area_sqm": 459.2,
                    "building_confidence": 0.990234375,
                    "primary_building_area_sqm": 459.2,
                    "primary_building_clipped_area_sqm": 438.5,
                    "primary_building_unclipped_area_sqm": 459.2,
                    "primary_building_confidence": 0.990234375,
                    "aoi_id": "0_0",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 3,
                    "building_total_area_sqm": 778.0999999999999,
                    "building_total_clipped_area_sqm": 562.5,
                    "building_total_unclipped_area_sqm": 778.1,
                    "building_confidence": 0.9999999478459358,
                    "primary_building_area_sqm": 412.7,
                    "primary_building_clipped_area_sqm": 335.7,
                    "primary_building_unclipped_area_sqm": 412.7,
                    "primary_building_confidence": 0.998046875,
                    "aoi_id": "0_1",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "N",
                    "building_count": 0,
                    "building_total_area_sqm": 0.0,
                    "building_total_clipped_area_sqm": 0.0,
                    "building_total_unclipped_area_sqm": 0.0,
                    "building_confidence": 1.0,
                    "primary_building_area_sqm": 0.0,
                    "primary_building_clipped_area_sqm": 0.0,
                    "primary_building_unclipped_area_sqm": 0.0,
                    "primary_building_confidence": 1.0,
                    "aoi_id": "0_2",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "N",
                    "building_count": 0,
                    "building_total_area_sqm": 0.0,
                    "building_total_clipped_area_sqm": 0.0,
                    "building_total_unclipped_area_sqm": 0.0,
                    "building_confidence": 1.0,
                    "primary_building_area_sqm": 0.0,
                    "primary_building_clipped_area_sqm": 0.0,
                    "primary_building_unclipped_area_sqm": 0.0,
                    "primary_building_confidence": 1.0,
                    "aoi_id": "0_3",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 5,
                    "building_total_area_sqm": 1028.7999999999997,
                    "building_total_clipped_area_sqm": 632.9,
                    "building_total_unclipped_area_sqm": 1028.8,
                    "building_confidence": 0.999999999987466,
                    "primary_building_area_sqm": 211.7,
                    "primary_building_clipped_area_sqm": 211.7,
                    "primary_building_unclipped_area_sqm": 211.7,
                    "primary_building_confidence": 0.994140625,
                    "aoi_id": "1_0",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 6,
                    "building_total_area_sqm": 892.8999999999999,
                    "building_total_clipped_area_sqm": 620.9,
                    "building_total_unclipped_area_sqm": 892.9,
                    "building_confidence": 0.9999999999997962,
                    "primary_building_area_sqm": 324.8,
                    "primary_building_clipped_area_sqm": 308.2,
                    "primary_building_unclipped_area_sqm": 324.8,
                    "primary_building_confidence": 0.994140625,
                    "aoi_id": "1_1",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 4,
                    "building_total_area_sqm": 943.2,
                    "building_total_clipped_area_sqm": 635.7,
                    "building_total_unclipped_area_sqm": 943.2,
                    "building_confidence": 0.9999999999272404,
                    "primary_building_area_sqm": 419.5,
                    "primary_building_clipped_area_sqm": 403.4,
                    "primary_building_unclipped_area_sqm": 419.5,
                    "primary_building_confidence": 0.998046875,
                    "aoi_id": "1_2",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "N",
                    "building_count": 0,
                    "building_total_area_sqm": 0.0,
                    "building_total_clipped_area_sqm": 0.0,
                    "building_total_unclipped_area_sqm": 0.0,
                    "building_confidence": 1.0,
                    "primary_building_area_sqm": 0.0,
                    "primary_building_clipped_area_sqm": 0.0,
                    "primary_building_unclipped_area_sqm": 0.0,
                    "primary_building_confidence": 1.0,
                    "aoi_id": "1_3",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 5,
                    "building_total_area_sqm": 873.0999999999999,
                    "building_total_clipped_area_sqm": 719.4,
                    "building_total_unclipped_area_sqm": 873.1,
                    "building_confidence": 0.9999999999992326,
                    "primary_building_area_sqm": 314.0,
                    "primary_building_clipped_area_sqm": 304.8,
                    "primary_building_unclipped_area_sqm": 314.0,
                    "primary_building_confidence": 0.994140625,
                    "aoi_id": "2_0",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 4,
                    "building_total_area_sqm": 528.8000000000001,
                    "building_total_clipped_area_sqm": 492.8,
                    "building_total_unclipped_area_sqm": 528.8,
                    "building_confidence": 0.9999999996944098,
                    "primary_building_area_sqm": 178.9,
                    "primary_building_clipped_area_sqm": 177.7,
                    "primary_building_unclipped_area_sqm": 178.9,
                    "primary_building_confidence": 0.998046875,
                    "aoi_id": "2_1",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 2,
                    "building_total_area_sqm": 689.1,
                    "building_total_clipped_area_sqm": 449.5,
                    "building_total_unclipped_area_sqm": 689.1,
                    "building_confidence": 0.9999809265136719,
                    "primary_building_area_sqm": 508.6,
                    "primary_building_clipped_area_sqm": 329.9,
                    "primary_building_unclipped_area_sqm": 508.6,
                    "primary_building_confidence": 0.998046875,
                    "aoi_id": "2_2",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 1,
                    "building_total_area_sqm": 3717.3,
                    "building_total_clipped_area_sqm": 637.0,
                    "building_total_unclipped_area_sqm": 3717.3,
                    "building_confidence": 0.998046875,
                    "primary_building_area_sqm": 3717.3,
                    "primary_building_clipped_area_sqm": 637.0,
                    "primary_building_unclipped_area_sqm": 3717.3,
                    "primary_building_confidence": 0.998046875,
                    "aoi_id": "2_3",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 3,
                    "building_total_area_sqm": 1589.5,
                    "building_total_clipped_area_sqm": 636.3,
                    "building_total_unclipped_area_sqm": 1589.5,
                    "building_confidence": 0.999999962747097,
                    "primary_building_area_sqm": 1089.8,
                    "primary_building_clipped_area_sqm": 316.9,
                    "primary_building_unclipped_area_sqm": 1089.8,
                    "primary_building_confidence": 0.990234375,
                    "aoi_id": "3_0",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 4,
                    "building_total_area_sqm": 1139.0,
                    "building_total_clipped_area_sqm": 626.4,
                    "building_total_unclipped_area_sqm": 1139.0,
                    "building_confidence": 0.9999999981810106,
                    "primary_building_area_sqm": 457.5,
                    "primary_building_clipped_area_sqm": 346.3,
                    "primary_building_unclipped_area_sqm": 457.5,
                    "primary_building_confidence": 0.990234375,
                    "aoi_id": "3_1",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 5,
                    "building_total_area_sqm": 1048.7,
                    "building_total_clipped_area_sqm": 531.3,
                    "building_total_unclipped_area_sqm": 1048.7,
                    "building_confidence": 0.999999999998721,
                    "primary_building_area_sqm": 250.2,
                    "primary_building_clipped_area_sqm": 240.9,
                    "primary_building_unclipped_area_sqm": 250.2,
                    "primary_building_confidence": 0.998046875,
                    "aoi_id": "3_2",
                    "mesh_date": "2021-01-23",
                },
                {
                    "building_present": "Y",
                    "building_count": 1,
                    "building_total_area_sqm": 306.2,
                    "building_total_clipped_area_sqm": 70.3,
                    "building_total_unclipped_area_sqm": 306.2,
                    "building_confidence": 0.990234375,
                    "primary_building_area_sqm": 306.2,
                    "primary_building_clipped_area_sqm": 70.3,
                    "primary_building_unclipped_area_sqm": 306.2,
                    "primary_building_confidence": 0.990234375,
                    "aoi_id": "3_3",
                    "mesh_date": "2021-01-23",
                },
            ]
        )
        pd.testing.assert_frame_equal(df, expected)

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
        features_gdf = gpd.GeoDataFrame(
            [
                # This should be the primary
                {
                    "feature_id": 0,
                    "aoi_id": 0,
                    "confidence": 0.94,
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

        features_gdf = parcels.filter_features_in_parcels(features_gdf)

        rollup_df = parcels.parcel_rollup(parcels_gdf, features_gdf, classes_df, "us", "nearest")
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
        pd.testing.assert_frame_equal(rollup_df, expected)

    def test_filter_and_rollup_gridded(self, cache_directory: Path, parcel_gdf_au_tests: gpd.GeoDataFrame):

        packs = ["building", "vegetation"]
        country = "au"
        parcel_gdf = parcel_gdf_au_tests

        feature_api = FeatureApi(cache_dir=cache_directory)
        classes_df = feature_api.get_feature_classes(packs)

        features_gdf, metadata_df, error_df = feature_api.get_features_gdf_bulk(
            parcel_gdf, packs=packs
        )

        # No error
        assert len(error_df) == 0

        # Had a bug that didn't do AOI IDs properly
        assert features_gdf[AOI_ID_COLUMN_NAME].isna().sum() == 0
        assert metadata_df[AOI_ID_COLUMN_NAME].isna().sum() == 0

        # We get results in all AOIs
        assert len(features_gdf.aoi_id.unique()) == len(parcel_gdf)

        features_gdf_filtered = parcels.filter_features_in_parcels(features_gdf, config=None)
        assert len(features_gdf) > len(features_gdf_filtered)*0.95 # Very little should have been filtered out in these examples.
        assert len(features_gdf_filtered.aoi_id.unique()) == len(parcel_gdf)

        # Check rollup matches what's expected
        rollup_df = parcels.parcel_rollup(
            parcel_gdf, features_gdf, classes_df, country=country, primary_decision="largest_intersection"
        )
        assert len(rollup_df) == len(parcel_gdf)
        final_df = metadata_df.merge(rollup_df, on=AOI_ID_COLUMN_NAME).merge(parcel_gdf, on=AOI_ID_COLUMN_NAME)
        assert len(final_df) == len(parcel_gdf)
