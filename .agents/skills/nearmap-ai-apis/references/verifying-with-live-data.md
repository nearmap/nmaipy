# Verifying a claim against the live API

Use this when the references don't settle a question, or when someone needs proof on their own data. It requires a Nearmap API key with the relevant entitlements, in the `API_KEY` environment variable. nmaipy also loads a `.env` file, but which one depends on how it is run (from a repo checkout it finds the repo's), so set the variable. Never paste or echo the key.

## Principles

- **Replay the user's own request when you have it.** Run their `curl` (or URL and body) with your own API key in place of theirs, changing nothing else: the endpoint, parameters (`parcelMode`, packs, `since`/`until`, system version) and AOI are what produced the behaviour. Change one parameter at a time from there. If your response differs from theirs, check first that the same classes and includes came back: your key's permissions may differ.

- **Look at raw payloads, not exports.** `FeatureApi.get_features(...)` returns the response dict unchanged, so you see what the API sent before nmaipy filters or renames anything.
- **Compare with one variable changed.** Query the same AOI with `parcel_mode=True` and `parcel_mode=False` (two `FeatureApi` instances), or query two AOIs that share a feature. Check that both responses have the same `surveyId` before comparing them.
- **Pin the survey** with `since=until=<surveyDate>`. Results then don't shift when a newer survey lands. Resource ids are not stable across AI re-runs, so don't use them for pinning.
- **Make a shared-feature test by splitting a parcel.** Cut it through a roof's centroid into two halves. The roof becomes multiparcel in each half, and its children land wholly on one side or the other.
- **Measure areas in an equal-area CRS:** `nmaipy.constants.AREA_CRS[country]`, e.g. `esri:102003` for the US.
- **`FeatureApi.payload_gdf` mutates the payload it is given.** It swaps `clippedGeometry` into `geometry`. Pass it a `copy.deepcopy(payload)` if you still need the raw geometry afterwards.
- **Use a cache directory** (`FeatureApi(cache_dir=...)`), so repeated analysis doesn't re-query.
- **Match the exporter's system version.** `FeatureApi` defaults to `system_version_prefix="gen6-"`, while the exporter sends no prefix. Pass `system_version_prefix=None` to reproduce an export's rows.

## Skeleton

```python
import copy
import os

from shapely.geometry import box, shape

from nmaipy.constants import ROOF_ID
from nmaipy.feature_api import FeatureApi
from nmaipy.feature_attributes import flatten_roof_attributes

api_on = FeatureApi(parcel_mode=True, cache_dir="cache_pm")   # API key from the API_KEY env var
api_off = FeatureApi(parcel_mode=False, cache_dir="cache_nopm")
kwargs = dict(packs=["building", "roof_char"], include=["hurricaneScore"], since="2026-10-08", until="2026-10-08")

parcel = ...  # shapely Polygon in EPSG:4326
on, off = api_on.get_features(parcel, "us", **kwargs), api_off.get_features(parcel, "us", **kwargs)
assert on["surveyId"] == off["surveyId"]

# Split through the largest roof to create two parcels that share it
roof = max((f for f in on["features"] if f["classId"] == ROOF_ID), key=lambda f: f["unclippedAreaSqm"])
cx = shape(roof["geometry"]).centroid.x
minx, miny, maxx, maxy = parcel.bounds
west = parcel.intersection(box(minx - 1, miny - 1, cx, maxy + 1))
east = parcel.intersection(box(cx, miny - 1, maxx + 1, maxy + 1))
halves = {side: api_on.get_features(g, "us", **kwargs) for side, g in (("W", west), ("E", east))}

# nmaipy's view of one half
gdf, _ = FeatureApi.payload_gdf(copy.deepcopy(halves["W"]), aoi_id="W", parcel_mode=True)
gdf = gdf.reset_index()
row = gdf[gdf.feature_id == roof["id"]].iloc[0].to_dict()
flat = flatten_roof_attributes([row], country="us", child_features=gdf[gdf.class_id != ROOF_ID])
```

Typical comparisons:
- the roof's `attributes` in W vs E;
- the roof's includes in W vs E;
- `flat["tree_overhang_ratio"]` vs `hurricaneScore.modelInputFeatures.treeOverhangRatio`;
- for each child, whether it intersects its half, and its `belongsToParcel` and `clippedAreaSqm`.

nmaipy's own `tests/test_parcel_mode_semantics.py` is a complete worked version of this recipe.

## Showing it

Spatial behaviour is easier to see than to read. Draw one example: the AOI, the feature's whole outline and its in-AOI part, with the API's area fields in the title.

```python
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape

from nmaipy.constants import API_CRS, AREA_CRS


def show(aoi, feature, country="us", path="example.png"):
    """Draw a query AOI and one raw API feature, in metres, and save it as an image."""
    outline = shape(feature["geometry"])
    layers = gpd.GeoSeries([aoi, outline, outline.intersection(aoi)], crs=API_CRS).to_crs(AREA_CRS[country])
    ax = layers.iloc[[0]].boundary.plot(color="black", linewidth=1)
    layers.iloc[[1]].plot(ax=ax, color="tab:orange", alpha=0.35)
    layers.iloc[[2]].plot(ax=ax, color="tab:blue", alpha=0.6)
    ax.set_title(
        f"{feature['description']}: areaSqm {feature['areaSqm']}, clippedAreaSqm {feature['clippedAreaSqm']}, "
        f"unclippedAreaSqm {feature['unclippedAreaSqm']}",
        fontsize=9,
    )
    ax.set_axis_off()
    plt.savefig(path, dpi=150, bbox_inches="tight")
```

Black is the AOI, orange the feature's whole outline, blue its part inside the AOI. To give the user something interactive instead, write the same layers to GeoJSON (`layers.to_crs(API_CRS).to_file("example.geojson")`) for geojson.io or QGIS.

Save these files in the user's working folder (in an nmaipy checkout, the gitignored `data/`), not a temporary system directory. Then make sure the user sees the picture: open it for them or show it where your environment displays images, and say in one line what it shows. A file path alone often shows nothing, and the asker may not know how to open it.

## Reporting what you found

State the sample: how many features or parcels, which packs, and which survey date. Say what was compared. Separate what you observed from what you infer. A single parcel is an example, not proof of a general rule. Patterns only become convincing across tens of parcels.
