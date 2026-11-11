"""Live drift check: does the real API still answer the way the golden fake does?

Usage: python -m tests.golden.drift [--probe] [--write-vocabulary]

Needs API_KEY, and never prints it. Exits 1 when the live responses differ from the fake's field allowlist
(`FIELD_PATHS`, `IGNORED_LIVE_FIELDS`), from `vocabulary.json`, from the fake's enumerated values (`ENUM_PATHS`) and
class descriptions, or from the rules the fake follows. `--probe` prints the error bodies' key structure and the live
survey dates; `--write-vocabulary` rewrites `vocabulary.json`.
"""

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
import dataclasses
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry

from nmaipy.api_common import APIError
from nmaipy.constants import AREA_CRS, GRID_SIZE_DEGREES, SQUARED_METERS_TO_SQUARED_FEET
from nmaipy.feature_api import FeatureApi
from nmaipy.geometry_utils import split_geometry_into_grid
from nmaipy.roof_age_api import RoofAgeApi

from . import fake_api, world
from .aois import LARGE_ADELAIDE_AOI_WKT
from .fake_api import CONNECTED_CLASS_IDS, FIELD_PATHS, IGNORED_LIVE_FIELDS, FakeApi, key_paths
from .replay import CASES

DATA_DIR = Path(__file__).parents[1] / "data"
OCEAN_SQUARE = Polygon([(-72.5, 39.5), (-72.4996, 39.5), (-72.4996, 39.5004), (-72.5, 39.5004), (-72.5, 39.5)])
BEFORE_ANY_SURVEY = "2001-01-01"
AREA_TOLERANCE_SQM = 0.5
AREA_TOLERANCE_FRACTION = 0.01
RATIO_TOLERANCE = 0.005
SQFT_TOLERANCE = 0.5 + 0.05 * SQUARED_METERS_TO_SQUARED_FEET
ENUM_PATHS: dict[str, frozenset[str]] = {
    "classes": frozenset({"classes[].type", "classes[].availability[].status", "classes[].availability[].perspective"}),
    "features": frozenset({"perspective"}),
    "roofage": frozenset(
        {
            "features[].type",
            "features[].properties.kind",
            "features[].properties.evidenceType",
            "features[].properties.evidenceTypeDescription",
        }
    ),
}
assert all(paths <= FIELD_PATHS[endpoint] for endpoint, paths in ENUM_PATHS.items()), "ENUM_PATHS not in FIELD_PATHS"
LINK = re.compile(r"^https://apps\.nearmap\.com/maps/#/@(-?[\d.]+),(-?[\d.]+),20\.00z,0d/V/(\d{8})\?locationMarker$")


@dataclasses.dataclass(frozen=True)
class Exchange:
    """One API answer: the endpoint, the case and request polygon that produced it, its status and parsed body."""

    endpoint: str
    case: str | None
    request: BaseGeometry | None
    status: int
    body: dict
    grid_cell: bool = False


def main() -> None:
    """Run the live drift check (or the probe, or the vocabulary refresh) and exit 1 on any difference."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="store_true", help="print error-body keys and live survey dates")
    parser.add_argument("--write-vocabulary", action="store_true", help=f"rewrite {world.VOCABULARY_PATH.name}")
    args = parser.parse_args()
    api_key = os.environ.get("API_KEY", "")
    assert api_key, "the drift check needs API_KEY"
    try:
        _run(args, api_key)
    except Exception:
        raise SystemExit(traceback.format_exc().replace(api_key, "APIKEYREMOVED")) from None


def collect_live(api_key: str, cache_dir: Path) -> list[Exchange]:
    """Make the drift check's live requests: the catalogue, each case on public sample AOIs, and the error bodies."""
    features = FeatureApi(api_key=api_key, cache_dir=cache_dir / "features", system_version_prefix=None, maxretry=3)
    exchanges = [
        Exchange("packs", None, None, 200, features._get_feature_api_results_as_data(features.PACKS_URL)[1]),
        Exchange("classes", None, None, 200, features._get_feature_api_results_as_data(features.CLASSES_URL)[1]),
    ]
    for case, aois in sample_aois().items():
        kwargs = CASES[case]
        query = dict(packs=kwargs["packs"], since=kwargs.get("since"), until=kwargs.get("until"))
        for aoi in aois:
            if features.should_grid_aoi(aoi, kwargs["country"]):
                exchanges += _grid_cells(features, case, aoi, query)
            else:
                body = features.get_features(aoi, kwargs["country"], **query)
                exchanges.append(Exchange("features", case, aoi, 200, body))
        if kwargs.get("roof_age"):
            exchanges += _roof_age(api_key, cache_dir / "roof_age", case, aois)

    us = CASES["us"]
    exchanges.append(_error(features.get_features, "features", "us", OCEAN_SQUARE, "us", packs=us["packs"]))
    first = sample_aois()["us"][0]
    no_survey = dict(packs=us["packs"], since=BEFORE_ANY_SURVEY, until=BEFORE_ANY_SURVEY)
    exchanges.append(_error(features.get_features, "features", "us", first, "us", **no_survey))
    roof_age = RoofAgeApi(api_key=api_key, cache_dir=cache_dir / "roof_age_errors", until_as_of_date=us["until"])
    exchanges.append(_error(roof_age.get_roof_age_by_aoi, "roofage", "us", OCEAN_SQUARE, "ocean"))
    return exchanges


def sample_aois() -> dict[str, list[BaseGeometry]]:
    """The public sample AOIs already in the repository that each case's live requests use."""
    us = pd.read_csv(DATA_DIR / "test_parcels_2.csv").head(10)
    au = pd.read_csv(DATA_DIR / "test_parcels.csv")
    au_geoms = [shapely.from_wkt(w) for w in au["geometry"].dropna()]
    au_geoms = [g for g in au_geoms if g.geom_type == "Polygon" and not g.is_empty and g.is_valid]
    return {
        "us": [shapely.from_wkt(w) for w in us["geometry"]],
        "au": au_geoms + [shapely.from_wkt(LARGE_ADELAIDE_AOI_WKT)],
    }


def fake_exchanges() -> list[Exchange]:
    """The fake's answers to the requests each case's golden run makes, without running the exporter."""
    exchanges = []
    for case in CASES:
        fake = FakeApi(world.build(case), CASES[case].get("since"), CASES[case].get("until"))
        exchanges += [Exchange(e, case, r, s, b, g) for e, r, s, b, g in fake_api.case_answers(fake, case, CASES[case])]
    return exchanges


def probe(live: list[Exchange]) -> list[str]:
    """Error bodies' key structure and each case's live survey dates, for authoring the fake (no other values)."""
    lines = []
    for e in live:
        if e.status >= 400:
            lines.append(f"{e.endpoint} {e.status}: {sorted(key_paths(e.body))}")
    dates = defaultdict(set)
    for e in live:
        if e.endpoint == "features" and e.status == 200:
            dates[(e.case, "grid cells" if e.grid_cell else "parcels")].add(e.body["surveyDate"])
        if e.endpoint == "roofage" and e.status == 200:
            dates[(e.case, "roof age asOfDate")] |= {f["properties"].get("asOfDate") for f in e.body["features"]}
    lines += [f"{case} {kind}: {sorted(d)}" for (case, kind), d in sorted(dates.items())]
    return lines


def vocabulary(live: list[Exchange]) -> dict:
    """Names only: pack codes and class ids, and per case and feature class its attribute blocks and the union of their
    components over all features."""
    packs = {
        p["code"]: {
            "featureClasses": sorted(c["id"] for c in p["featureClasses"]),
            "attributeClasses": sorted(c["id"] for c in p.get("attributeClasses", [])),
        }
        for e in live
        if e.endpoint == "packs"
        for p in e.body["packs"]
        if p["code"] in {code for case in CASES.values() for code in case["packs"]}
    }
    attributes = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for e in live:
        if e.endpoint != "features" or e.status != 200:
            continue
        for feature in e.body["features"]:
            for block in feature["attributes"]:
                components = {c["description"] for c in block.get("components", [])}
                attributes[e.case][feature["classId"]][block["description"]] |= components
    return {
        "packs": packs,
        "attributes": {
            case: {cid: {b: sorted(c) for b, c in blocks.items()} for cid, blocks in by_class.items()}
            for case, by_class in attributes.items()
        },
    }


def check_fields(live: list[Exchange], fake: list[Exchange]) -> list[str]:
    """Live key paths must equal the allowlist plus the ignored list, and the fake's JSON types must occur live."""
    differences = []
    live_types, fake_types = _types_by_path(live), _types_by_path(fake)
    for endpoint, allowed in FIELD_PATHS.items():
        seen = set(live_types[endpoint])
        ignored = {p.split(":", 1)[1] for p in IGNORED_LIVE_FIELDS if p.startswith(f"{endpoint}:")}
        unknown = {p for p in seen - allowed if not any(_under(p, i) for i in ignored)}
        differences += [f"{endpoint}: live field {p} not in FIELD_PATHS" for p in sorted(unknown)]
        differences += [f"{endpoint}: IGNORED_LIVE_FIELDS entry {i} not seen live" for i in sorted(ignored - seen)]
        differences += [f"{endpoint}: FIELD_PATHS field {p} not seen live" for p in sorted(allowed - seen)]
        for path in sorted(allowed & seen):
            if extra := fake_types[endpoint].get(path, set()) - live_types[endpoint][path]:
                differences.append(f"{endpoint}: {path} has fake types {sorted(extra)} never seen live")
    return differences


def check_vocabulary(live: list[Exchange]) -> list[str]:
    """The live vocabulary must equal `vocabulary.json`."""
    expected = json.loads(world.VOCABULARY_PATH.read_text())
    actual = json.loads(json.dumps(vocabulary(live)))
    return [f"vocabulary: {d}" for d in _dict_differences(expected, actual)]


def check_values(live: list[Exchange], fake: list[Exchange]) -> list[str]:
    """Every value the fake emits at an `ENUM_PATHS` path must occur live, and live class descriptions must equal
    `nmaipy/class_descriptions.json` for every class the fake serves."""
    differences = []
    live_values, fake_values = _values_by_path(live), _values_by_path(fake)
    for endpoint, paths in sorted(ENUM_PATHS.items()):
        for path in sorted(paths):
            unseen = fake_values[endpoint][path] - live_values[endpoint][path]
            differences += [f"{endpoint}: {path} fake value {v!r} never seen live" for v in sorted(unseen, key=repr)]
    served = {cid for e in fake for cid, _ in _class_descriptions(e)}
    renamed = {
        (e.endpoint, cid, description)
        for e in live
        for cid, description in _class_descriptions(e)
        if cid in served and description != world.CLASS_DESCRIPTIONS[cid]
    }
    differences += [
        f"{endpoint}: class {cid} is {description!r} live, {world.CLASS_DESCRIPTIONS[cid]!r} in the fake"
        for endpoint, cid, description in sorted(renamed)
    ]
    return differences


def check_rules(live: list[Exchange]) -> list[str]:
    """The rules `fake_api` follows must hold on live responses."""
    differences = []
    for e in live:
        if e.endpoint == "features" and e.status == 200:
            differences += [f"features {e.case}: {d}" for d in _feature_rules(e)]
        if e.endpoint == "roofage" and e.status == 200:
            differences += [f"roofage {e.case}: {d}" for d in _roof_age_rules(e)]
    fake_errors = {(endpoint, status): keys for endpoint, status, keys in fake_api.ERROR_KEYS}
    for e in live:
        if e.status >= 400 and fake_errors.get((e.endpoint, e.status)) != frozenset(e.body):
            differences.append(f"{e.endpoint} {e.status}: error body keys {sorted(e.body)} differ from the fake's")
    return differences


def value_ranges(live: list[Exchange], fake: list[Exchange]) -> list[str]:
    """A side-by-side of value ranges (informational, never a failure)."""
    lines = ["value ranges (min / median / max)", f"{'':28}{'fake':>36}{'live':>36}"]
    for name, pick in _RANGES.items():
        cells = []
        for exchanges in (fake, live):
            values = np.array([v for e in exchanges for v in pick(e) if v is not None], dtype=float)
            cells.append(f"{values.min():.3g} / {np.median(values):.3g} / {values.max():.3g}" if len(values) else "-")
        lines.append(f"{name:28}{cells[0]:>36}{cells[1]:>36}")
    return lines


def _run(args: argparse.Namespace, api_key: str) -> None:
    with tempfile.TemporaryDirectory() as cache_dir:
        live = collect_live(api_key, Path(cache_dir))
    if args.probe:
        print("\n".join(probe(live)))
        return
    if args.write_vocabulary:
        world.VOCABULARY_PATH.write_text(json.dumps(vocabulary(live), indent=1, sort_keys=True) + "\n")
        print(f"wrote {world.VOCABULARY_PATH}")
        return
    fake = fake_exchanges()
    differences = check_fields(live, fake) + check_vocabulary(live) + check_values(live, fake) + check_rules(live)
    print("\n".join(value_ranges(live, fake)))
    print("\n".join(differences) if differences else "No drift")
    sys.exit(1 if differences else 0)


def _grid_cells(features: FeatureApi, case: str, aoi: BaseGeometry, query: dict) -> list[Exchange]:
    region = CASES[case]["country"]
    gdf, metadata, error, _ = features.get_features_gdf(aoi, region, aoi_id="drift", **query)
    assert error is None, f"the live grid request failed: {error}"
    exchanges = []
    for cell in split_geometry_into_grid(aoi, GRID_SIZE_DEGREES).geometry:
        url, body, _ = features._create_post_request(features.FEATURES_URL, cell, disable_parcel_mode=True, **query)
        path = Path(features._post_request_cache_path(url, body))
        assert path.exists(), f"no live response cached for the grid cell {cell.wkt}: the grid differs"
        exchanges.append(Exchange("features", case, cell, 200, json.loads(path.read_text()), grid_cell=True))
    return exchanges


def _roof_age(api_key: str, cache_dir: Path, case: str, aois: list[BaseGeometry]) -> list[Exchange]:
    until = CASES[case].get("until")
    api = RoofAgeApi(api_key=api_key, cache_dir=cache_dir, until_as_of_date=until)
    exchanges = []
    for i, aoi in enumerate(aois):
        api.get_roof_age_by_aoi(aoi, str(i))
        path = Path(api._get_cache_path(api._build_cache_key(aoi=aoi), until_as_of_date=until))
        exchanges.append(Exchange("roofage", case, aoi, 200, json.loads(path.read_text())))
    return exchanges


def _error(
    call: Callable[..., object], endpoint: str, case: str, aoi: BaseGeometry, *args: Any, **kwargs: Any
) -> Exchange:
    try:
        call(aoi, *args, **kwargs)
    except APIError as error:
        return Exchange(endpoint, case, aoi, error.status_code, json.loads(error.text))
    raise AssertionError(f"expected an error from {endpoint} for {aoi.wkt}")


def _types_by_path(exchanges: Iterable[Exchange]) -> dict[str, dict[str, set[str]]]:
    types = defaultdict(lambda: defaultdict(set))
    for e in exchanges:
        for path, json_type in key_paths(e.body).items():
            types[e.endpoint][path] |= json_type
    return types


def _values_by_path(exchanges: Iterable[Exchange]) -> dict[str, dict[str, set[object]]]:
    values = defaultdict(lambda: defaultdict(set))
    for e in exchanges:
        for path, value in fake_api._walk(e.body, ""):
            if path in ENUM_PATHS.get(e.endpoint, ()):
                values[e.endpoint][path].add(value)
    return values


def _class_descriptions(e: Exchange) -> Iterator[tuple[str, str]]:
    if e.endpoint == "packs":
        for pack in e.body["packs"]:
            for c in pack["featureClasses"] + pack.get("attributeClasses", []):
                yield c["id"], c["description"]
    elif e.endpoint == "classes":
        yield from ((c["id"], c["description"]) for c in e.body["classes"])
    elif e.endpoint == "features" and e.status == 200:
        yield from ((f["classId"], f["description"]) for f in e.body["features"])


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}.") or path.startswith(f"{prefix}[]")


def _dict_differences(expected: object, actual: object, where: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = [f"{where}/{k}: missing live" for k in sorted(set(expected) - set(actual))]
        differences += [f"{where}/{k}: new live" for k in sorted(set(actual) - set(expected))]
        for k in sorted(set(expected) & set(actual)):
            differences += _dict_differences(expected[k], actual[k], f"{where}/{k}")
        return differences
    return [] if expected == actual else [f"{where}: {expected} expected, {actual} live"]


def _feature_rules(e: Exchange) -> list[str]:
    differences = []
    body, request, country = e.body, e.request, CASES[e.case]["country"]
    if [f["id"] for f in body["features"]] != sorted(f["id"] for f in body["features"]):
        differences.append("features are not sorted by id")
    if not (m := LINK.match(body["link"])) or m[3] != body["surveyDate"].replace("-", ""):
        differences.append(f"link {body['link']!r} does not match the fake's format")
    elif max(abs(float(m[1]) - request.centroid.y), abs(float(m[2]) - request.centroid.x)) > 2e-6:
        differences.append("link is not at the request centroid")
    for f in body["features"]:
        name = f"{f['description']} {f['id']}"
        geometry = shape(f["geometry"])
        for block in f["attributes"]:
            vocabulary_components = (
                world.vocabulary()["attributes"].get(e.case, {}).get(f["classId"], {}).get(block["description"])
            )
            if (
                vocabulary_components
                and sorted(c["description"] for c in block.get("components", [])) != vocabulary_components
            ):
                differences.append(
                    f"{name}: block {block['description']!r} does not list exactly its vocabulary components"
                )
        if f["classId"] in CONNECTED_CLASS_IDS:
            if f["areaSqm"] != f["clippedAreaSqm"]:
                differences.append(f"{name}: connected areaSqm != clippedAreaSqm")
            if not _close(_area(geometry.difference(request), country), 0):
                differences.append(f"{name}: connected geometry extends past the request")
        elif f["areaSqm"] != f["unclippedAreaSqm"]:
            differences.append(f"{name}: discrete areaSqm != unclippedAreaSqm")
        if not _close(f["clippedAreaSqm"], _area(geometry.intersection(request), country)):
            differences.append(f"{name}: clippedAreaSqm != area(geometry ∩ request)")
        for prefix in ("area", "clippedArea", "unclippedArea"):
            sqm, sqft = f[f"{prefix}Sqm"], f[f"{prefix}Sqft"]
            if round(sqm, 1) != sqm or abs(sqft - sqm * SQUARED_METERS_TO_SQUARED_FEET) > SQFT_TOLERANCE + 1e-3 * sqft:
                differences.append(f"{name}: {prefix} sqm/sqft rounding differs")
        if (clipped := f.get("clippedGeometry")) is not None:
            expected = geometry.intersection(request)
            if not _close(_area(shape(clipped).symmetric_difference(expected), country), 0, _area(expected, country)):
                differences.append(f"{name}: clippedGeometry != geometry ∩ request")
        if e.grid_cell and ("belongsToParcel" in f or "clippedGeometry" in f):
            differences.append(f"{name}: grid cell feature has parcelMode fields")
        for block in f["attributes"]:
            for c in block.get("components", []):
                share = c["areaSqm"] / f["unclippedAreaSqm"]
                if abs(c["ratio"] - share) > RATIO_TOLERANCE + 0.1 / f["unclippedAreaSqm"]:
                    differences.append(f"{name}: {block['description']} {c['description']} ratio != area share")
    return differences


def _roof_age_rules(e: Exchange) -> list[str]:
    differences = []
    for f in e.body["features"]:
        props, geometry = f["properties"], shape(f["geometry"])
        if str(f["id"]) != str(props["hilbertId"]):
            differences.append(f"{f['id']}: id != hilbertId")
        if not geometry.intersects(e.request):
            differences.append(f"{f['id']}: instance does not intersect the AOI")
        if not _close(props["area"], _area(geometry, CASES[e.case]["country"]), fraction=0.02):
            differences.append(f"{f['id']}: area != geometry area")
    return differences


def _area(geometry: BaseGeometry, country: str) -> float:
    return float(gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(AREA_CRS[country]).area.iloc[0])


def _close(a: float, b: float, scale: float | None = None, fraction: float = AREA_TOLERANCE_FRACTION) -> bool:
    return abs(a - b) <= max(AREA_TOLERANCE_SQM, fraction * (scale if scale is not None else max(abs(a), abs(b))))


def _features(e: Exchange) -> list[dict]:
    return e.body.get("features", []) if e.endpoint == "features" and e.status == 200 else []


_RANGES = {
    "feature confidence": lambda e: [f["confidence"] for f in _features(e)],
    "feature areaSqm": lambda e: [f["areaSqm"] for f in _features(e)],
    "component ratio": lambda e: [
        c["ratio"] for f in _features(e) for b in f["attributes"] for c in b.get("components", [])
    ],
    "component confidence": lambda e: [
        c["confidence"] for f in _features(e) for b in f["attributes"] for c in b.get("components", [])
    ],
    "polygon vertices": lambda e: [shapely.get_num_coordinates(shape(f["geometry"])) for f in _features(e)],
    "features per response": lambda e: [len(_features(e))] if e.endpoint == "features" and e.status == 200 else [],
    "roof age trustScore": lambda e: [
        f["properties"]["trustScore"] for f in e.body.get("features", []) if e.endpoint == "roofage"
    ],
}


if __name__ == "__main__":
    main()
