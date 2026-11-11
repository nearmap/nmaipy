"""An in-process fake of the Feature API (v4 bulk) and the Roof Age API (v1), answering from a synthetic `World`.

This is the only module that turns the world into API JSON. It knows nothing about HTTP libraries: `FakeApi.handle`
takes a method, URL and body and returns a `FakeResponse`. The fake emits exactly the key paths in `FIELD_PATHS`;
`IGNORED_LIVE_FIELDS` lists the live paths it leaves out on purpose (the drift check enforces both).
"""

from collections import defaultdict
from collections.abc import Callable, Iterator
import dataclasses
from functools import cache
import json
from urllib.parse import parse_qsl, urlencode, urlsplit

import numpy as np
import pyproj
import shapely
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from nmaipy.constants import (
    API_CRS,
    AREA_CRS,
    DRIVEABLE_ID,
    GRID_SIZE_DEGREES,
    LEAF_OFF_VEG_ID,
    MAX_AOI_AREA_SQM_BEFORE_GRIDDING,
    ROOF_ID,
    SQUARED_METERS_TO_SQUARED_FEET,
    SURFACES_IDS,
    VEG_IDS,
    VEG_WOODY_COMPOSITE_ID,
)
from nmaipy.geometry_utils import split_geometry_into_grid

from .world import CLASS_DESCRIPTIONS, Block, Feature, RoofInstance, Survey, World, class_id, synthetic_id, vocabulary

# The classes the API clips to the request; the fake keeps its own list so it does not follow nmaipy.constants.
CONNECTED_CLASS_IDS = frozenset([*SURFACES_IDS, *VEG_IDS, VEG_WOODY_COMPOSITE_ID, DRIVEABLE_ID, LEAF_OFF_VEG_ID])
HOST = "api.nearmap.com"
FEATURES_ROOT = "/ai/features/v4/bulk"
ROOF_AGE_PATH = "/ai/roofage/v1/resources/latest.json"
NO_SURVEY = {"message": "No synthetic survey matches this request"}
NO_ROOF_INSTANCES = {"error": "No synthetic roof instances here"}
ERROR_KEYS = (
    ("features", 403, frozenset({"message"})),
    ("features", 404, frozenset(NO_SURVEY)),
    ("roofage", 404, frozenset(NO_ROOF_INSTANCES)),
)
ROOF_AGE_RESOURCE_ID = synthetic_id("roof_age/latest")
PREVIOUS_SYSTEM_VERSION = "gen6-golden_world-0.9"

_FEATURE = "features[]."
_BLOCK = "features[].attributes[]."
_COMPONENT = "features[].attributes[].components[]."
_INSTANCE = "features[].properties."
FIELD_PATHS: dict[str, frozenset[str]] = {
    "packs": frozenset(
        {"packs", "packs[]", "packs[].code", "packs[].postcat"}
        | {f"packs[].{kind}{suffix}" for kind in ("featureClasses", "attributeClasses") for suffix in ("", "[]")}
        | {
            f"packs[].{kind}[].{key}"
            for kind in ("featureClasses", "attributeClasses")
            for key in ("id", "description")
        }
    ),
    "classes": frozenset(
        {"classes", "classes[]", "classes[].id", "classes[].type", "classes[].description"}
        | {"classes[].availability", "classes[].availability[]"}
        | {f"classes[].availability[].{key}" for key in ("systemVersion", "perspective", "status")}
    ),
    "features": frozenset(
        {"systemVersion", "link", "surveyDate", "surveyId", "resourceId", "perspective", "postcat", "3dDate"}
        | {"features", "features[]", "message"}
        | {
            _FEATURE + key
            for key in (
                "id",
                "classId",
                "description",
                "confidence",
                "parentId",
                "geometry",
                "geometry.type",
                "geometry.coordinates",
                "areaSqm",
                "areaSqft",
                "clippedAreaSqm",
                "clippedAreaSqft",
                "unclippedAreaSqm",
                "unclippedAreaSqft",
                "surveyDate",
                "meshDate",
                "fidelity",
                "isFootprint",
                "belongsToParcel",
                "clippedGeometry",
                "clippedGeometry.type",
                "clippedGeometry.coordinates",
                "attributes",
                "attributes[]",
            )
        }
        | {
            _BLOCK + key
            for key in (
                "classId",
                "description",
                "components",
                "components[]",
                "has3dAttributes",
                "height",
                "heightM",
                "heightFt",
                "numStories",
                "numStories.1",
                "numStories.2",
                "numStories.3+",
                "pitch",
                "available",
                "value",
            )
        }
        | {
            _COMPONENT + key
            for key in ("classId", "description", "areaSqm", "areaSqft", "confidence", "ratio", "dominant")
        }
    ),
    "roofage": frozenset(
        {"type", "resourceId", "modelVersion", "presentationVersion", "features", "features[]", "error"}
        | {f"features[].{key}" for key in ("type", "id", "geometry", "geometry.type", "geometry.coordinates")}
        | {
            "features[].properties",
            f"{_INSTANCE}timeline",
            f"{_INSTANCE}timeline[]",
            f"{_INSTANCE}timeline[].captureDate",
        }
        | {
            _INSTANCE + key
            for key in (
                "afterInstallationCaptureDate",
                "area",
                "asOfDate",
                "assessorData",
                "beforeInstallationCaptureDate",
                "evidenceType",
                "evidenceTypeDescription",
                "hilbertId",
                "installationDate",
                "kind",
                "mapBrowserUrl",
                "maxCaptureDate",
                "minCaptureDate",
                "numberOfCaptures",
                "relevantPermits",
                "trustScore",
            )
        }
    ),
}
IGNORED_LIVE_FIELDS: frozenset[str] = frozenset(
    {
        "packs:packs[].featureClasses[].internalId",
        "packs:packs[].featureClasses[].supportedVersions",
        "packs:packs[].featureClasses[].supportedVersionsInternal",
        "packs:packs[].attributeClasses[].internalId",
        "packs:packs[].attributeClasses[].supportedVersions",
        "packs:packs[].attributeClasses[].supportedVersionsInternal",
        "classes:classes[].internalId",
        "classes:classes[].schema",
        "features:credits",
        "features:extentMinX",
        "features:extentMinY",
        "features:extentMaxX",
        "features:extentMaxY",
        "features:internalSystemVersion",
        "features:partialResource",
        "features:surveyTags",
        "features:taskId",
        "features:features[].internalClassId",
        "features:features[].attributes[].internalClassId",
        "features:features[].attributes[].components[].internalClassId",
        "roofage:features[].properties.timeline[].primaryResourceWebtileUid",
        "roofage:features[].properties.timeline[].sdSurveyId",
    }
)


class UnknownRequest(Exception):
    """A request the fake has no route or no answer for."""


@dataclasses.dataclass(frozen=True)
class FakeResponse:
    """One fake HTTP answer; `endpoint` names the `FIELD_PATHS` entry its body follows."""

    endpoint: str
    status: int
    content_type: str
    text: str


class FakeApi:
    """Answers Feature API and Roof Age API requests from a `World`, following the live API's rules."""

    def __init__(self, world: World, since: str | None = None, until: str | None = None) -> None:
        self.world = world
        self.since = since
        self.until = until
        self._aois = world.aoi_geometries
        for region in world.errors:
            assert (region.endpoint, region.status, frozenset(region.body)) in ERROR_KEYS, region

    def handle(self, method: str, url: str, body: bytes | str | None) -> FakeResponse:
        """Answer one request; raise `UnknownRequest` for anything the fake does not route."""
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        route = (method, parts.netloc, parts.path)
        if route == ("GET", HOST, f"{FEATURES_ROOT}/packs.json"):
            return _ok("packs", self._packs())
        if route == ("GET", HOST, f"{FEATURES_ROOT}/classes.json"):
            return _ok("classes", self._classes())
        if route == ("POST", HOST, f"{FEATURES_ROOT}/features.json"):
            return self._features(query, json.loads(body))
        if route == ("POST", HOST, ROOF_AGE_PATH):
            return self._roof_age(json.loads(body))
        raise UnknownRequest(f"{method} {parts.netloc}{parts.path}")

    def _packs(self) -> dict:
        packs = vocabulary()["packs"]
        return {
            "packs": [
                {
                    "code": code,
                    "postcat": False,
                    "featureClasses": [_catalogue_class(c) for c in packs[code]["featureClasses"]],
                    "attributeClasses": [_catalogue_class(c) for c in packs[code]["attributeClasses"]],
                }
                for code in sorted(packs)
            ]
        }

    def _classes(self) -> dict:
        kinds = {}
        for pack in vocabulary()["packs"].values():
            kinds |= {c: "Attribute" for c in pack["attributeClasses"]} | {c: "Feature" for c in pack["featureClasses"]}
        current = {s.system_version for s in self.world.surveys}
        return {
            "classes": [
                {
                    **_catalogue_class(cid),
                    "type": kind,
                    "availability": [
                        {
                            "systemVersion": v,
                            "perspective": "Vert",
                            "status": "alpha" if cid in self.world.alpha_only and v in current else "prod",
                        }
                        for v in sorted(current | {PREVIOUS_SYSTEM_VERSION})
                    ],
                }
                for cid, kind in sorted(kinds.items())
            ]
        }

    def _features(self, query: dict[str, str], body: dict) -> FakeResponse:
        request = shape(body["aoi"])
        if error := self._error("features", request):
            return error
        surveys = [
            s
            for s in self.world.surveys
            if s.region.intersects(request) and query.get("since", "") <= s.date <= (query.get("until") or "9999")
        ]
        if not surveys:
            return _response("features", 404, NO_SURVEY)
        survey = max(surveys, key=lambda s: s.date)
        packs = vocabulary()["packs"]
        requested = query["packs"].split(",")
        feature_classes = {c for p in requested for c in packs[p]["featureClasses"]}
        attribute_classes = {c for p in requested for c in packs[p]["attributeClasses"]}
        parcel_aoi = self._parcel_aoi(request) if query["parcelMode"] == "true" else None

        candidates = [
            f for f in self.world.features if f.class_id in feature_classes and f.geometry.within(survey.region)
        ]
        selected = {f.name: f for f in candidates if _intersection_area(f, request, self.world.country) > 0}
        roofs = {name for name, f in selected.items() if f.class_id == ROOF_ID}
        selected |= {f.name: f for f in candidates if f.parent in roofs}
        flags = {name: self._flag(f, parcel_aoi, selected) for name, f in selected.items()} if parcel_aoi else {}
        records = [
            self._feature(f, request, survey, attribute_classes, flags.get(f.name))
            for f in sorted(selected.values(), key=lambda f: f.id)
        ]
        centroid = request.centroid
        return _ok(
            "features",
            {
                "systemVersion": survey.system_version,
                "link": f"https://apps.nearmap.com/maps/#/@{centroid.y:.7f},{centroid.x:.7f},20.00z,0d/V/"
                f"{survey.date.replace('-', '')}?locationMarker",
                "surveyDate": survey.date,
                "surveyId": survey.survey_id,
                "resourceId": survey.resource_id,
                "perspective": "Vert",
                "postcat": False,
                "3dDate": survey.mesh_date,
                "features": [r for r in records if r is not None],
            },
        )

    def _parcel_aoi(self, request: BaseGeometry) -> str:
        matches = [aoi_id for aoi_id, geometry in self._aois.items() if geometry.equals_exact(request, 1e-9)]
        if len(matches) != 1:
            raise UnknownRequest(f"parcelMode request for a polygon that is not one AOI of the world: {request.wkt}")
        return matches[0]

    def _flag(self, feature: Feature, aoi_id: str, selected: dict[str, Feature]) -> str:
        """The API's per-parcel judgement: "belongs", "outside" or "multiparcel" (children follow their roof)."""
        if feature.class_id in CONNECTED_CLASS_IDS:
            return "belongs"
        parent = selected.get(feature.parent)
        if parent is None or parent.class_id != ROOF_ID:
            return self.world.parcel_flags.get((aoi_id, feature.name), "belongs")
        parent_flag = self._flag(parent, aoi_id, selected)
        if parent_flag != "multiparcel":
            return parent_flag
        touches = _intersection_area(feature, self._aois[aoi_id], self.world.country) > 0
        return "multiparcel" if touches else "outside"

    def _feature(
        self, feature: Feature, request: BaseGeometry, survey: Survey, attribute_classes: set[str], flag: str | None
    ) -> dict | None:
        country = self.world.country
        whole = _area(feature.geometry, country)
        clip = _polygonal(feature.geometry.intersection(request))
        clipped = _area(clip, country)
        if feature.class_id in CONNECTED_CLASS_IDS:
            geometry, area = clip, clipped
        else:
            geometry, area = feature.geometry, whole
        record = {
            "id": feature.id,
            "classId": feature.class_id,
            "description": CLASS_DESCRIPTIONS[feature.class_id],
            "confidence": feature.confidence,
            "parentId": feature.parent_id,
            "geometry": mapping(geometry),
            **_areas("area", area),
            **_areas("clippedArea", clipped),
            **_areas("unclippedArea", whole),
            "attributes": [
                _block(block, whole) for block in feature.attributes if class_id(block.description) in attribute_classes
            ],
            "surveyDate": survey.date,
            "meshDate": survey.mesh_date,
        }
        if feature.fidelity is not None:
            record["fidelity"] = feature.fidelity
        if feature.is_footprint is not None:
            record["isFootprint"] = feature.is_footprint
        if flag is not None:
            record["belongsToParcel"] = flag != "outside"
            if flag == "multiparcel":
                record["clippedGeometry"] = mapping(clip)
        return record

    def _roof_age(self, body: dict) -> FakeResponse:
        if (cutoffs := (body.get("sinceAsOfDate"), body.get("untilAsOfDate"))) != (self.since, self.until):
            raise UnknownRequest(f"roof age cutoffs (since, until) {cutoffs} != {(self.since, self.until)}")
        request = shape(body["aoi"])
        if error := self._error("roofage", request):
            return error
        instances = sorted(
            (i for i in self.world.roof_instances if i.geometry.intersects(request)), key=lambda i: i.hilbert_id
        )
        if not instances:
            return _response("roofage", 404, NO_ROOF_INSTANCES)
        return _ok(
            "roofage",
            {
                "type": "FeatureCollection",
                "resourceId": ROOF_AGE_RESOURCE_ID,
                "modelVersion": "A.1",
                "presentationVersion": "1",
                "features": [self._roof_instance(i) for i in instances],
            },
        )

    def _roof_instance(self, instance: RoofInstance) -> dict:
        captures = sorted(instance.capture_dates)
        properties = {
            "kind": instance.kind,
            "installationDate": instance.installation_date,
            "trustScore": instance.trust_score,
            "evidenceType": instance.evidence_type,
            "evidenceTypeDescription": instance.evidence_type_description,
            "area": round(_area(instance.geometry, self.world.country), 2),
            "asOfDate": self.world.roof_age_as_of_date,
            "assessorData": False,
            "relevantPermits": False,
            "hilbertId": instance.hilbert_id,
            "minCaptureDate": captures[0],
            "maxCaptureDate": captures[-1],
            "numberOfCaptures": len(captures),
            "timeline": [{"captureDate": d} for d in captures],
        }
        if instance.installation_date is not None:
            properties["afterInstallationCaptureDate"] = min(d for d in captures if d >= instance.installation_date)
            if before := [d for d in captures if d < instance.installation_date]:
                properties["beforeInstallationCaptureDate"] = max(before)
        if instance.map_browser_url:
            centroid = instance.geometry.centroid
            date = captures[-1].replace("-", "")
            properties["mapBrowserUrl"] = (
                f"https://apps.nearmap.com/maps/#/@{centroid.y:.6f},{centroid.x:.6f},20z,0d/V/{date}"
            )
        return {
            "type": "Feature",
            "id": int(instance.hilbert_id),
            "geometry": mapping(instance.geometry),
            "properties": properties,
        }

    def _error(self, endpoint: str, request: BaseGeometry) -> FakeResponse | None:
        for region in self.world.errors:
            if region.endpoint == endpoint and region.region.intersects(request):
                return _response(endpoint, region.status, region.body)
        return None


def key_paths(body: object, prefix: str = "") -> dict[str, set[str]]:
    """Every key path in a JSON body (list items as `[]`, geometry coordinates as one leaf) with its JSON types."""
    paths = defaultdict(set)
    for path, value in _walk(body, prefix):
        paths[path].add(_json_type(value))
    return dict(paths)


def case_answers(fake: FakeApi, case: str, kwargs: dict) -> Iterator[tuple[str, BaseGeometry | None, int, dict, bool]]:
    """The fake's answers to the requests a case's export makes: (endpoint, request polygon, status, body, grid)."""
    root = f"https://{HOST}{FEATURES_ROOT}"
    for endpoint in ("packs", "classes"):
        yield _answer(fake.handle("GET", f"{root}/{endpoint}.json?apikey=", None), None)
    dates = {k: kwargs[k] for k in ("since", "until") if kwargs.get(k)}
    for aoi in fake.world.aoi_geometries.values():
        grid = _area(aoi, fake.world.country) > MAX_AOI_AREA_SQM_BEFORE_GRIDDING
        requests = split_geometry_into_grid(aoi, GRID_SIZE_DEGREES).geometry if grid else [aoi]
        for request in requests:
            query = urlencode({"parcelMode": str(not grid).lower(), **dates, "packs": ",".join(kwargs["packs"])})
            body = json.dumps({"aoi": mapping(request)})
            yield _answer(fake.handle("POST", f"{root}/features.json?apikey=&{query}", body), request, grid)
        if kwargs.get("roof_age"):
            cutoffs = {f"{k}AsOfDate": v for k, v in dates.items()}
            body = json.dumps({"aoi": mapping(aoi), **cutoffs})
            yield _answer(fake.handle("POST", f"https://{HOST}{ROOF_AGE_PATH}?bulk=true", body), aoi)


def _answer(response: FakeResponse, request: BaseGeometry | None, grid: bool = False) -> tuple:
    return response.endpoint, request, response.status, json.loads(response.text), grid


def _walk(value: object, path: str) -> Iterator[tuple[str, object]]:
    if path:
        yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            if key != "coordinates" or not isinstance(item, list):
                yield from _walk(item, f"{path}.{key}" if path else key)
            else:
                yield f"{path}.{key}", item
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item, f"{path}[]")


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    return {dict: "object", list: "array", str: "string"}.get(type(value), "number")


def _ok(endpoint: str, body: dict) -> FakeResponse:
    return _response(endpoint, 200, body)


def _response(endpoint: str, status: int, body: dict) -> FakeResponse:
    return FakeResponse(endpoint, status, "application/json", json.dumps(body, separators=(",", ":")))


def _catalogue_class(cid: str) -> dict:
    assert cid in CLASS_DESCRIPTIONS, f"pack class {cid} is not in nmaipy/class_descriptions.json"
    return {"id": cid, "description": CLASS_DESCRIPTIONS[cid]}


def _block(block: Block, whole_area: float) -> dict:
    record = {"classId": class_id(block.description), "description": block.description, **dict(block.fields)}
    if block.components is not None:
        record["components"] = []
        for c in block.components:
            component = {
                "classId": class_id(c.description),
                "description": c.description,
                **_areas("area", c.ratio * whole_area),
                "confidence": c.confidence,
                "ratio": c.ratio,
            }
            if c.dominant is not None:
                component["dominant"] = c.dominant
            record["components"].append(component)
    return record


def _areas(prefix: str, sqm: float) -> dict:
    return {f"{prefix}Sqm": round(sqm, 1), f"{prefix}Sqft": round(sqm * SQUARED_METERS_TO_SQUARED_FEET)}


def _intersection_area(feature: Feature, request: BaseGeometry, country: str) -> float:
    return _area(_polygonal(feature.geometry.intersection(request)), country)


def _polygonal(geometry: BaseGeometry) -> BaseGeometry:
    parts = [g for g in shapely.get_parts(geometry) if g.geom_type == "Polygon" and not g.is_empty]
    return shapely.union_all(parts) if parts else shapely.Polygon()


def _area(geometry: BaseGeometry, country: str) -> float:
    return float(shapely.area(shapely.transform(geometry, _projector(country))))


@cache
def _projector(country: str) -> Callable[[np.ndarray], np.ndarray]:
    transformer = pyproj.Transformer.from_crs(API_CRS, AREA_CRS[country], always_xy=True)
    return lambda coords: np.column_stack(transformer.transform(coords[:, 0], coords[:, 1]))
