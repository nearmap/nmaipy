"""The golden `us` parcels tiled on a lattice, and a fake API that answers for any number of the tiled copies.

Block `k` of the tiling is a copy of the template parcels `TEMPLATE_AOIS`, moved to lattice cell
`(i, j) = (k % W, k // W)`. The fake maps each request to its cell from the request's centroid, moves it back into
the template frame, answers it from one `FakeApi` over the template parcels, and moves the answer to the cell with its
feature and roof instance ids made unique to the cell. It keeps no state per AOI, so its memory does not grow with the
number of AOIs.
"""

import copy
import dataclasses
import json
import math
import threading
from urllib.parse import urlsplit
import uuid

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from . import world
from .fake_api import FEATURES_ROOT, HOST, ROOF_AGE_PATH, FakeApi, FakeResponse, UnknownRequest

TEMPLATE_AOIS = ("u1", "u2", "u3", "u4", "u5")
STEP = (0.003, 0.001)
MAX_CELLS_PER_AXIS = 10**6

_US = world.us_world()
TEMPLATE = dataclasses.replace(_US, aois={aoi_id: _US.aois[aoi_id] for aoi_id in TEMPLATE_AOIS})
_BOUNDS = shapely.total_bounds(
    [
        *TEMPLATE.aoi_geometries.values(),
        *(f.geometry for f in TEMPLATE.features),
        *(r.geometry for r in TEMPLATE.roof_instances),
    ]
)
_ORIGIN = _BOUNDS[:2]
_EXTENT = _BOUNDS[2:] - _BOUNDS[:2]
assert (_EXTENT < STEP).all(), f"the lattice step {STEP} does not clear the template extent {_EXTENT}"

_FEATURES_ROUTE = ("POST", HOST, f"{FEATURES_ROOT}/features.json")
_ROOF_AGE_ROUTE = ("POST", HOST, ROOF_AGE_PATH)


def aois(n: int) -> dict[str, str]:
    """The first `n` tiled AOIs in block order: `b{k}_{template AOI id}` to WKT."""
    width = math.ceil(math.sqrt(math.ceil(n / len(TEMPLATE_AOIS))))
    geometries = TEMPLATE.aoi_geometries
    tiled = {}
    for index in range(n):
        k, member = divmod(index, len(TEMPLATE_AOIS))
        aoi_id = TEMPLATE_AOIS[member]
        dx, dy = _offset(k % width, k // width)
        tiled[f"b{k}_{aoi_id}"] = shapely.to_wkt(affinity.translate(geometries[aoi_id], dx, dy), rounding_precision=7)
    return tiled


def template_aoi(aoi_id: str) -> str:
    """The template AOI id a tiled AOI id `b{k}_{template AOI id}` copies."""
    return aoi_id.split("_", 1)[1]


def block(aoi_id: str) -> int:
    """The block index `k` of a tiled AOI id `b{k}_{template AOI id}`."""
    return int(aoi_id.split("_", 1)[0].removeprefix("b"))


class TiledFakeApi:
    """Answers Feature API and Roof Age API requests for the AOIs of `aois(n)`, for any `n`, from the template.

    Safe to call from several threads: `FakeApi` calls are serialised, and template answers are memoised by request,
    so the memo holds at most one answer per template AOI and endpoint.
    """

    def __init__(self, until: str | None) -> None:
        self._fake = FakeApi(TEMPLATE, None, until)
        self._until = until
        self._lock = threading.Lock()
        self._memo: dict[tuple[str, str, str], tuple[FakeResponse, dict | None]] = {}
        self._instance_index = {r.hilbert_id: index for index, r in enumerate(TEMPLATE.roof_instances)}

    def handle(self, method: str, url: str, body: bytes | str | None) -> FakeResponse:
        """Answer one request; raise `UnknownRequest` for anything the fake does not route or place in a cell."""
        parts = urlsplit(url)
        route = (method, parts.netloc, parts.path)
        if method == "GET":
            with self._lock:
                return self._fake.handle(method, url, body)
        if route not in (_FEATURES_ROUTE, _ROOF_AGE_ROUTE):
            raise UnknownRequest(f"{method} {parts.netloc}{parts.path}")
        assert body is not None, f"{method} {url} has no body"
        payload = json.loads(body)
        request = shape(payload["aoi"])
        i, j = _cell(request)
        offset = _offset(i, j)
        payload["aoi"] = mapping(shapely.transform(request, lambda c: (c - offset).round(7)))
        answer, parsed = self._template_answer(method, url, json.dumps(payload, sort_keys=True))
        if parsed is None:
            return answer
        moved = copy.deepcopy(parsed)
        for record in moved["features"]:
            if route == _FEATURES_ROUTE:
                record["id"] = _feature_id(i, j, record["id"])
                if record.get("parentId"):
                    record["parentId"] = _feature_id(i, j, record["parentId"])
                if "clippedGeometry" in record:
                    record["clippedGeometry"] = _moved(record["clippedGeometry"], offset)
            else:
                hilbert_id = self._hilbert_id(i, j, record["properties"]["hilbertId"])
                record["id"] = int(hilbert_id)
                record["properties"]["hilbertId"] = hilbert_id
            record["geometry"] = _moved(record["geometry"], offset)
        return dataclasses.replace(answer, text=json.dumps(moved))

    def roof_age_misses(self) -> frozenset[str]:
        """The template AOIs that the template answers with a 404 from the Roof Age API."""
        url = f"https://{HOST}{ROOF_AGE_PATH}"
        return frozenset(
            aoi_id
            for aoi_id, geometry in TEMPLATE.aoi_geometries.items()
            if self._fake.handle(
                "POST", url, json.dumps({"aoi": mapping(geometry), "untilAsOfDate": self._until})
            ).status
            == 404
        )

    def _template_answer(self, method: str, url: str, body: str) -> tuple[FakeResponse, dict | None]:
        key = (method, url, body)
        with self._lock:
            if key not in self._memo:
                answer = self._fake.handle(method, url, body)
                self._memo[key] = (answer, json.loads(answer.text) if answer.status == 200 else None)
                assert len(self._memo) <= 2 * len(TEMPLATE_AOIS), "the memo outgrew one request per template AOI"
            return self._memo[key]

    def _hilbert_id(self, i: int, j: int, template_id: str) -> str:
        cell = j * MAX_CELLS_PER_AXIS + i
        return str(10**18 + cell * len(self._instance_index) + self._instance_index[template_id])


def _cell(request: BaseGeometry) -> tuple[int, int]:
    position = np.array([request.centroid.x, request.centroid.y]) - _ORIGIN
    i, j = (int(v) for v in np.floor(position / STEP))
    if not (0 <= i < MAX_CELLS_PER_AXIS and 0 <= j < MAX_CELLS_PER_AXIS) or (position - _offset(i, j) > _EXTENT).any():
        raise UnknownRequest(f"the request centroid {request.centroid.wkt} is in no tiled block")
    return i, j


def _offset(i: int, j: int) -> np.ndarray:
    return np.array([i * STEP[0], j * STEP[1]])


def _feature_id(i: int, j: int, template_id: str) -> str:
    return str(uuid.uuid5(world.NAMESPACE, f"tiled/{i}/{j}/{template_id}"))


def _moved(geometry: dict, offset: np.ndarray) -> dict:
    return mapping(shapely.transform(shape(geometry), lambda c: c + offset))
