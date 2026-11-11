"""The synthetic world the golden fake API serves: surveys, features, roof instances and error regions, as data.

Every coordinate is synthetic (rectangles laid out in metres from an arbitrary origin), so no AOI is a real parcel.
Class ids are nmaipy's; class descriptions come from the bundled `nmaipy/class_descriptions.json`; attribute block
and component names come only from `vocabulary.json`, which `tests.golden.drift --write-vocabulary` writes from the
live API.
"""

import dataclasses
from functools import cache
import json
import math
from pathlib import Path
import uuid

import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from nmaipy import constants

VOCABULARY_PATH = Path(__file__).parent / "vocabulary.json"
CLASS_DESCRIPTIONS: dict[str, str] = json.loads(
    (Path(constants.__file__).parent / "class_descriptions.json").read_text(encoding="utf-8")
)
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/nearmap/nmaipy/tests/golden")
METRES_PER_DEGREE = 111_320.0
SYSTEM_VERSION = "gen6-golden_world-1.0"


@dataclasses.dataclass(frozen=True)
class Survey:
    """One survey: requests that intersect `region` and whose date window holds `date` are answered from it."""

    name: str
    region: Polygon
    date: str
    mesh_date: str = ""
    system_version: str = SYSTEM_VERSION

    @property
    def survey_id(self) -> str:
        return synthetic_id(f"survey/{self.name}")

    @property
    def resource_id(self) -> str:
        return synthetic_id(f"resource/{self.name}")


@dataclasses.dataclass(frozen=True)
class Component:
    """An attribute block component: `ratio` is its share of the feature's whole (unclipped) area."""

    description: str
    ratio: float
    confidence: float | None
    dominant: bool | None = None


@dataclasses.dataclass(frozen=True)
class Block:
    """An attribute block: components, or plain fields such as `has3dAttributes`."""

    description: str
    components: tuple[Component, ...] | None = None
    fields: tuple[tuple[str, object], ...] = ()


@dataclasses.dataclass(frozen=True)
class Feature:
    """A whole feature. `parent` names the feature it belongs to; `lifecycle` is a parent id the API never returns."""

    name: str
    class_id: str
    geometry: BaseGeometry
    confidence: float
    parent: str = ""
    lifecycle: str = ""
    attributes: tuple[Block, ...] = ()
    fidelity: float | None = None
    is_footprint: bool | None = None

    @property
    def id(self) -> str:
        return synthetic_id(f"feature/{self.name}")

    @property
    def parent_id(self) -> str:
        if self.parent:
            return synthetic_id(f"feature/{self.parent}")
        return synthetic_id(f"lifecycle/{self.lifecycle}") if self.lifecycle else ""


@dataclasses.dataclass(frozen=True)
class RoofInstance:
    """A Roof Age API instance, returned whole for any AOI it intersects."""

    name: str
    geometry: Polygon
    kind: str
    installation_date: str | None
    trust_score: float
    evidence_type: int
    evidence_type_description: str
    capture_dates: tuple[str, ...]
    map_browser_url: bool = True

    @property
    def hilbert_id(self) -> str:
        return str(10**18 + uuid.uuid5(NAMESPACE, f"roof_instance/{self.name}").int % (9 * 10**18))


@dataclasses.dataclass(frozen=True)
class ErrorRegion:
    """Requests to `endpoint` that intersect `region` get `status` with `body`."""

    region: Polygon
    endpoint: str
    status: int
    body: dict


@dataclasses.dataclass(frozen=True)
class World:
    """Everything the fake serves for one case. `parcel_flags` maps (AOI id, feature name) to "outside" (the API
    says the feature does not belong to the parcel) or "multiparcel" (it belongs, clipped to the parcel)."""

    case: str
    country: str
    aois: dict[str, str]
    surveys: tuple[Survey, ...]
    features: tuple[Feature, ...]
    parcel_flags: dict[tuple[str, str], str]
    alpha_only: frozenset[str] = frozenset()
    roof_instances: tuple[RoofInstance, ...] = ()
    roof_age_as_of_date: str = "2025-01-01"
    errors: tuple[ErrorRegion, ...] = ()

    @property
    def aoi_geometries(self) -> dict[str, BaseGeometry]:
        return {aoi_id: shapely.from_wkt(wkt) for aoi_id, wkt in self.aois.items() if wkt.startswith("POLYGON ((")}


def build(case: str) -> World:
    """The world of a golden case."""
    return {"us": us_world, "au": au_world}[case]()


def synthetic_id(name: str) -> str:
    """A stable uuid for a readable name."""
    return str(uuid.uuid5(NAMESPACE, name))


@cache
def vocabulary() -> dict:
    """The live API's names (`vocabulary.json`)."""
    return json.loads(VOCABULARY_PATH.read_text(encoding="utf-8"))


def class_id(description: str) -> str:
    """The id of the bundled class with this description, else a synthetic id (components with no class)."""
    ids = [cid for cid, d in CLASS_DESCRIPTIONS.items() if d == description]
    assert len(ids) <= 1, f"{description!r} names several classes"
    return ids[0] if ids else synthetic_id(f"class/{description}")


def us_world() -> World:
    """Five parcels, a 403 and a no-survey 404, with roof ages; 3D survey."""
    o = (-100.5, 40.5)
    aoi = {
        "u1": _rect(o, 0, 0, 30, 40),
        "u2": _rect(o, 40, 0, 30, 40),
        "u3": _rect(o, 70, 0, 30, 40),
        "u4": _rect(o, 110, 0, 30, 40),
        "u5": _rect(o, 150, 0, 30, 40),
        "u403": _rect(o, 0, 200, 30, 30),
        "u404": _rect(o, 0, 1000, 30, 30),
    }
    b = _Builder("us", o)
    building_3d = (
        Block(
            "Building 3d attributes",
            fields=(
                ("has3dAttributes", True),
                ("height", 7.25),
                ("heightM", 7.3),
                ("heightFt", 24),
                ("numStories", {"1": 0.1, "2": 0.85, "3+": 0.05}),
            ),
        ),
        Block("Building pitch", fields=(("available", True), ("value", 30.5))),
        Block("Ground height", fields=(("available", True), ("value", 120.25))),
    )
    building_2d = (
        Block("Building 3d attributes", fields=(("has3dAttributes", False),)),
        Block("Building pitch", fields=(("available", False),)),
        Block("Ground height", fields=(("available", False),)),
    )

    b.house(
        "R1",
        (8, 12, 14, 14),
        materials={"Shingle": 0.714, "Metal": 0.286},
        types={"Hip": 0.571, "Gable": 0.429, "Flat (Deprecated)": 0.02},
        overhang={"Tree Overhang": 0.082},
        roof_3d=(("has3dAttributes", True), ("pitch", 30.5)),
        condition={"Tile or Shingle Staining": 0.02, "Roof with Permanent Repair": 0.02},
        building_attributes=building_3d,
        children={
            "Shingle": (8, 12, 10, 14),
            "Metal": (18, 12, 4, 14),
            "Hip": (8, 12, 8, 14),
            "Gable": (16, 12, 6, 14),
            "Flat (Deprecated)": (20, 24, 2, 2),
            "Tree Overhang": (8, 22, 4, 4),
            "Tile or Shingle Staining": (10, 14, 2, 2),
            "Roof with Permanent Repair": (19, 14, 2, 2),
        },
    )
    b.add(
        "P1",
        "Swimming Pool",
        (24, 2, 4, 6),
        0.93,
        attributes=(
            b.block(constants.POOL_ID, "Pool Condition", {"Covered Swimming Pool": 0.25, "Empty swimming pool": 0.0}),
        ),
    )
    b.add("L1", "Lawn Grass", (-5, 0, 40, 10), 0.88)
    b.add("V1", "Medium and High Vegetation (>2m)", (25, 10, 20, 20), 0.9)
    b.add("C1", "Concrete Slab", (2, 2, 8, 6), 0.86)
    b.add("D1", "Driveway", None, 0.8, geometry=MultiPolygon([_rect(o, 1, 30, 4, 9), _rect(o, 25, 30, 4, 9)]))

    b.house(
        "R2",
        (62, 10, 16, 20),
        materials={"Tile": 0.4375, "Metal": 0.5625},
        types={"Gable": 1.0},
        overhang={"Tree Overhang": 0.05},
        roof_3d=(("has3dAttributes", True), ("pitch", 22.0)),
        condition={"Roof Rusting": 0.0},
        building_attributes=building_3d,
        children={"Tile": (62, 10, 7, 20), "Metal": (69, 10, 9, 20), "Gable": (62, 10, 16, 20)},
        extra_children={"R2/overhang": ("Tree Overhang", (74, 26, 4, 4))},
    )
    b.house(
        "R3",
        (45, 35, 15, 15),
        materials={"Shingle": 1.0},
        types={"Hip": 1.0},
        overhang={"Tree Overhang": 0.0},
        roof_3d=(("has3dAttributes", True), ("pitch", 25.0)),
        condition={"Roof Rusting": 0.0},
        building_attributes=building_3d,
        children={"Hip": (45, 35, 15, 15)},
    )
    b.house(
        "R4",
        (85, 30, 12, 12),
        materials={"Shingle": 0.9, "Metal": 0.1},
        types={"Dutch Gable": 0.8, "Dormer Window": 0.02},
        overhang={"Tree Overhang": 0.0},
        roof_3d=(("has3dAttributes", True), ("pitch", 35.0)),
        condition={"Roof Rusting": 0.0},
        building_attributes=building_3d,
        children={"Dutch Gable": (85, 30, 12, 10)},
        extra_children={"R4/dormer": ("Dormer Window", (88, 40.5, 2, 1))},
    )
    b.house(
        "R5",
        (155, 10, 15, 15),
        materials={"Tile": 1.0},
        types={"Hip": 1.0},
        overhang={"Tree Overhang": 0.0},
        roof_3d=(("has3dAttributes", False),),
        condition={"Roof Rusting": 0.0},
        building_attributes=building_2d,
        children={"Tile": (155, 10, 15, 15), "Hip": (155, 10, 15, 15)},
    )
    b.house(
        "R13",
        (172, 30, 6, 6),
        materials={"Metal": 1.0},
        types={"Flat": 1.0},
        overhang={"Tree Overhang": 0.0},
        roof_3d=(("has3dAttributes", False),),
        condition={"Roof Rusting": 0.0},
        building=False,
    )
    b.add("B14", "Building", (150, 30, 4, 6), 0.9, lifecycle="B14", attributes=building_2d, fidelity=0.8)
    b.add("V5", "Very Low Vegetation (<0.5m)", (150, 0, 30, 8), 0.77)
    b.flag(["u2", "u3"], "R2", "multiparcel")
    b.flag(["u2"], "R3", "outside")

    roof_instances = (
        RoofInstance(
            "RI1",
            _rect(o, 8.5, 12.5, 13.5, 13.5),
            "roof",
            "2012-06-15",
            61.5,
            5,
            "multiple_images",
            ("2011-03-01", "2013-04-10", "2024-10-05"),
        ),
        RoofInstance(
            "RI2",
            _rect(o, 62, 10, 16, 19.5),
            "roof",
            "2018-09-01",
            74.0,
            7,
            "multiple_images_with_change",
            ("2017-04-03", "2018-11-20", "2024-10-05"),
        ),
        RoofInstance(
            "RT1",
            Polygon([_point(o, 21, 28), _point(o, 30, 28), _point(o, 30, 19)]),
            "roof",
            "2009-05-01",
            40.0,
            5,
            "multiple_images",
            ("2008-02-01", "2010-03-15"),
        ),
        RoofInstance("RX1", _rect(o, 1, 31, 4, 7), "roof", "2016-08-20", 55.0, 5, "multiple_images", ("2016-09-30",)),
        RoofInstance("RI5", _rect(o, 155, 10, 15, 15), "roof", None, 0.0, 3, "single_image", ("2024-10-05",)),
        RoofInstance(
            "RP5",
            _rect(o, 150, 0, 30, 40),
            "parcel",
            None,
            0.0,
            0,
            "no_evidence",
            ("2024-10-05",),
            map_browser_url=False,
        ),
    )
    return World(
        case="us",
        country="us",
        aois={k: v.wkt for k, v in aoi.items()},
        surveys=(Survey("us", _rect(o, -50, -50, 350, 350), "2025-03-18", mesh_date="2025-03-18"),),
        features=tuple(b.features),
        parcel_flags=b.flags,
        alpha_only=frozenset({class_id("All-in Vegetation")}),
        roof_instances=roof_instances,
        errors=(
            ErrorRegion(
                _rect(o, -10, 190, 50, 50), "features", 403, {"message": "Not authorized for this synthetic area"}
            ),
        ),
    )


def au_world() -> World:
    """Small parcels (one empty, three invalid rows) and one AOI large enough to be gridded; 2D surveys."""
    small, large = (149.5, -33.5), (139.5, -34.5)
    aoi = {
        "a1": _rect(small, 0, 0, 25, 35).wkt,
        "a2": _rect(small, 35, 0, 25, 35).wkt,
        "a3": _rect(small, 60, 0, 25, 35).wkt,
        "a4": _rect(small, 95, 0, 25, 35).wkt,
        "blank": "",
        "point": "POINT(149.5 -33.5)",
        "empty": "POLYGON(EMPTY)",
        "big": _rect(large, 0, 0, 1100, 1000).wkt,
    }
    s = _Builder("au", small)
    s.house(
        "R6",
        (5, 8, 15, 16),
        materials={"Tile": 0.8, "Metal": 0.2},
        types={"Hip": 0.6, "Gable": 0.4},
        overhang={"Tree Overhang": 0.1, "Leaf-off Tree Overhang": 0.02},
        roof_3d=(("has3dAttributes", False),),
        children={
            "Tile": (5, 8, 15, 12.8),
            "Metal": (5, 20.8, 15, 3.2),
            "Hip": (5, 8, 9, 16),
            "Gable": (14, 8, 6, 16),
            "Tree Overhang": (5, 20, 6, 4),
            "Leaf-off Tree Overhang": (17, 22, 2, 2),
        },
    )
    s.add("V6", "Medium and High Vegetation (>2m)", (-5, 25, 35, 15), 0.91)
    s.add("W6", "Woody Vegetation", (0, 0, 10, 6), 0.87)
    s.house(
        "R7",
        (52, 8, 16, 16),
        materials={"Tile": 0.375, "Metal": 0.625},
        types={"Gable": 1.0},
        overhang={"Tree Overhang": 0.0625},
        roof_3d=(("has3dAttributes", False),),
        children={"Tile": (52, 8, 6, 16), "Metal": (58, 8, 10, 16), "Gable": (52, 8, 16, 16)},
        extra_children={"R7/overhang": ("Tree Overhang", (64, 20, 4, 4))},
    )
    s.house(
        "R8",
        (70, -12, 10, 14),
        materials={"Metal": 1.0},
        types={"Hip": 1.0},
        overhang={"Tree Overhang": 0.0},
        roof_3d=(("has3dAttributes", False),),
        children={"Hip": (70, -12, 10, 14)},
    )
    s.house(
        "R12",
        (72, 14, 10, 12),
        materials={"Tile": 1.0},
        types={"Hip": 1.0},
        overhang={"Tree Overhang": 0.0},
        roof_3d=(("has3dAttributes", False),),
        children={"Tile": (72, 14, 10, 12)},
    )
    s.flag(["a2", "a3"], "R7", "multiparcel")
    s.flag(["a3"], "R8", "outside")

    g = _Builder("au", large)
    g.house(
        "R9",
        (450, 100, 20, 15),
        materials={"Metal": 1.0},
        types={"Gable": 1.0},
        overhang={"Tree Overhang": 0.0},
        roof_3d=(("has3dAttributes", False),),
        children={"Metal": (450, 100, 20, 15)},
    )
    g.house("R10", (900, 540, 30, 30), bare=True)
    g.house(
        "R11",
        (1090, 300, 20, 15),
        materials={"Tile": 1.0},
        types={"Hip": 1.0},
        overhang={"Tree Overhang": 0.0},
        roof_3d=(("has3dAttributes", False),),
    )
    g.add("V9", "Medium and High Vegetation (>2m)", (430, 530, 60, 50), 0.9)

    return World(
        case="au",
        country="au",
        aois=aoi,
        surveys=(
            Survey("au-small", _rect(small, -50, -50, 250, 150), "2025-01-20"),
            Survey("au-large", _rect(large, -50, -50, 1200, 1100), "2025-03-06"),
        ),
        features=tuple(s.features + g.features),
        parcel_flags=s.flags,
        alpha_only=frozenset({class_id("All-in Vegetation")}),
    )


class _Builder:
    """Lays out features in metres from an origin, checking every attribute name against the vocabulary."""

    def __init__(self, case: str, origin: tuple[float, float]) -> None:
        self.case, self.origin = case, origin
        self.features: list[Feature] = []
        self.flags: dict[tuple[str, str], str] = {}

    def add(
        self,
        name: str,
        description: str,
        box: tuple[float, float, float, float] | None,
        confidence: float,
        geometry: BaseGeometry | None = None,
        **kwargs: object,
    ) -> None:
        geometry = geometry if geometry is not None else _rect(self.origin, *box)
        self.features.append(Feature(name, class_id(description), geometry, confidence, **kwargs))

    def block(
        self, feature_class: str, description: str, components: dict[str, float] | None = None, **fields
    ) -> Block:
        names = vocabulary()["attributes"][self.case][feature_class]
        assert description in names, f"{description!r} is not a live attribute block of {feature_class}"
        if components is None:
            return Block(description, fields=tuple(fields.items()))
        assert set(components) <= set(names[description]), f"{set(components) - set(names[description])} not live"
        largest = max(components, key=components.get)
        dominant = (lambda c: c == largest) if description == "Roof material" else (lambda c: None)
        return Block(
            description,
            tuple(
                Component(c, r, round(0.5 + r / 2, 3) if r else None, dominant(c))
                for c, r in {**dict.fromkeys(names[description], 0.0), **components}.items()
            ),
        )

    def house(
        self,
        roof: str,
        box: tuple[float, float, float, float],
        materials: dict[str, float] | None = None,
        types: dict[str, float] | None = None,
        overhang: dict[str, float] | None = None,
        roof_3d: tuple[tuple[str, object], ...] = (),
        condition: dict[str, float] | None = None,
        building_attributes: tuple[Block, ...] = (),
        children: dict[str, tuple[float, float, float, float]] | None = None,
        extra_children: dict[str, tuple[str, tuple[float, float, float, float]]] | None = None,
        bare: bool = False,
        building: bool = True,
    ) -> None:
        """A roof, its deprecated building twin (the roof's parent), a building, and the roof's child features."""
        roof_blocks = ()
        if not bare:
            roof_blocks = (
                self.block(constants.ROOF_ID, "Roof material", materials),
                self.block(constants.ROOF_ID, "Roof types", types),
                self.block(constants.ROOF_ID, "Roof overhang attributes", overhang),
                self.block(constants.ROOF_ID, "Roof 3d attributes", **dict(roof_3d)),
            )
            if condition is not None:
                roof_blocks += (self.block(constants.ROOF_ID, "Roof Condition", condition),)
        for blk in building_attributes:
            assert blk.description in vocabulary()["attributes"][self.case][constants.BUILDING_NEW_ID], blk.description
        if building:
            self.add(
                f"{roof}/building", "Building", box, 0.97, lifecycle=roof, attributes=building_attributes, fidelity=0.9
            )
        self.add(
            f"{roof}/twin",
            "Building (Deprecated)",
            box,
            0.97,
            lifecycle=roof,
            attributes=building_attributes,
            fidelity=0.9,
        )
        self.add(
            roof, "Roof", box, 0.96, parent=f"{roof}/twin", attributes=roof_blocks, fidelity=0.85, is_footprint=False
        )
        for description, child_box in (children or {}).items():
            self.add(f"{roof}/{description}", description, child_box, 0.8, parent=roof)
        for name, (description, child_box) in (extra_children or {}).items():
            self.add(name, description, child_box, 0.8, parent=roof)

    def flag(self, aois: list[str], roof: str, flag: str) -> None:
        """Flag a roof, its twin and its building for these AOIs (children follow their roof)."""
        for aoi in aois:
            for name in (roof, f"{roof}/twin", f"{roof}/building"):
                self.flags[(aoi, name)] = flag


def _rect(origin: tuple[float, float], x: float, y: float, width: float, height: float) -> Polygon:
    corners = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    return Polygon([_point(origin, cx, cy) for cx, cy in corners])


def _point(origin: tuple[float, float], x: float, y: float) -> tuple[float, float]:
    lon = origin[0] + x / (METRES_PER_DEGREE * math.cos(math.radians(origin[1])))
    return round(lon, 7), round(origin[1] + y / METRES_PER_DEGREE, 7)
