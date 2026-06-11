import json
import os
from pathlib import Path

# Default columns
AOI_ID_COLUMN_NAME = "aoi_id"
LAT_PRIMARY_COL_NAME = "lat"
LON_PRIMARY_COL_NAME = "lon"
SINCE_COL_NAME = "since"
UNTIL_COL_NAME = "until"
SURVEY_RESOURCE_ID_COL_NAME = "survey_resource_id"

DEFAULT_URL_ROOT = "api.nearmap.com/ai/features/v4/bulk"

# ============================================================================
# Roof Age API Configuration
# ============================================================================
ROOF_AGE_URL_ROOT = "api.nearmap.com/ai/roofage/v1"
ROOF_AGE_RESOURCE_PATH = "resources"
ROOF_AGE_DEFAULT_RESOURCE_ID = "latest"

# Roof Age dataset resource ids. These identify *datasets* (Roof Age API "resources"),
# not model versions — the per-row `model_version` field returned by the API is the
# source of truth for which model produced a given record. Current API team practice
# is to bump the dataset alias whenever the underlying model changes, but the two are
# not formally 1:1.
#
# `latest` is a pointer maintained by the API team that today serves A.0; it will be
# bumped to A.1 (and beyond) as new datasets are released.
ROOF_AGE_A0_RESOURCE_ID = "81b605d0-d70d-592d-a1f0-a14c7d020912"
ROOF_AGE_A1_RESOURCE_ID = "cf6bf06a-c8f7-58bd-9b1e-bce8e089a9bc"

# Friendly aliases for known datasets. Unknown values pass through and are
# treated as raw resource UUIDs.
ROOF_AGE_DATASET_ALIASES: dict[str, str] = {
    "latest": "latest",
    "A.0": ROOF_AGE_A0_RESOURCE_ID,
    "A.1": ROOF_AGE_A1_RESOURCE_ID,
}

# Datasets that do not support the `untilAsOfDate` / `sinceAsOfDate` body parameters.
# As of writing, only A.0. `latest` is intentionally NOT in this set: today it points
# at A.0 (so the API will reject cutoff requests with HTTP 500 against `latest`), but
# once it's bumped to A.1+ the same client invocation will start working without any
# code change here.
ROOF_AGE_NO_CUTOFF_RESOURCE_IDS = frozenset({ROOF_AGE_A0_RESOURCE_ID})


def resolve_roof_age_dataset(value: str) -> str:
    """Resolve a dataset alias (e.g. ``A.1``) to its resource id; pass unknown values through."""
    return ROOF_AGE_DATASET_ALIASES.get(value, value)


def format_no_cutoff_error(*, flag: str = "--until") -> str:
    """Canonical error message for combining a cutoff flag with the A.0 dataset.

    Used by both exporters at all three rejection sites (argparse, constructor, AOI-load)
    so the wording stays consistent and only changes in one place if the rule shifts.

    `flag` may be passed with or without a leading ``--``; we canonicalise to dashed form
    so the rendered error reads consistently regardless of caller (argparse passes
    ``"--until"``; constructor sites pass bare names like ``"until"``).
    """
    canonical = flag if flag.startswith("--") else f"--{flag}"
    return (
        f"{canonical} is not supported on the A.0 Roof Age dataset (which lacks "
        "`untilAsOfDate` and `sinceAsOfDate` body parameter support). "
        "Pass --roof-age-dataset A.1 (or any A.1+ resource UUID), or drop "
        f"{canonical} if you only need it for the Feature API."
    )


# Roof Age API pagination settings
ROOF_AGE_DEFAULT_PAGE_LIMIT = 1000  # Default max features per page (API default)
ROOF_AGE_NEXT_CURSOR_FIELD = "nextCursor"  # Field name for pagination cursor in response

# Minimum IoU threshold for roof-to-roof-instance matching
# Matches below this threshold are not trusted and will not be assigned as parent/primary
MIN_ROOF_INSTANCE_IOU_THRESHOLD = 0.005

# Roof Age API response field names (camelCase, used for pre-conversion logic in roof_age_api.py)
ROOF_AGE_AREA_FIELD = "area"
ROOF_AGE_TIMELINE_FIELD = "timeline"
ROOF_AGE_RELEVANT_PERMITS_DETAILS_FIELD = "relevantPermitsDetails"
ROOF_AGE_ASSESSOR_DATA_DETAILS_FIELD = "assessorDataDetails"
ROOF_AGE_HILBERT_ID_FIELD = "hilbertId"
ROOF_AGE_RESOURCE_ID_FIELD = "resourceId"
ROOF_AGE_MODEL_VERSION_FIELD = "modelVersion"
ROOF_AGE_MAPBROWSER_URL_FIELD = "mapBrowserUrl"

# Roof Age API columns that should receive the roof_age_ prefix during export.
# Ordered list of snake_case column names from _parse_response() that are
# specific to the Roof Age API. Only these get the ``roof_age_`` prefix
# applied during flattening; all other columns (standard feature columns,
# exporter columns, Feature API columns) are left untouched. Iteration order
# below is the canonical column order applied wherever roof-age data is
# emitted (roof_instance.csv, rollup primary_roof_instance_*, roof.csv
# primary_child_roof_age_*, building.csv chained primary_child_roof_*_*).
ROOF_AGE_PREFIX_COLUMNS = (
    "map_browser_url",
    "installation_date",
    "evidence_type",
    "evidence_type_description",
    "trust_score",
    "before_installation_capture_date",
    "after_installation_capture_date",
    "min_capture_date",
    "max_capture_date",
    "number_of_captures",
    "as_of_date",
    "until_date",
    "kind",
    "assessor_data",
    "assessor_data_details",
    "relevant_permits",
    "relevant_permits_details",
    "model_version",
)


# ============================================================================
# HTTP Request Configuration
# ============================================================================
# These constants control retry behavior and timeouts for API requests.
# The retry logic uses exponential backoff to handle transient failures
# gracefully while preventing indefinite blocking on persistent errors.

# Maximum number of retry attempts for failed requests
# Uses full jitter backoff: sleep = uniform(0, min(BACKOFF_MAX, BACKOFF_FACTOR * 2^n))
# Retry delays are approximately: 0s, 0-1s, 0-2s, 0-4s, 0-8s, 0-16s, 0-32s, 0-60s (capped), ...
MAX_RETRIES = 10

# Exponential backoff multiplier for retries
# Full jitter formula: sleep = uniform(0, min(BACKOFF_MAX, BACKOFF_FACTOR * 2^(retry_number - 1)))
# Jitter scales with retry count: fast recovery on early retries, strong desync on late ones.
BACKOFF_FACTOR = 0.5

# Maximum backoff time in seconds (caps the full jitter range on late retries)
# With 480 parallel workers all retrying at once: 480 / 60 ≈ 8 req/s
BACKOFF_MAX = 60

# Maximum random sleep before a thread's first AOI request, to spread the chunk-start burst.
# With 50 threads: requests spread over 5s = ~10 req/s at startup instead of an instant spike.
THREAD_STARTUP_JITTER_SECONDS = 5.0

# Maximum time to wait for initial server response (connection + waiting for first byte)
TIMEOUT_SECONDS = 120

# Maximum time to wait for reading complete response body after initial response
# Set lower than TIMEOUT_SECONDS to detect stalled connections faster
READ_TIMEOUT_SECONDS = 90

# Delay between retries for ChunkedEncodingError (network-level errors)
CHUNKED_ENCODING_RETRY_DELAY = 1.0

# Log requests that exceed this duration (helps identify performance issues)
SLOW_REQUEST_THRESHOLD_SECONDS = 60

# Sentinel value for error cases where no HTTP status code is available
DUMMY_STATUS_CODE = -1

# ============================================================================
# API Warmup Configuration
# ============================================================================
# When running with many parallel workers, submitting all chunks simultaneously
# can overwhelm the API before its autoscaling has time to respond. This
# setting staggers initial chunk submissions so the API sees a gradually
# rising load and can scale backend capacity along with it.

# Seconds to wait between adding each parallel worker during the warmup
# period. Applied to the first N chunks, where N = number of parallel
# processes. With the current 60s value, a 12-process run reaches full
# concurrency at 11:00 and an 8-process run at 7:00. Bumped from 40s
# after observing sustained HTTP 500s past the 6-minute mark on larger
# workloads — the API autoscaler needed more headroom before saturation.
# Set to 0 to disable warmup.
API_WARMUP_INTERVAL_SECONDS = 60.0

# ============================================================================
# Post-Processing Parallelism Configuration
# ============================================================================
# Thread count for parallel parquet reads during rollup/error consolidation.
# These reads are I/O-bound (S3 GET requests), so threads overlap network latency.
PARALLEL_READ_WORKERS = 8

# Higher thread count for S3 I/O-bound operations. S3 reads have high per-request
# latency (~50-200ms) so more threads overlap network round-trips effectively.
# Memory impact is negligible (thread stacks only, no extra data in flight).
S3_PARALLEL_READ_WORKERS = 24

# Feature streaming prefetch: worker threads that read chunks ahead while the
# main thread processes and writes the current chunk. Used for both local-disk
# and S3 streaming — the higher S3_PARALLEL_READ_WORKERS value is reserved for
# one-shot parallel reads where results don't accumulate.
#
# The count is derived per run as round(FEATURE_PREFETCH_MULTIPLIER * processes),
# floored at FEATURE_PREFETCH_FLOOR (see _resolve_prefetch_workers in exporter.py),
# so it tracks --processes: dropping processes to cap RAM also shrinks the prefetch
# buffer in step, and a bigger box (more processes) reads further ahead. Memory
# bounded: at most (workers + 1) dense-chunk feature tables resident at once
# (multi-GB each on heavy workloads).
FEATURE_PREFETCH_MULTIPLIER = 1.5
FEATURE_PREFETCH_FLOOR = 2

# ============================================================================
# Primary Feature Selection Configuration
# ============================================================================

# Distance tolerance in meters for "nearest" primary selection
# If no feature contains the target point, the nearest non-small feature
# within this tolerance is selected as primary
NEAREST_TOLERANCE_METERS = 1.0

# ============================================================================
# Geometry Processing Configuration
# ============================================================================

# Grid cell size when subdividing large AOIs for parallel processing
# Approximately 500m at the equator
GRID_SIZE_DEGREES = 0.005

# Maximum AOI area in square meters before forcing gridding
# This threshold (1 sq km) prevents backend API issues that occurred when the limit
# was raised from 1 to 25 sq km. The conservative value ensures stable API responses.
MAX_AOI_AREA_SQM_BEFORE_GRIDDING = 1_000_000  # 1 square kilometer

# Projections
LAT_LONG_CRS = "WGS 84"
AREA_CRS = {
    "au": "epsg:3577",
    "ca": "esri:102001",
    "nz": "epsg:2193",
    "us": "esri:102003",
}
API_CRS = "EPSG:4326"

IMPERIAL_COUNTRIES = ["us"]


class MeasurementUnits:
    def __init__(self, country):
        self.country = country

    def area_units(self):
        if self.country in IMPERIAL_COUNTRIES:
            area_units = "sqft"
        else:
            area_units = "sqm"
        return area_units


def country_area_suffix(country: str) -> str:
    """Return the area-column suffix for the given country: 'sqft' for imperial, 'sqm' otherwise."""
    return "sqft" if country.lower() in IMPERIAL_COUNTRIES else "sqm"


def wrong_unit_area_columns(country: str) -> list[str]:
    """Return the area columns to drop from per-feature exports for the given country.

    The Feature API returns both metric and imperial columns; per-feature exports keep only
    the country-correct family to match rollup behaviour.
    """
    drop = "sqm" if country.lower() in IMPERIAL_COUNTRIES else "sqft"
    return [f"area_{drop}", f"clipped_area_{drop}", f"unclipped_area_{drop}"]


# Error Codes
AOI_EXCEEDS_MAX_SIZE = "AOI_EXCEEDS_MAX_SIZE"
POLYGON_TOO_COMPLEX = "POLYGON_TOO_COMPLEX"

# Units
METERS_TO_FEET = 3.28084
SQUARED_METERS_TO_SQUARED_FEET = METERS_TO_FEET * METERS_TO_FEET

# The address fields expected by the address endpoint. state should be statecode (2 digit) but these
# are the API fields
ADDRESS_FIELDS = ("streetAddress", "city", "state", "zip")

# Class IDs
BUILDING_ID = "a2e4ae39-8a61-5515-9d18-8900aa6e6072"  # DEPRECATED: Legacy clone of roof semantic
BUILDING_NEW_ID = "1878ccf6-46ec-55a7-a20b-0cf658afb755"  # Current semantic building definition
ROOF_ID = "c08255a4-ba9f-562b-932c-ff76f2faeeeb"
BUILDING_LIFECYCLE_ID = "91987430-6739-5e16-b92f-b830dd7d52a6"  # damage scores are attached to this class
BUILDING_UNDER_CONSTRUCTION_ID = "4794d3ec-0ee7-5def-ad56-f82ff7639bce"
FLAT_DEPRECATED_ROOF_ID = "224f98d3-b853-542a-8b18-e1e46e3a8200"  # DEPRECATED: Replaced by CLASS_1191_FLAT

# Deprecated class IDs - filtered out early in processing, before any saved chunk files
DEPRECATED_CLASS_IDS = [
    BUILDING_ID,  # Replaced by BUILDING_NEW_ID
    FLAT_DEPRECATED_ROOF_ID,  # Replaced by CLASS_1191_FLAT
]

# Roof Instance - a temporal slice of a roof from the Roof Age API
# This is a "virtual" feature class that represents roof installation date information
# Roof instances may spatially correspond to roof objects from the Feature API, but are
# tracked separately as they come from different data sources with different semantics
# Note: This is a synthetic UUID (not a real Feature API class_id) but valid format
# f00f = "roof", 1a9e = "age" in hex-speak
ROOF_INSTANCE_CLASS_ID = "f00f1a9e-0001-4000-a000-000000000001"

BUILDING_STYLE_CLASS_IDS = [
    BUILDING_LIFECYCLE_ID,
    BUILDING_NEW_ID,
    ROOF_ID,
    BUILDING_UNDER_CONSTRUCTION_ID,
    BUILDING_ID,  # Deprecated but still valid
    ROOF_INSTANCE_CLASS_ID,
]


TRAMPOLINE_ID = "753621ee-0b9f-515e-9bcf-ea40b96612ab"
POOL_ID = "0339726f-081e-5a6e-b9a9-42d95c1b5c8a"
CONSTRUCTION_ID = "a2a81381-13c6-57dc-a967-af696e45f6c7"
SOLAR_ID = "3680e1b8-8ae1-5a15-8ec7-820078ef3298"
SOLAR_HW_ID = "c1143023-135b-54fd-9a07-8de0ff55de51"
CAR_ID = "8337e0e1-e171-5292-89cc-99c0da2a4fe4"
WHEELED_CONSTRUCTION_VEHICLE_ID = "75efd1e7-c253-59f0-b3aa-95f9c17efa93"
CONSTRUCTION_CRANE_ID = "6a2c2adb-0914-56b3-8a2d-871b803a0dd7"
BOAT_ID = "62a0958e-2139-5688-a776-b88c6049d50e"
SILO_ID = "b64ecdb0-6810-5c70-835c-9e2a5f2a4d84"
SKYLIGHT_ID = "3f5a737e-6d56-538a-ac26-f2934bbbb695"
PLAYGROUND_ID = "7741703d-4ce4-54e1-a9ee-05a0a1851137"

VEG_VERYLOW_ID = "a7d921b7-393c-4121-b317-e9cda3e4c19b"
VEG_LOW_ID = "2780fa70-7713-437c-ad98-656b8a5cc4f2"
VEG_MEDHIGH_ID = "dfd8181b-80c9-4234-9d05-0eef927e3aca"
VEG_WOODY_1107_ID = "eaa83113-44b3-505e-9515-ba8a8d403dd4"
VEG_WOODY_COMPOSITE_ID = "30fc0c55-2b61-569f-b424-44082987ecb9"
VEG_IDS = [VEG_VERYLOW_ID, VEG_LOW_ID, VEG_MEDHIGH_ID, VEG_WOODY_1107_ID]

DIRT_GRAVEL_SAND_ID = "0ad1355f-5dfd-403b-8b8b-b7d8ed95731f"
WATER_BODY_ID = "2e0bd9e3-3b67-4990-84dc-1b4812fdd02b"
CONCRETE_ID = "290897be-078b-4948-97aa-755289a67a29"
ASPHALT_ID = "97a1f8a8-7cf2-4e81-82b4-753ee225d9ed"
LAWN_GRASS_ID = "68dc5061-5842-4a17-8073-e278a91b607d"
SURFACES_IDS = [
    WATER_BODY_ID,
    CONCRETE_ID,
    ASPHALT_ID,
    LAWN_GRASS_ID,
    DIRT_GRAVEL_SAND_ID,
]

METAL_ROOF_ID = "4424186a-0b42-5608-a5a0-d4432695c260"
TILE_ROOF_ID = "516fdfd5-0be9-59fe-b849-92faef8ef26e"
SHINGLE_ROOF_ID = "4bbf8dbd-cc81-5773-961f-0121101422be"

HIP_ROOF_ID = "ac0a5f75-d8aa-554c-8a43-cee9684ef9e9"
GABLE_ROOF_ID = "59c6e27e-6ef2-5b5c-90e7-31cfca78c0c2"
DUTCH_GABLE_ROOF_ID = "3719eb40-d6d1-5071-bbe6-379a551bb65f"
TURRET_ROOF_ID = "89582082-e5b8-5853-bc94-3a0392cab98a"

TREE_OVERHANG_ID = "8e9448bd-4669-5f46-b8f0-840fee25c34c"

STRUCTURALLY_DAMAGED_ROOF = "f907e625-26b3-59db-a806-d41f62ce1f1b"
TEMPORARY_REPAIR = "abb1f304-ce01-527b-b799-cbfd07551b2c"
ROOF_PONDING = "f41e02b0-adc0-5b46-ac95-8c59aa9fe317"
ROOF_RUSTING = "526496bf-7344-5024-82d7-77ceb671feb4"
TILE_SHINGLE_DISCOLOURATION = "cfa8951a-4c29-54de-ae98-e5f804c305e3"

LEAF_OFF_VEG_ID = "cd47dfd1-2c24-543c-89fd-7677b2cc100b"
DRIVEABLE_ID = "372fb6c1-a3ab-5019-ba0f-489ed12079de"

CLASSES_WITH_PRIMARY_FEATURE = BUILDING_STYLE_CLASS_IDS + [
    POOL_ID,
    ROOF_INSTANCE_CLASS_ID,
]  # Can add more where we particularly care about attributes for the largest feature

# Whitelist of class IDs that get individual per-class files (e.g. "roof.csv",
# "roof_features.parquet"). Classes not in this set are still included in the
# combined features.parquet but do not get their own tabular/geo exports.
PER_CLASS_FILE_CLASS_IDS = {
    BUILDING_LIFECYCLE_ID,
    BUILDING_NEW_ID,
    ROOF_ID,
    ROOF_INSTANCE_CLASS_ID,
    POOL_ID,
    SOLAR_ID,
}

# Map rollup column names to class IDs for is_primary merge
# Used to mark which features are primary in per-class exports
# Column names are derived from API class descriptions (lowercased, spaces→underscores)
PRIMARY_FEATURE_COLUMN_TO_CLASS = {
    "primary_roof_feature_id": ROOF_ID,
    "primary_building_feature_id": BUILDING_NEW_ID,
    "primary_building_(deprecated)_feature_id": BUILDING_ID,
    "primary_building_lifecycle_feature_id": BUILDING_LIFECYCLE_ID,
    "primary_building_under_construction_feature_id": BUILDING_UNDER_CONSTRUCTION_ID,
    "primary_swimming_pool_feature_id": POOL_ID,
    "primary_roof_instance_feature_id": ROOF_INSTANCE_CLASS_ID,
}

# Class IDs whose per-class exports should expose `is_primary`.
# Derived from PRIMARY_FEATURE_COLUMN_TO_CLASS so the two stay in lockstep.
PRIMARY_FEATURE_CLASS_IDS = frozenset(PRIMARY_FEATURE_COLUMN_TO_CLASS.values())

# Class IDs for which the Feature API populates a top-level `fidelity` field.
# Roof instances (Roof Age API) use `trust_score` instead of fidelity.
CLASSES_WITH_FIDELITY = frozenset(BUILDING_STYLE_CLASS_IDS) - {ROOF_INSTANCE_CLASS_ID}

# Human-readable descriptions for feature classes, loaded from class_descriptions.json.
# This file is auto-refreshed from the live Feature API during export runs.
# To manually refresh: python -c "from nmaipy.constants import refresh_class_descriptions; refresh_class_descriptions()"
_CLASS_DESCRIPTIONS_PATH = Path(__file__).parent / "class_descriptions.json"
try:
    with open(_CLASS_DESCRIPTIONS_PATH) as _f:
        FEATURE_CLASS_DESCRIPTIONS = json.load(_f)
except (FileNotFoundError, json.JSONDecodeError):
    FEATURE_CLASS_DESCRIPTIONS = {}


def refresh_class_descriptions(api_key: str = None):
    """Fetch class descriptions from the live Feature API and update class_descriptions.json.

    Preserves synthetic entries (e.g. Roof Instance from Roof Age API) that aren't
    returned by the Feature API.
    """
    from nmaipy.feature_api import FeatureApi

    api = FeatureApi(api_key=api_key or os.environ.get("API_KEY"))
    classes_df = api.get_feature_classes()
    api.cleanup()
    descriptions = dict(zip(classes_df.index, classes_df["description"]))
    # Preserve synthetic entries not from Feature API
    for cid in PER_CLASS_FILE_CLASS_IDS:
        if cid not in descriptions:
            descriptions.setdefault(cid, FEATURE_CLASS_DESCRIPTIONS.get(cid, f"class_{cid[:8]}"))
    _write_class_descriptions(descriptions)
    FEATURE_CLASS_DESCRIPTIONS.update(descriptions)
    return descriptions


def _write_class_descriptions(descriptions: dict):
    """Write class descriptions dict to the JSON file."""
    try:
        with open(_CLASS_DESCRIPTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError:
        pass  # Read-only install (e.g. site-packages); in-memory update still applies


CONNECTED_CLASS_IDS = (
    SURFACES_IDS
    + VEG_IDS
    + [
        DRIVEABLE_ID,
        LEAF_OFF_VEG_ID,
    ]
)

CLASS_1111_YARD_DEBRIS = "405efaad-4b79-585d-88d8-43ae5fbb6f05"
CLASS_1054_POLE = "46f2f9ce-8c0f-50df-a9e0-4c2026dd3f95"

# Roof Condition / Malady Classes
CLASS_1050_TARP = "abb1f304-ce01-527b-b799-cbfd07551b2c"  # "temporary repair",
CLASS_1052_RUST = "526496bf-7344-5024-82d7-77ceb671feb4"  # "rust",
CLASS_1079_MISSING_SHINGLES = "dec855e2-ae6f-56b5-9cbb-f9967ff8ca12"  # "missing tiles or shingles",
CLASS_1139_DEBRIS = "8ab218a7-8173-5f1e-a5cb-bb2cd386a73e"  # "debris",
CLASS_1140_EXPOSED_DECK = "2905ba1c-6d96-58bc-9b1b-5911b3ead023"  # "exposed_deck",
CLASS_1051_PONDING = "f41e02b0-adc0-5b46-ac95-8c59aa9fe317"  # "ponding",
CLASS_1144_STAINING = "319f552f-f4b7-520d-9b16-c8abb394b043"
CLASS_1146_WORN_SHINGLES = "97a6f930-82ae-55f2-b856-635e2250af29"
CLASS_1147_EXPOSED_UNDERLAYMENT = "2322ca41-5d3d-5782-b2b7-1a2ffd0c4b78"
CLASS_1149_PATCHING = "8b30838b-af41-5d1d-bdbd-29e682fe3b00"
CLASS_1186_STRUCTURAL_DAMAGE = "c0224852-4310-57dd-95fe-42bff1c0a3f0"

# Roof Shapes
CLASS_1013_HIP = "ac0a5f75-d8aa-554c-8a43-cee9684ef9e9"
CLASS_1014_GABLE = "59c6e27e-6ef2-5b5c-90e7-31cfca78c0c2"
CLASS_1015_DUTCH_GABLE = "3719eb40-d6d1-5071-bbe6-379a551bb65f"
CLASS_1019_GAMBREL = "4bb630b9-f9eb-5f95-85b8-f0c6caf16e9b"
CLASS_1020_CONICAL = "89582082-e5b8-5853-bc94-3a0392cab98a"
CLASS_1173_PARAPET = "1234ea84-e334-5c58-88a9-6554be3dfc05"
CLASS_1174_MANSARD = "7eb3b1b6-0d75-5b1f-b41c-b14146ff0c54"
CLASS_1176_JERKINHEAD = "924afbab-aae6-5c26-92e8-9173e4320495"
CLASS_1178_QUONSET = "e92bc8a2-9fa3-5094-b3b6-2881d94642ab"
CLASS_1180_BOWSTRING_TRUSS = "09b925d2-df1d-599b-89f1-3ffd39df791e"

# Roof Materials
CLASS_1191_FLAT = "1ab60ef7-e770-5ab6-995e-124676b2be11"
CLASS_1007_TILE = "516fdfd5-0be9-59fe-b849-92faef8ef26e"
CLASS_1008_ASPHALT_SHINGLE = "4bbf8dbd-cc81-5773-961f-0121101422be"
CLASS_1009_METAL_PANEL = "4424186a-0b42-5608-a5a0-d4432695c260"
CLASS_1100_BALLASTED = "4558c4fb-3ddf-549d-b2d2-471384be23d1"
CLASS_1101_MOD_BIT = "87437e20-d9f5-57e1-8b87-4a9c81ec3b65"
CLASS_1103_TPO = "383930f1-d866-5aa3-9f97-553311f3162d"
CLASS_1104_EPDM = "64db6ea0-7248-53f5-b6a6-6ed733c5f9b8"
CLASS_1105_WOOD_SHAKE = "9fc4c92e-4405-573e-bce6-102b74ab89a3"
CLASS_1160_CLAY_TILE = "09ed6bf9-182a-5c79-ae59-f5531181d298"
CLASS_1163_SLATE = "cdc50dcc-e522-5361-8f02-4e30673311bb"
CLASS_1165_BUILT_UP = "3563c8f1-e81e-52c7-bd56-eaa937010403"
CLASS_1168_ROOF_COATING = "b2573072-b3a5-5f7c-973f-06b7649665ff"

ROOF_CHAR_IDS = [
    METAL_ROOF_ID,
    TILE_ROOF_ID,
    SHINGLE_ROOF_ID,
    CLASS_1191_FLAT,
    HIP_ROOF_ID,
    GABLE_ROOF_ID,
    DUTCH_GABLE_ROOF_ID,
    TURRET_ROOF_ID,
    TREE_OVERHANG_ID,
]
