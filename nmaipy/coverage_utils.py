import concurrent.futures
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import shape
from shapely.ops import unary_union
from tqdm import tqdm
from urllib3.util.retry import Retry

from nmaipy import (  # noqa: F401 — import side effect: installs APIKeyFilter on the "nmaipy" / urllib3 / requests loggers.
    api_common,
    log,
)

# Use the project-level nmaipy logger so emissions are scrubbed by APIKeyFilter.
# A child logger via logging.getLogger(__name__) would propagate to nmaipy's
# handlers but bypass its filter — empirically verified to leak apikey=... values
# into log output. The api_common import above is what installs the filter; without
# it, even routing through log.get_logger() leaks (the filter is set up as a module
# side effect in api_common).
logger = log.get_logger()

s = requests.Session()

AI_COVERAGE = "ai"
STANDARD_COVERAGE = "standard"
FORBIDDEN_403 = 403
DATETIMEDTYPE = "datetime64[ns]"  # Datetime type for pandas

# (connect, read) timeout in seconds. Without this a hung connection blocks a
# worker thread indefinitely — a real hazard under heavy threaded per-point load.
DEFAULT_TIMEOUT = (10, 60)

retries = Retry(total=20, backoff_factor=0.1, status_forcelist=[408, 429, 500, 502, 503, 504])
s.mount("https://", HTTPAdapter(max_retries=retries, pool_maxsize=100, pool_connections=100))

# ---------------------------------------------------------------------------
# Post-catastrophe event discovery (Coverage API)
# ---------------------------------------------------------------------------
# Given a lat/lon, resolve the latest ImpactResponse event id + footprint so the operator
# can then subdivide the boundary into AOIs and run the Damage Conflation exporter with the
# discovered event id. Deployed-API facts (differ from the public doc): the event id is on
# the `postCatEventId` survey tag (NOT `eventId`); the point query takes `{lon},{lat}` in the
# URL path; and the event footprint is fetched in one shot via the survey tag filter
# `/coverage/v2/surveys?include=postCatEventId:<id>` (the "Filter Surveys" `include=<type>:<name>`
# grammar — `filter=`/`tags=` are not accepted), then the returned surveys' boundaries are unioned.
COVERAGE_V2_URL = "https://api.nearmap.com/coverage/v2"
POST_CAT_EVENT_ID_TAG = "postCatEventId"
POST_CAT_EVENT_NAME_TAG = "postCatEventName"
POST_CAT_EVENT_DATE_TAG = "postCatEventDate"
POST_CAT_EVENT_TYPE_TAG = "postCatEventType"
# The /coverage/v2/surveys endpoint paginates via offset/limit and reports `total`; page
# through all of an event's surveys so a large event (e.g. thousands of post-cat captures)
# isn't silently truncated at a single page.
_COVERAGE_PAGE_SIZE = 1000
_MAX_COVERAGE_PAGES = 100  # safety bound (~100k surveys) against a missing/inconsistent `total`


def get_payload(request_string, apikey=None, timeout=DEFAULT_TIMEOUT, bearer_token=None):
    """
    Basic wrapper code to retrieve the JSON payload from the API, and raise an error if no response is given.
    Using urllib3, this also implements exponential backoff and retries for robustness.

    The apikey is sent via requests' ``params`` (merged into the query string by requests) rather
    than baked into ``request_string``, so the key never appears in the URL variable that flows
    through this module or into any log line here. When ``bearer_token`` is given it authenticates
    via an ``Authorization: Bearer`` header instead, and the apikey param is omitted.

    Per-request logging is at debug level: at 10k+ points an info line per call
    floods the logs. Failures are logged at warning/error.
    """
    if bearer_token:
        params = None
        headers = {"Authorization": f"Bearer {bearer_token}"}
    else:
        params = {"apikey": apikey} if apikey else None
        headers = None
    response = s.get(request_string, params=params, headers=headers, timeout=timeout)

    # response.content is server-controlled. The Nearmap coverage API doesn't
    # echo apikeys in response bodies, and the nmaipy-logger APIKeyFilter would
    # scrub them if it did — so this is safe today. Worth knowing if either
    # assumption changes.
    if response.ok:
        logger.debug(f"Status Code: {response.status_code} ({response.reason})")
        return response.json()
    elif response.status_code == FORBIDDEN_403:
        logger.debug(f"Status Code: {response.status_code} ({response.reason}) — {response.content!r}")
        return FORBIDDEN_403
    else:
        logger.error(f"Status Code: {response.status_code} ({response.reason}) — {response.content!r}")
        return response.status_code


def poly2coordstring(poly):
    """
    Turn a shapely polygon into the format required by the API for a query polygon.
    """
    coords = poly.boundary.coords[:]
    flat_coords = np.array(coords).flatten()
    coordstring = ",".join(flat_coords.astype(str))
    return coordstring


def _resolve_apikey(apikey, bearer_token=None):
    """Return the given apikey, or fall back to the API_KEY env var.

    In bearer mode no apikey is needed, so return None. Raise only when neither an apikey
    (arg or env) nor a bearer token is available.
    """
    if bearer_token:
        return None
    key = apikey or os.environ.get("API_KEY")
    if not key:
        raise ValueError("No API key or bearer token provided and the API_KEY environment variable is not set")
    return key


def _post_cat_events_at_point(lat, lon, since=None, until=None, apikey=None, bearer_token=None):
    """Return ``{event_id: {event_name, event_date, event_type}}`` for post-cat events whose
    surveys cover ``(lat, lon)``. Empty dict if none. Raises ValueError on a failed request."""
    apikey = _resolve_apikey(apikey, bearer_token)
    # Endpoint takes the point as `{lon},{lat}` in the path (see get_surveys_from_point).
    url = f"{COVERAGE_V2_URL}/point/{lon},{lat}?fields=id,captureDate,tags&limit=100"
    if since is not None:
        url += f"&since={since}"
    if until is not None:
        url += f"&until={until}"
    response = get_payload(url, apikey=apikey, bearer_token=bearer_token)
    if isinstance(response, int):
        raise ValueError(f"Coverage API request failed at lat={lat}, lon={lon} (status {response})")

    events = {}
    for survey in response.get("surveys", []):
        tags = survey.get("tags") or {}
        event_id = tags.get(POST_CAT_EVENT_ID_TAG)
        if event_id:
            events.setdefault(
                event_id,
                {
                    "event_name": tags.get(POST_CAT_EVENT_NAME_TAG),
                    "event_date": tags.get(POST_CAT_EVENT_DATE_TAG),
                    "event_type": tags.get(POST_CAT_EVENT_TYPE_TAG),
                },
            )
    return events


def _select_latest_event(events):
    """Pick ``(event_id, info)`` with the newest ``event_date``; warn when >1 event is present."""
    ordered = sorted(events.items(), key=lambda kv: kv[1]["event_date"] or "", reverse=True)
    if len(ordered) > 1:
        others = ", ".join(f"{eid} ({i['event_name']}, {i['event_date']})" for eid, i in ordered[1:])
        top_id, top = ordered[0]
        logger.warning(
            f"{len(ordered)} distinct post-cat events cover this point; using the latest: "
            f"{top_id} ({top['event_name']}, {top['event_date']}). Others: {others}. "
            f"Pass since/until to target a specific event."
        )
    return ordered[0]


def latest_event_id_at_point(lat, lon, since=None, until=None, apikey=None, bearer_token=None):
    """Discover the latest ImpactResponse catastrophe event id covering ``(lat, lon)``.

    Queries the Coverage API point endpoint and returns the ``postCatEventId`` of the event
    with the most recent ``postCatEventDate``. Logs a warning listing the rest when more than
    one distinct event covers the point.

    Args:
        lat, lon: query point in WGS84 degrees.
        since, until: optional ``YYYY-MM-DD`` window to narrow the search.
        apikey: Nearmap API key; defaults to the ``API_KEY`` environment variable.
        bearer_token: short-lived identity JWT; when set, auth uses a Bearer header and no apikey.

    Returns:
        The event id (string UUID).

    Raises:
        ValueError: if no post-cat event is found at the point (in the window).
    """
    events = _post_cat_events_at_point(lat, lon, since, until, apikey, bearer_token)
    if not events:
        window = f" in {since}..{until}" if (since or until) else ""
        raise ValueError(f"No post-catastrophe event found at lat={lat}, lon={lon}{window}")
    return _select_latest_event(events)[0]


def event_boundary(event_id, since=None, until=None, apikey=None, bearer_token=None):
    """Assemble the footprint polygon for ``event_id`` as a shapely (Multi)Polygon (EPSG:4326).

    Uses the Coverage API's documented tag filter to fetch **every** survey tagged with this
    event — ``/coverage/v2/surveys?include=postCatEventId:<event_id>`` — then unions their
    vertical-imagery boundaries. This is a single, event-wide query with no spatial search.
    Pass ``since``/``until`` to get the footprint as of a point in time (it grows as captures
    land).

    Args:
        event_id: the ``postCatEventId`` to assemble a boundary for.
        since, until: optional ``YYYY-MM-DD`` window to bound the captures considered.
        apikey: Nearmap API key; defaults to the ``API_KEY`` environment variable.

    Returns:
        A shapely ``Polygon``/``MultiPolygon`` in EPSG:4326.

    Raises:
        ValueError: on a failed request, or if no surveys for ``event_id`` are found.
    """
    apikey = _resolve_apikey(apikey, bearer_token)
    window = ""
    if since is not None:
        window += f"&since={since}"
    if until is not None:
        window += f"&until={until}"

    boundaries = []
    offset = 0
    for _ in range(_MAX_COVERAGE_PAGES):
        # The `include=<type>:<name>` tag filter is what the "Filter Surveys" doc means; it
        # returns the event's surveys globally (the /surveys endpoint requires this non-date
        # filter param). The endpoint paginates via offset/limit and reports `total`.
        url = (
            f"{COVERAGE_V2_URL}/surveys"
            f"?include={POST_CAT_EVENT_ID_TAG}:{event_id}"
            "&resources=tiles:Vert&fields=id,captureDate,resources,resources:boundary,tags"
            f"&limit={_COVERAGE_PAGE_SIZE}&offset={offset}{window}"
        )
        response = get_payload(url, apikey=apikey, bearer_token=bearer_token)
        if isinstance(response, int):
            raise ValueError(f"Coverage API boundary request failed for event {event_id} (status {response})")

        surveys = response.get("surveys", [])
        for survey in surveys:
            # The server filters by the include tag; re-check client-side defensively.
            if (survey.get("tags") or {}).get(POST_CAT_EVENT_ID_TAG) != event_id:
                continue
            for tile in survey.get("resources", {}).get("tiles", []) or []:
                geom = tile.get("boundary")
                if geom:
                    boundaries.append(shape(geom))

        offset += len(surveys)
        total = response.get("total")
        if not surveys:
            break
        if total is not None:
            if offset >= total:
                break
        elif len(surveys) < _COVERAGE_PAGE_SIZE:  # no `total` reported — stop on a short page
            break
    else:
        # Loop exhausted _MAX_COVERAGE_PAGES without terminating — surface rather than silently truncate.
        raise ValueError(
            f"Event {event_id} boundary paging exceeded {_MAX_COVERAGE_PAGES} pages "
            f"({len(boundaries)} boundaries so far) — aborting rather than truncating"
        )

    if not boundaries:
        win = f" in {since}..{until}" if (since or until) else ""
        raise ValueError(f"No surveys found for event {event_id}{win}")
    return unary_union(boundaries)


def discover_event(lat, lon, since=None, until=None, apikey=None, bearer_token=None):
    """Discover the latest event id AND its boundary at a point, in one pair of queries.

    The entry point for the "hit a point, get the event id + footprint" workflow: resolves the
    latest ``postCatEventId`` covering ``(lat, lon)``, then fetches that event's full footprint
    via the tag filter. The caller then subdivides ``boundary`` into AOIs and runs the Damage
    Conflation exporter with ``event_id``.

    Args:
        lat, lon: query point in WGS84 degrees.
        since, until: optional ``YYYY-MM-DD`` window (applied to discovery and the boundary).
        apikey: Nearmap API key; defaults to the ``API_KEY`` environment variable.

    Returns:
        ``(event_id, boundary)`` — a string UUID and a shapely (Multi)Polygon in EPSG:4326.

    Raises:
        ValueError: if no post-cat event is found at the point.
    """
    events = _post_cat_events_at_point(lat, lon, since, until, apikey, bearer_token)
    if not events:
        window = f" in {since}..{until}" if (since or until) else ""
        raise ValueError(f"No post-catastrophe event found at lat={lat}, lon={lon}{window}")
    event_id, _info = _select_latest_event(events)
    boundary = event_boundary(event_id, since=since, until=until, apikey=apikey, bearer_token=bearer_token)
    return event_id, boundary


def get_surveys_from_point(
    lon,
    lat,
    since,
    until,
    apikey,
    coverage_type,
    include_disaster=False,
    has_3d=False,
    prerelease=False,
    limit=100,
    timeout=DEFAULT_TIMEOUT,
    bearer_token=None,
):
    fields = "id,captureDate,resources,tags"
    if coverage_type == STANDARD_COVERAGE:
        url = f"https://api.nearmap.com/coverage/v2/point/{lon},{lat}?fields={fields}&limit={limit}&resources=tiles:Vert,aifeatures,3d"
        if include_disaster:
            url += f"&include=disaster"
    elif coverage_type == AI_COVERAGE:
        url = f"https://api.nearmap.com/ai/features/v4/coverage.json?point={lon},{lat}&limit={limit}"
        if has_3d:
            url += "&3dCoverage=true"
    else:
        raise ValueError(f"Unknown coverage type {coverage_type}")
    if since is not None:
        url += f"&since={since}"
    if until is not None:
        url += f"&until={until}"
    if prerelease:
        url += "&prerelease=true"
    response = get_payload(url, apikey=apikey, timeout=timeout, bearer_token=bearer_token)
    if not isinstance(response, int):
        if coverage_type == STANDARD_COVERAGE:
            return std_coverage_response_to_dataframe(response), response
        elif coverage_type == AI_COVERAGE:
            return ai_coverage_response_to_dataframe(response), response
        else:
            raise ValueError(f"Unknown coverage type {coverage_type}")
    elif response == FORBIDDEN_403:
        logger.debug(f"Unauthorised area request at {lat=}, {lon=}, {since=}, {until=} with code {response}")
        return None, None
    else:
        logger.error(f"Failed request at {lat=}, {lon=}, {since=}, {until=} with code {response}")
        return None, None


def get_survey_resource_id_from_survey_id_query(resources):
    """
    Get the survey resource id from the resources list, after being given the "resources" field from the survey_id query coverage/v2/surveys/{survey_id} API.
    """
    if len(resources) == 1:
        return resources[0]["id"]
    else:
        raise Exception("More than one resource returned from survey_id query")


def get_survey_resource_id_from_standard_coverage(resources):
    """
    Get the survey resource id from the resources list. This is the id that can be used with the AI Feature API to get an exact match (rather than since/until dates).
    """
    if "aifeatures" in resources:
        return resources["aifeatures"][-1]["id"]
    else:
        return None


def std_coverage_response_to_dataframe(survey_response):
    """
    Convert the JSON response from the standard coverage API into a pandas dataframe.
    """
    df_survey = pd.DataFrame(survey_response["surveys"])
    if len(df_survey) == 0:
        return df_survey
    else:
        for resource_type in ["tiles", "3d", "aifeatures"]:
            df_survey[resource_type] = df_survey["resources"].apply(lambda d: resource_type in d)
        return df_survey


def ai_coverage_response_to_dataframe(response):
    """
    Convert the JSON response from the AI coverage API into a pandas dataframe.
    """
    if response["results"] is not None:
        df_coverage = pd.DataFrame(response["results"])
        # df_coverage = df_coverage.drop(columns="classes")
        return df_coverage
    else:
        return None


def threaded_get_coverage_from_point_results(
    df,
    apikey,
    longitude_col="longitude",
    latitude_col="latitude",
    since_col="since",
    until_col="until",
    since=None,
    until=None,
    threads=20,
    coverage_type=STANDARD_COVERAGE,
    include_disaster=False,
    has_3d=False,
    prerelease=False,
    limit=100,
    timeout=DEFAULT_TIMEOUT,
    bearer_token=None,
):
    """
    Get coverage for a dataframe of points using a thread pool.

    The query window is resolved per row: a present ``since_col`` / ``until_col``
    column overrides the global ``since`` / ``until`` argument. When the column is
    absent (or ``*_col`` is ``None``), the global value is used for every row — so
    callers can pass a single window without synthesising per-row columns.
    """
    df = df.copy()

    # A column overrides the global window only when it actually exists.
    has_since_col = since_col is not None and since_col in df.columns
    has_until_col = until_col is not None and until_col in df.columns
    # Normalise datetime columns to "yyyy-mm-dd" strings (the API wants strings).
    if has_since_col and df[since_col].dtype == DATETIMEDTYPE:
        df[since_col] = df[since_col].dt.strftime("%Y-%m-%d")
    if has_until_col and df[until_col].dtype == DATETIMEDTYPE:
        df[until_col] = df[until_col].dt.strftime("%Y-%m-%d")

    jobs = []
    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
        for _, row in df.iterrows():
            row_since = str(row[since_col]) if has_since_col else since
            row_until = str(row[until_col]) if has_until_col else until
            jobs.append(
                executor.submit(
                    get_surveys_from_point,
                    row[longitude_col],
                    row[latitude_col],
                    row_since,
                    row_until,
                    apikey,
                    coverage_type,
                    include_disaster,
                    has_3d,
                    prerelease,
                    limit,
                    timeout,
                    bearer_token,
                )
            )

    return [pd.DataFrame(job.result()[0]) for job in jobs]


def get_coverage_from_points(
    df_points,
    api_key,
    coverage_type="standard",
    chunk_size=10000,
    threads=20,
    coverage_chunk_cache_dir="coverage_chunks",
    id_col=None,
    since=None,
    until=None,
    include_disaster=False,
    has_3d=False,
    prerelease=False,
    limit=100,
    timeout=DEFAULT_TIMEOUT,
    bearer_token=None,
):
    """
    Given a GeoDataFrame of points, get a full history of all surveys that intersect with each point from the coverage API,
    and whether the survey has AI and/or 3D resources attached. This doesn't tell us what generation of AI data is available -
    that will require a subsequent run against the AI Feature API coverage endpoint to determine versions (or just pulling the
    data and ignoring version).

    Parameters:
    -----------
    df_points : GeoDataFrame
        A GeoDataFrame of points to check for coverage of imagery, 3D and AI.
    api_key : str
        The Nearmap API key to use for authentication.
    coverage_type : str, optional
        The type of coverage to retrieve. One of "standard" or "ai". Default "standard".
    chunk_size : int, optional
        The number of points to process in each chunk. Default is 10000.
    threads : int, optional
        The number of threads to use for making API calls. Default is 20.
    coverage_chunk_cache_dir : str, optional
        The directory to cache coverage chunks. Default is "coverage_chunks".
    id_col : str
        The name of the column in `df_points` that contains the unique identifier
        for each point. Required — must NOT be `"id"` (collides with each survey
        response's own `id` field).
    since, until : str, optional
        Global date window (YYYY-MM-DD) applied to every point. Overridden per-row by
        `since` / `until` columns when those are present in `df_points`.

    Returns:
    --------
    df_coverage : DataFrame
        A DataFrame containing the coverage data for each point.
    """
    if id_col is None:
        raise ValueError(
            "id_col is required. Pass the name of the column in df_points that "
            "uniquely identifies each point (e.g. 'aoi_id'). It must NOT be 'id', "
            "which collides with each survey response's own 'id' field."
        )
    if id_col == "id":
        raise ValueError(
            "id_col='id' collides with the per-survey 'id' field returned by the "
            "coverage API. Use a distinct column name (e.g. 'aoi_id')."
        )
    df_coverage = []
    df_coverage_empty = None
    coverage_chunk_cache_dir = Path(coverage_chunk_cache_dir)
    coverage_chunk_cache_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, len(df_points), chunk_size)):
        f = coverage_chunk_cache_dir / f"coverage_chunk_{i}-{i+chunk_size}.parquet"
        if not f.exists():
            df_point_chunk = df_points.iloc[i : i + chunk_size, :]
            logger.debug(f"Pulling chunk from API for {f}.")
            # Multi-threaded pulls are ok - the API is designed to cope fine with 10-20 threads running in parallel pulling requests.
            c = threaded_get_coverage_from_point_results(
                df_point_chunk,
                since_col="since",
                until_col="until",
                since=since,
                until=until,
                apikey=api_key,
                threads=threads,
                coverage_type=coverage_type,
                include_disaster=include_disaster,
                has_3d=has_3d,
                prerelease=prerelease,
                limit=limit,
                timeout=timeout,
                bearer_token=bearer_token,
            )
            c_with_idx = []
            for j in range(len(c)):
                row_id = df_point_chunk[id_col].iloc[j]
                c_tmp = c[j].copy()
                if len(c_tmp) > 0:
                    c_tmp[id_col] = row_id
                    c_with_idx.append(c_tmp)
            if len(c_with_idx) > 0:
                c = pd.concat(c_with_idx)
                if coverage_type == STANDARD_COVERAGE:
                    c["survey_resource_id"] = c.resources.apply(get_survey_resource_id_from_standard_coverage)
                    c = c.rename(columns={"id": "survey_id"})
                elif coverage_type == AI_COVERAGE:
                    c = c.rename(columns={"surveyId": "survey_id"})
                    c = c.rename(columns={"id": "survey_resource_id"})
                if (
                    df_coverage_empty is None
                ):  # Set an empty dataframe with the right columns for writing dummy parquet cache files
                    df_coverage_empty = pd.DataFrame([], columns=c.columns).astype(c.dtypes)
            else:
                c = df_coverage_empty
            if c is not None:
                c.to_parquet(f)

        else:
            logger.debug(f"Reading chunk from parquet for {f}.")
            c = pd.read_parquet(f)

        if c is not None:
            if len(c) > 0:
                if coverage_type == STANDARD_COVERAGE:
                    cols = [
                        id_col,
                        "captureDate",
                        "survey_id",
                        "survey_resource_id",
                        "tiles",
                        "aifeatures",
                        "3d",
                        "tags",
                    ]
                elif coverage_type == AI_COVERAGE:
                    cols = [
                        id_col,
                        "captureDate",
                        "survey_id",
                        "survey_resource_id",
                        "systemVersion",
                        "postcat",
                        "perspective",
                    ]
                # reindex (not .loc[…]) so a survey response missing an optional
                # field (e.g. tags / 3d) yields a NaN column instead of KeyError.
                c = c.reindex(columns=cols).set_index(id_col)
                c["captureDate"] = pd.to_datetime(c["captureDate"])
                df_coverage.append(c)
    if len(df_coverage) > 0:
        df_coverage = pd.concat(df_coverage)
        return df_coverage
    else:
        return None


def threaded_get_coverage_from_survey_ids(
    df,
    apikey,
    survey_id_col="survey_id",
    threads=20,
    prerelease=False,
    limit=100,
    bearer_token=None,
):
    """
    Wrapper function to get coverage from a dataframe with survey_id's in it, using a thread pool.
    """
    jobs = []

    df = df.copy()

    # Send each parcel to a thread worker
    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
        # Set since_col/until_col to string "yyyy-mm-dd" format if datetimes

        for i, row in df.iterrows():
            jobs.append(
                executor.submit(
                    get_surveys_from_id,
                    row[survey_id_col],
                    apikey,
                    limit,
                    bearer_token=bearer_token,
                )
            )

    results = []
    for job in jobs:
        df_job, _ = job.result()
        results.append(pd.DataFrame(df_job))
    return results


def get_surveys_from_id(survey_id, apikey, limit=100, timeout=DEFAULT_TIMEOUT, bearer_token=None):
    # NOTE: this references `id_check_response_to_dataframe`, which is not defined
    # anywhere in the package — the survey-ids path is pre-existing dead code with
    # no callers. Left in place (out of scope for this cleanup); flagged so a
    # future reviver knows the response parser still needs implementing.
    fields = "id,captureDate,resources"
    url = f"https://api.nearmap.com/coverage/v2/surveys/{survey_id}?fields={fields}&limit={limit}&resources=tiles:Vert,aifeatures,3d"
    response = get_payload(url, apikey=apikey, timeout=timeout, bearer_token=bearer_token)
    if not isinstance(response, int):
        return id_check_response_to_dataframe(response), response  # noqa: F821 (see NOTE above)
    elif response == FORBIDDEN_403:
        logger.debug(f"Unauthorised area request at {survey_id=} with code {response}")
        return None, None
    else:
        logger.error(f"Failed request at {survey_id=} with code {response}")
        return None, None


def get_coverage_from_survey_ids(
    df,
    api_key,
    chunk_size=10000,
    threads=20,
    coverage_chunk_cache_dir="coverage_chunks",
    id_col="id",
    limit=100,
    bearer_token=None,
):
    """
    Given a GeoDataFrame with survey_ids as a column, get a set of all survey resource IDs that are attached to those survey_ids (such as aifeatures, tiles)

    Parameters:
    -----------
    df : GeoDataFrame
        A GeoDataFrame of points to check for coverage of imagery, 3D and AI.
    api_key : str
        The Nearmap API key to use for authentication.
    chunk_size : int, optional
        The number of points to process in each chunk. Default is 10000.
    threads : int, optional
        The number of threads to use for making API calls. Default is 20.
    coverage_chunk_cache_dir : str, optional
        The directory to cache coverage chunks. Default is "coverage_chunks".
    id_col : str, optional
        The name of the column in `df_points` that contains the unique identifier for each point.

    Returns:
    --------
    df_coverage : DataFrame
        A DataFrame containing the coverage data for each survey_id.
    """
    df_coverage = []
    df_coverage_empty = None
    coverage_chunk_cache_dir = Path(coverage_chunk_cache_dir)
    coverage_chunk_cache_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, len(df), chunk_size)):
        f = coverage_chunk_cache_dir / f"coverage_chunk_{i}-{i+chunk_size}.parquet"
        if not f.exists():
            df_point_chunk = df.iloc[i : i + chunk_size, :]
            logger.debug(f"Pulling chunk from API for {f}.")
            # Multi-threaded pulls are ok - the API is designed to cope fine with 10-20 threads running in parallel pulling requests.
            c = threaded_get_coverage_from_survey_ids(
                df_point_chunk,
                survey_id_col="survey_id",
                apikey=api_key,
                threads=threads,
                limit=limit,
                bearer_token=bearer_token,
            )
            c_with_idx = []
            for j in range(len(c)):
                row_id = df_point_chunk.iloc[j].name
                c_tmp = c[j].copy()
                if len(c_tmp) > 0:
                    c_tmp[id_col] = row_id
                    c_with_idx.append(c_tmp)
            if len(c_with_idx) > 0:
                c = pd.concat(c_with_idx)
                c["survey_resource_id"] = c["resources"].apply(get_survey_resource_id_from_survey_id_query)
                c = c.rename(columns={"id": "survey_id"})
                # Set an empty dataframe with the right columns for dummy cache files.
                # NB: must be nested here — previously this else clobbered a real
                # `c` with the empty frame on every chunk after the first.
                if df_coverage_empty is None:
                    df_coverage_empty = pd.DataFrame([], columns=c.columns).astype(c.dtypes)
            else:
                c = df_coverage_empty
            if c is not None:
                c.to_parquet(f)
        else:
            logger.debug(f"Reading chunk from parquet for {f}.")
            c = pd.read_parquet(f)

        if c is not None:
            if len(c) > 0:
                cols = [id_col, "captureDate", "survey_id", "survey_resource_id", "tiles", "aifeatures", "3d"]
                c = c.reindex(columns=cols).set_index(id_col)
                c["captureDate"] = pd.to_datetime(c["captureDate"])
                df_coverage.append(c)
    if len(df_coverage) > 0:
        df_coverage = pd.concat(df_coverage)
        return df_coverage
    else:
        return None
