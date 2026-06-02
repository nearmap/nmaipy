import concurrent.futures
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
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


def get_payload(request_string, timeout=DEFAULT_TIMEOUT):
    """
    Basic wrapper code to retrieve the JSON payload from the API, and raise an error if no response is given.
    Using urllib3, this also implements exponential backoff and retries for robustness.

    Per-request logging is at debug level: at 10k+ points an info line per call
    floods the logs. Failures are logged at warning/error.
    """
    response = s.get(request_string, timeout=timeout)

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
    url += f"&apikey={apikey}"
    response = get_payload(url, timeout=timeout)
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
                )
            )

    results = []
    for job in jobs:
        df_job, _ = job.result()
        results.append(pd.DataFrame(df_job))
    return results


def get_surveys_from_id(survey_id, apikey, limit=100, timeout=DEFAULT_TIMEOUT):
    # NOTE: this references `id_check_response_to_dataframe`, which is not defined
    # anywhere in the package — the survey-ids path is pre-existing dead code with
    # no callers. Left in place (out of scope for this cleanup); flagged so a
    # future reviver knows the response parser still needs implementing.
    fields = "id,captureDate,resources"
    url = f"https://api.nearmap.com/coverage/v2/surveys/{survey_id}?fields={fields}&limit={limit}&resources=tiles:Vert,aifeatures,3d"
    url += f"&apikey={apikey}"
    response = get_payload(url, timeout=timeout)
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
