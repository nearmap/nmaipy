import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import concurrent.futures
import logging


import numpy as np
import pandas as pd
import geopandas as gpd


s = requests.Session()

AI_COVERAGE = "ai"
STANDARD_COVERAGE = "standard"
FORBIDDEN_403 = 403

retries = Retry(total=20, backoff_factor=0.1, status_forcelist=[408, 429, 500, 502, 503, 504])
s.mount("https://", HTTPAdapter(max_retries=retries, pool_maxsize=100, pool_connections=100))


def get_payload(request_string):
    """
    Basic wrapper code to retrieve the JSON payload from the API, and raise an error if no response is given.
    Using urllib3, this also implements exponential backoff and retries for robustness.
    """
    response = s.get(
        request_string,
    )

    if response.ok:
        logging.info(f"Status Code: {response.status_code}")
        logging.info(f"Status Message: {response.reason}")
        payload = response.json()
        return payload
    elif response.status_code == FORBIDDEN_403:
        logging.info(f"Status Code: {response.status_code}")
        logging.info(f"Status Message: {response.reason}")
        payload = response.content
        logging.info(str(payload))
        return FORBIDDEN_403
    else:
        logging.error(f"Status Code: {response.status_code}")
        logging.error(f"Status Message: {response.reason}")
        logging.error(str(response))
        payload = response.content
        logging.error(str(payload))
        return response.status_code


def poly2coordstring(poly):
    """
    Turn a shapely polygon into the format required by the API for a query polygon.
    """
    coords = poly.boundary.coords[:]
    flat_coords = np.array(coords).flatten()
    coordstring = ",".join(flat_coords.astype(str))
    return coordstring


def get_surveys_from_point(lon, lat, since, until, apikey, coverage_type, limit=100):
    fields = "id,captureDate,resources"
    if coverage_type == STANDARD_COVERAGE:
        url = f"https://api.nearmap.com/coverage/v2/point/{lon},{lat}?fields={fields}&limit={limit}&resources=tiles:Vert,aifeatures,3d&apikey={apikey}"
    elif coverage_type == AI_COVERAGE:
        url = f"https://api.nearmap.com/ai/features/v4/coverage.json?point={lon},{lat}&limit={limit}&apikey={apikey}"
    else:
        raise ValueError(f"Unknown coverage type {coverage_type}")
    if since is not None:
        url += f"&since={since}"
    if until is not None:
        url += f"&until={until}"
    response = get_payload(url)
    if not isinstance(response, int):
        if coverage_type == STANDARD_COVERAGE:
            return std_coverage_response_to_dataframe(response), response
        elif coverage_type == AI_COVERAGE:
            return ai_coverage_response_to_dataframe(response), response
        else:
            raise ValueError(f"Unknown coverage type {coverage_type}")
    elif response == FORBIDDEN_403:
        logging.info(f"Unauthorised area request at {lat=}, {lon=}, {since=}, {until=} with code {response}")
        return None, None
    else:
        logging.error(f"Failed request at {lat=}, {lon=}, {since=}, {until=} with code {response}")
        return None, None


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
        df_coverage = df_coverage.drop(columns="classes")
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
    threads=20,
    coverage_type=STANDARD_COVERAGE,
    ):
    """
    Wrapper function to get coverage from a dataframe of points, using a thread pool.
    """
    jobs = []

    if since_col is None:
        since = None
    if until_col is None:
        until = None

    # Send each parcel to a thread worker
    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
        for i, row in df.iterrows():
            if since_col is not None:
                since = str(row[since_col])
            if until_col is not None:
                until = str(row[until_col])
            jobs.append(
                executor.submit(
                    get_surveys_from_point, row[longitude_col], row[latitude_col], since, until, apikey, coverage_type
                )
            )

    results = []
    for job in jobs:
        df_job, _ = job.result()
        results.append(pd.DataFrame(df_job))
    return results
