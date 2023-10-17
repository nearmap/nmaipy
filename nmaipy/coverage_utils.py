import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import concurrent.futures
import logging


import numpy as np
import pandas as pd
import geopandas as gpd


s = requests.Session()

retries = Retry(total=20, backoff_factor=0.1, status_forcelist=[408, 429, 500, 502, 503])
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
    elif response.status_code == 403:
        logging.info(f"Status Code: {response.status_code}")
        logging.info(f"Status Message: {response.reason}")
        payload = response.content
        logging.info(str(payload))
        return None
    else:
        logging.error(f"Status Code: {response.status_code}")
        logging.error(f"Status Message: {response.reason}")
        logging.error(str(response))
        payload = response.content
        logging.error(str(payload))
        return None


def poly2coordstring(poly):
    """
    Turn a shapely polygon into the format required by the API for a query polygon.
    """
    coords = poly.boundary.coords[:]
    flat_coords = np.array(coords).flatten()
    coordstring = ",".join(flat_coords.astype(str))
    return coordstring


def get_surveys_from_point(lat, lon, since, until, apikey, limit=100):
    fields = "id,captureDate,resources"
    url = f"https://api.nearmap.com/coverage/v2/point/{lon},{lat}?fields={fields}&limit={limit}&apikey={apikey}"
    if since is not None:
        url += f"&since={since}"
    if until is not None:
        url += f"&until={until}"
    response = get_payload(url)
    if response is not None:
        return survey_response_to_dataframe(response["surveys"])
    else:
        logging.error(f"Failed request at {lat=}, {lon=}, {since=}, {until=}")


def survey_response_to_dataframe(survey_response):
    df_survey = pd.DataFrame(survey_response)
    if len(df_survey) == 0:
        return df_survey
    else:
        for resource_type in ["tiles", "3d", "aifeatures"]:
            df_survey[resource_type] = df_survey["resources"].apply(lambda d: resource_type in d)
        return df_survey


def threaded_get_coverage_from_point_results(
    df, apikey, longitude_col="longitude", latitude_col="latitude", since_col="since", until_col="until", threads=20
):
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
                executor.submit(get_surveys_from_point, row[latitude_col], row[longitude_col], since, until, apikey)
            )

    results = []
    for job in jobs:
        results.append(pd.DataFrame(job.result()))
    return results