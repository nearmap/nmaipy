"""Tests for Bearer-token (short-lived identity JWT) authentication in the API clients.

Bearer mode authenticates via an ``Authorization: Bearer <jwt>`` header instead of the
``?apikey=`` query parameter. These tests assert:
  - the credential guard accepts a bearer token with no api key,
  - URL builders omit the apikey param (and stay well-formed) in bearer mode,
  - the session carries the Authorization header,
  - a bearer token is scrubbed from logs/errors.
"""

import logging

import pytest
import responses
from shapely.geometry import box

from nmaipy import coverage_utils
from nmaipy.api_common import APIKeyFilter, clean_api_key_from_string
from nmaipy.damage_conflation_api import DamageConflationApi
from nmaipy.feature_api import FeatureApi
from nmaipy.roof_age_api import RoofAgeApi

_JWT = "eyJhbGciOiJSUzI1NiJ9.payload.signature"
_EVENT_ID = "2f510853-5d55-50f4-9102-2c02de08190e"


def test_bearer_token_needs_no_api_key(monkeypatch):
    """A bearer token alone must satisfy the credential guard (no API_KEY set)."""
    monkeypatch.delenv("API_KEY", raising=False)
    api = FeatureApi(api_key=None, bearer_token=_JWT)
    assert api.bearer_token == _JWT
    assert api.api_key == ""  # never None, so URL builders never interpolate "None"


def test_bearer_mode_discards_env_api_key(monkeypatch):
    """Bearer mode must zero the api key even when API_KEY is set in the environment,
    so a code path that misses the bearer check fails loudly instead of silently
    falling back to the long-lived key."""
    monkeypatch.setenv("API_KEY", "env-key-that-must-not-be-used")
    api = FeatureApi(bearer_token=_JWT)
    assert api.api_key == ""


def test_no_credential_still_raises(monkeypatch):
    """Neither api key nor bearer token (nor cache) must still raise."""
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API KEY or bearer token"):
        FeatureApi(api_key=None)


def test_post_url_omits_apikey_in_bearer_mode():
    """_create_post_request must not carry ?apikey= and must be well-formed."""
    api = FeatureApi(api_key=None, bearer_token=_JWT)
    url, _body, _exact = api._create_post_request(api.FEATURES_URL, geometry=box(0, 0, 1, 1))
    assert "apikey=" not in url
    assert "?&" not in url
    assert "&&" not in url
    # parcelMode is always appended; it must be the first query param, opened with "?".
    assert "?parcelMode=" in url


def test_post_url_includes_apikey_in_key_mode():
    """Control: key mode still uses ?apikey=."""
    api = FeatureApi(api_key="dummy")
    url, _body, _exact = api._create_post_request(api.FEATURES_URL, geometry=box(0, 0, 1, 1))
    assert "apikey=dummy" in url


def test_strip_empty_query():
    """Both collapse branches: "?&" with params, and the lone trailing "?" with none."""
    assert FeatureApi._strip_empty_query("https://x/f.json?&a=1&b=2") == "https://x/f.json?a=1&b=2"
    assert FeatureApi._strip_empty_query("https://x/f.json?") == "https://x/f.json"
    assert FeatureApi._strip_empty_query("https://x/f.json?a=1") == "https://x/f.json?a=1"


def test_both_credentials_warns_and_uses_bearer(caplog):
    """An explicit api_key alongside a bearer token warns and is discarded."""
    # The nmaipy logger has propagate=False, so attach caplog's handler directly.
    nmaipy_logger = logging.getLogger("nmaipy")
    nmaipy_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="nmaipy"):
            api = FeatureApi(api_key="dummy", bearer_token=_JWT)
    finally:
        nmaipy_logger.removeHandler(caplog.handler)
    assert api.bearer_token == _JWT
    assert api.api_key == ""
    assert any("api_key is ignored" in r.getMessage() for r in caplog.records)


def test_clean_api_key_scrubs_raw_bearer_token():
    """A raw token without the "Bearer " prefix is still scrubbed by _clean_api_key."""
    feature_api = FeatureApi(bearer_token=_JWT)
    assert _JWT not in feature_api._clean_api_key(f"request with {_JWT} leaked")
    roof_age_api = RoofAgeApi(bearer_token=_JWT)  # exercises the BaseApiClient implementation
    assert _JWT not in roof_age_api._clean_api_key(f"request with {_JWT} leaked")


@responses.activate
def test_packs_bearer_sends_header_and_no_apikey():
    """get_packs in bearer mode: Authorization header present, no apikey query param."""
    responses.add(
        responses.GET,
        api_url := "https://api.nearmap.com/ai/features/v4/bulk/packs.json",
        json={"packs": []},
        status=200,
    )
    api = FeatureApi(api_key=None, bearer_token=_JWT)
    api.get_packs()
    req = responses.calls[0].request
    assert req.headers["Authorization"] == f"Bearer {_JWT}"
    assert "apikey=" not in req.url
    assert req.url.startswith(api_url)


@responses.activate
def test_packs_key_mode_no_auth_header():
    """Control: key mode sends apikey in the URL and no Authorization header."""
    responses.add(
        responses.GET, "https://api.nearmap.com/ai/features/v4/bulk/packs.json", json={"packs": []}, status=200
    )
    api = FeatureApi(api_key="dummy")
    api.get_packs()
    req = responses.calls[0].request
    assert "apikey=dummy" in req.url
    assert "Authorization" not in req.headers


def test_bearer_token_scrubbed_by_clean_helper():
    """A bearer token must be redacted from strings destined for logs/errors."""
    dirty = f"GET /packs Authorization: Bearer {_JWT} failed"
    cleaned = clean_api_key_from_string(dirty)
    assert _JWT not in cleaned
    assert "Bearer REMOVED" in cleaned


def test_bearer_token_scrubbed_by_log_filter():
    """APIKeyFilter must redact a bearer token in a log record."""
    f = APIKeyFilter()
    rec = logging.LogRecord("n", logging.INFO, __file__, 1, f"Authorization: Bearer {_JWT}", None, None)
    assert f.filter(rec) is True
    assert _JWT not in rec.getMessage()


# --- RoofAgeApi ---


def test_roof_age_bearer_needs_no_api_key(monkeypatch):
    """RoofAgeApi accepts a bearer token with no API_KEY, and threads it to the base client."""
    monkeypatch.delenv("API_KEY", raising=False)
    api = RoofAgeApi(bearer_token=_JWT)
    assert api.bearer_token == _JWT
    assert api.api_key == ""


def test_roof_age_session_sets_bearer_header(monkeypatch):
    """The inherited session scope sets the Authorization header in bearer mode."""
    monkeypatch.delenv("API_KEY", raising=False)
    api = RoofAgeApi(bearer_token=_JWT)
    with api._session_scope() as session:
        assert session.headers["Authorization"] == f"Bearer {_JWT}"


def test_roof_age_session_no_header_in_key_mode():
    """Control: key mode sets no Authorization header on the session."""
    api = RoofAgeApi(api_key="dummy")
    with api._session_scope() as session:
        assert "Authorization" not in session.headers


# --- coverage_utils ---


@responses.activate
def test_coverage_get_payload_bearer_header():
    """get_payload sends Authorization: Bearer and no apikey param in bearer mode."""
    responses.add(responses.GET, "https://api.nearmap.com/coverage/v2/point/0,0", json={"surveys": []}, status=200)
    coverage_utils.get_payload("https://api.nearmap.com/coverage/v2/point/0,0", bearer_token=_JWT)
    req = responses.calls[0].request
    assert req.headers["Authorization"] == f"Bearer {_JWT}"
    assert "apikey=" not in req.url


@responses.activate
def test_coverage_get_payload_apikey_mode():
    """Control: key mode sends apikey as a query param, no Authorization header."""
    responses.add(responses.GET, "https://api.nearmap.com/coverage/v2/point/0,0", json={"surveys": []}, status=200)
    coverage_utils.get_payload("https://api.nearmap.com/coverage/v2/point/0,0", apikey="dummy")
    req = responses.calls[0].request
    assert "apikey=dummy" in req.url
    assert "Authorization" not in req.headers


def test_coverage_resolve_apikey_bearer_skips_key(monkeypatch):
    """_resolve_apikey returns None (no key needed) when a bearer token is given."""
    monkeypatch.delenv("API_KEY", raising=False)
    assert coverage_utils._resolve_apikey(None, bearer_token=_JWT) is None


def test_coverage_resolve_apikey_no_credential_raises(monkeypatch):
    """_resolve_apikey still raises when neither key nor bearer token is present."""
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key or bearer token"):
        coverage_utils._resolve_apikey(None)


# --- prefer3d 2D-fallback clone ---


def test_clone_for_2d_fallback_propagates_bearer(monkeypatch):
    """The prefer3d 2D-fallback clone must carry the bearer token, and must not
    silently fall back to a long-lived API_KEY from the environment (the clone's
    api_key arg is the zeroed "" and is falsy, so without the bearer kwarg the
    guard would resolve the env key)."""
    monkeypatch.setenv("API_KEY", "env-key-that-must-not-be-used")
    api = FeatureApi(bearer_token=_JWT, prefer3d=True)
    clone = api._clone_for_2d_fallback()
    assert clone.bearer_token == _JWT
    assert clone.api_key == ""


def test_clone_for_2d_fallback_key_mode_unchanged():
    """Control: key mode clones keep the api key and carry no bearer token."""
    api = FeatureApi(api_key="dummy", prefer3d=True)
    clone = api._clone_for_2d_fallback()
    assert clone.api_key == "dummy"
    assert clone.bearer_token is None


# --- DamageConflationApi ---

_DAMAGE_URL = f"https://api.nearmap.com/ai/damage/v2/events/{_EVENT_ID}/latest.geojson"
_EMPTY_FC = {"type": "FeatureCollection", "features": []}


@responses.activate
def test_damage_conflation_bearer_sends_header_and_no_apikey(monkeypatch):
    """get_damage_by_aoi in bearer mode: Authorization header present, no apikey param."""
    monkeypatch.delenv("API_KEY", raising=False)
    responses.add(responses.POST, _DAMAGE_URL, json=_EMPTY_FC, status=200)
    api = DamageConflationApi(event_id=_EVENT_ID, bearer_token=_JWT)
    api.get_damage_by_aoi(box(0, 0, 1, 1), aoi_id="a")
    req = responses.calls[0].request
    assert req.headers["Authorization"] == f"Bearer {_JWT}"
    assert "apikey=" not in req.url


@responses.activate
def test_damage_conflation_key_mode_no_auth_header():
    """Control: key mode sends apikey as a query param and no Authorization header."""
    responses.add(responses.POST, _DAMAGE_URL, json=_EMPTY_FC, status=200)
    api = DamageConflationApi(event_id=_EVENT_ID, api_key="dummy")
    api.get_damage_by_aoi(box(0, 0, 1, 1), aoi_id="a")
    req = responses.calls[0].request
    assert "apikey=dummy" in req.url
    assert "Authorization" not in req.headers


_ADDRESS = {"streetAddress": "1 Main St", "city": "Springfield", "state": "IL", "zip": "62701"}


@responses.activate
def test_damage_conflation_address_bearer_sends_header_and_no_apikey(monkeypatch):
    """get_damage_by_address in bearer mode: Authorization header present, no apikey param."""
    monkeypatch.delenv("API_KEY", raising=False)
    responses.add(responses.POST, _DAMAGE_URL, json=_EMPTY_FC, status=200)
    api = DamageConflationApi(event_id=_EVENT_ID, bearer_token=_JWT)
    api.get_damage_by_address(_ADDRESS, aoi_id="a")
    req = responses.calls[0].request
    assert req.headers["Authorization"] == f"Bearer {_JWT}"
    assert "apikey=" not in req.url


@responses.activate
def test_damage_conflation_address_key_mode_sends_apikey():
    """Control: key mode sends apikey as a query param and no Authorization header."""
    responses.add(responses.POST, _DAMAGE_URL, json=_EMPTY_FC, status=200)
    api = DamageConflationApi(event_id=_EVENT_ID, api_key="dummy")
    api.get_damage_by_address(_ADDRESS, aoi_id="a")
    req = responses.calls[0].request
    assert "apikey=dummy" in req.url
    assert "Authorization" not in req.headers


# --- RoofAgeApi request params ---

_ROOF_AGE_URL = "https://api.nearmap.com/ai/roofage/v1/resources/latest.json"


@responses.activate
def test_roof_age_aoi_bearer_omits_apikey_param(monkeypatch):
    """get_roof_age_by_aoi in bearer mode: Authorization header present, no apikey param."""
    monkeypatch.delenv("API_KEY", raising=False)
    responses.add(responses.POST, _ROOF_AGE_URL, json=_EMPTY_FC, status=200)
    api = RoofAgeApi(bearer_token=_JWT)
    api.get_roof_age_by_aoi(box(0, 0, 1, 1), aoi_id="a")
    req = responses.calls[0].request
    assert req.headers["Authorization"] == f"Bearer {_JWT}"
    assert "apikey=" not in req.url


@responses.activate
def test_roof_age_aoi_key_mode_sends_apikey_param():
    """Control: key mode sends apikey as a query param and no Authorization header."""
    responses.add(responses.POST, _ROOF_AGE_URL, json=_EMPTY_FC, status=200)
    api = RoofAgeApi(api_key="dummy")
    api.get_roof_age_by_aoi(box(0, 0, 1, 1), aoi_id="a")
    req = responses.calls[0].request
    assert "apikey=dummy" in req.url
    assert "Authorization" not in req.headers
