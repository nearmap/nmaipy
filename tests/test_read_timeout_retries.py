"""Tests for read-timeout retry semantics (silent-resend storm fix).

Read timeouts while waiting for response headers are retried INSIDE urllib3, invisibly
to the caller: each retry closes the connection (a client-closed-request entry in
server-side logs) and re-sends the same request, which the server pays for in full.
With the old ``read=maxretry`` budget, one response that legitimately needed longer
than READ_TIMEOUT_SECONDS to generate became a ~half-hour silent resend storm per AOI.

These tests pin the fixed behaviour:
  - urllib3 performs exactly ONE silent read retry, then the timeout surfaces
  - the surfaced exception is ``requests.exceptions.ConnectionError`` (NOT
    ``ReadTimeout``), and ``is_read_timeout_error`` recognises it
  - non-timeout ConnectionErrors (DNS, refused) are NOT classified as read timeouts
  - ``RetryRequest.on_retry`` counts every in-transport retry (the old
    ``response.history`` counting was redirect history — always zero)
  - FeatureApi converts surfaced read timeouts into AIFeatureAPIRequestSizeError
    so the AOI grids instead of erroring
"""

import json
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest
import requests
import urllib3
from requests.adapters import HTTPAdapter
from shapely.geometry import Polygon

from nmaipy.api_common import RetryRequest, is_read_timeout_error
from nmaipy.feature_api import AIFeatureAPIRequestSizeError, FeatureApi

TEST_READ_TIMEOUT = 0.5  # seconds — fast stand-in for READ_TIMEOUT_SECONDS
SLOW = 2.0  # server-side delay that exceeds the read timeout


class _RecordingHandler(BaseHTTPRequestHandler):
    """POST handler that sleeps per a plan and records how each attempt ended."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        server = self.server
        with server.state_lock:
            idx = server.request_count
            server.request_count += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        plan = server.sleep_plan
        time.sleep(plan[idx] if idx < len(plan) else plan[-1])
        body = json.dumps({"attempt": idx}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            with server.state_lock:
                server.aborted_attempts += 1

    def log_message(self, *args):
        pass


@pytest.fixture
def slow_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    server.state_lock = threading.Lock()
    server.request_count = 0
    server.aborted_attempts = 0
    server.sleep_plan = [0]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def _make_session(retry_counts, status_forcelist=(429, 500, 502, 503)):
    """Build a session with the same retry shape as BaseApiClient._session_scope."""

    def on_retry(cause):
        retry_counts.append(cause)

    retries = RetryRequest(
        total=20,
        backoff_factor=0.01,
        backoff_max=0.05,
        status_forcelist=list(status_forcelist),
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
        connect=20,
        read=1,
        redirect=20,
        on_retry=on_retry,
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


def test_read_timeout_surfaces_after_one_silent_retry(slow_server):
    """A persistently slow response must burn exactly 2 attempts (1 + 1 read retry),
    then surface as ConnectionError — not loop through the whole retry budget."""
    slow_server.sleep_plan = [SLOW]  # always slower than the read timeout
    retry_causes = []
    session = _make_session(retry_causes)
    url = f"http://127.0.0.1:{slow_server.server_address[1]}/features.json"

    with pytest.raises(requests.exceptions.ConnectionError) as exc_info:
        session.post(url, json={"aoi": "x"}, timeout=(5, TEST_READ_TIMEOUT))

    time.sleep(SLOW + 0.5)  # let aborted handler threads finish and record
    assert slow_server.request_count == 2, "expected exactly 1 original attempt + 1 silent retry"
    assert is_read_timeout_error(exc_info.value), "surfaced exception must be recognised as a read timeout"
    assert len(retry_causes) == 1
    assert isinstance(retry_causes[0], urllib3.exceptions.ReadTimeoutError)


def test_transient_read_timeout_recovers_silently(slow_server):
    """One slow attempt followed by a fast one must succeed with no surfaced error —
    the single silent retry covers transient stalls / dead connections."""
    slow_server.sleep_plan = [SLOW, 0]
    retry_causes = []
    session = _make_session(retry_causes)
    url = f"http://127.0.0.1:{slow_server.server_address[1]}/features.json"

    response = session.post(url, json={"aoi": "x"}, timeout=(5, TEST_READ_TIMEOUT))

    assert response.status_code == 200
    assert response.json() == {"attempt": 1}
    assert len(retry_causes) == 1, "the one silent read retry must be counted"


def test_status_retries_counted_via_on_retry(slow_server):
    """Status-forcelist retries (the 429/500/502/503 budget) must be counted too —
    the old response.history counting saw none of these."""
    slow_server.sleep_plan = [0]
    retry_causes = []
    session = _make_session(retry_causes)
    url = f"http://127.0.0.1:{slow_server.server_address[1]}/features.json"

    def failing_then_ok(handler):
        with handler.server.state_lock:
            idx = handler.server.request_count
            handler.server.request_count += 1
        handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
        if idx < 2:
            handler.send_response(500)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
        else:
            body = json.dumps({"attempt": idx}).encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

    with patch.object(_RecordingHandler, "do_POST", failing_then_ok):
        response = session.post(url, json={"aoi": "x"}, timeout=(5, TEST_READ_TIMEOUT))

    assert response.status_code == 200
    assert retry_causes == [500, 500], "both 500-triggered retries must be reported with their status"


def test_on_retry_survives_urllib3_immutable_copies():
    """urllib3 copies Retry state via new() on every increment; the callback must ride along."""
    calls = []
    retry = RetryRequest(total=5, read=2, on_retry=calls.append)
    copied = retry.new(total=4)
    assert copied.on_retry is retry.on_retry


def test_is_read_timeout_error_rejects_other_connection_errors():
    """DNS failures / connection-refused must NOT be classified as read timeouts —
    gridding an AOI because DNS is down would silently degrade its results."""
    refused = requests.exceptions.ConnectionError(
        urllib3.exceptions.MaxRetryError(None, "/features.json", reason=urllib3.exceptions.NewConnectionError(None, "refused"))
    )
    assert not is_read_timeout_error(refused)

    dns_gone = requests.exceptions.ConnectionError(socket.gaierror(8, "nodename nor servname provided"))
    assert not is_read_timeout_error(dns_gone)


def test_is_read_timeout_error_accepts_both_surfacing_paths():
    """Header-phase (MaxRetryError-wrapped) and mid-body (bare ReadTimeoutError-wrapped)
    timeouts must both be recognised."""
    header_phase = requests.exceptions.ConnectionError(
        urllib3.exceptions.MaxRetryError(
            None, "/features.json", reason=urllib3.exceptions.ReadTimeoutError(None, "/features.json", "Read timed out.")
        )
    )
    assert is_read_timeout_error(header_phase)

    mid_body = requests.exceptions.ConnectionError(
        urllib3.exceptions.ReadTimeoutError(None, "/features.json", "Read timed out.")
    )
    assert is_read_timeout_error(mid_body)


def test_surfaced_read_timeout_triggers_size_error_for_gridding():
    """A read timeout surfacing as ConnectionError must raise AIFeatureAPIRequestSizeError
    (the gridding trigger), exactly like the legacy plain-ReadTimeout path."""
    api = FeatureApi(api_key="TEST_KEY", cache_dir=None)
    polygon = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)])
    surfaced = requests.exceptions.ConnectionError(
        urllib3.exceptions.MaxRetryError(
            None, "/features.json", reason=urllib3.exceptions.ReadTimeoutError(None, "/features.json", "Read timed out.")
        )
    )

    with patch("requests.Session.post", side_effect=surfaced):
        with pytest.raises(AIFeatureAPIRequestSizeError) as exc_info:
            api._get_results(geometry=polygon, region="au", in_gridding_mode=False)
    assert exc_info.value.status_code == HTTPStatus.GATEWAY_TIMEOUT
    assert api._timeout_count == 1


def test_non_timeout_connection_error_propagates():
    """Infrastructure ConnectionErrors must NOT be converted to size errors/gridding."""
    api = FeatureApi(api_key="TEST_KEY", cache_dir=None)
    polygon = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)])
    refused = requests.exceptions.ConnectionError(
        urllib3.exceptions.MaxRetryError(None, "/features.json", reason=urllib3.exceptions.NewConnectionError(None, "refused"))
    )

    with patch("requests.Session.post", side_effect=refused):
        with pytest.raises(requests.exceptions.ConnectionError):
            api._get_results(geometry=polygon, region="au", in_gridding_mode=False)
    assert api._timeout_count == 0
