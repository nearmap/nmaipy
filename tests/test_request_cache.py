"""Tests for the per-request cache hot paths: orjson round-trip, in-flight
coalescing of duplicate requests, and tolerant cache writes.

These run without API_KEY by patching ``requests.Session.post`` to return
canned responses built from the real cached Feature API payload in
``tests/data/test_defensible_space_raw_payload.json`` — no synthetic mock
data (see project memory: real data only).
"""

import gzip
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from shapely.geometry import Polygon

from nmaipy import storage
from nmaipy.api_common import AIFeatureAPIError
from nmaipy.feature_api import FeatureApi


@pytest.fixture(scope="module")
def real_feature_payload() -> dict:
    path = Path(__file__).parent / "data" / "test_defensible_space_raw_payload.json"
    return json.loads(path.read_text())


def _square(lon: float, lat: float, size: float = 0.0005) -> Polygon:
    return Polygon(
        [
            (lon, lat),
            (lon + size, lat),
            (lon + size, lat + size),
            (lon, lat + size),
            (lon, lat),
        ]
    )


class _CannedResponse:
    """Minimal stand-in for requests.Response used by FeatureApi._fetch_results."""

    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.history = []
        self.text = json.dumps(body)
        self.content = self.text.encode("utf-8")
        self._body = body

    def json(self):
        return self._body


class TestCacheRoundTrip:
    """orjson-serialized cache entries must read back identical to the payload."""

    @pytest.mark.parametrize("compress", [False, True])
    def test_write_to_cache_round_trip(self, tmp_path, real_feature_payload, compress):
        api = FeatureApi(api_key="dummy", cache_dir=tmp_path, compress_cache=compress)
        ext = "json.gz" if compress else "json"
        cache_path = tmp_path / "151" / "-33" / f"roundtrip.{ext}"

        api._write_to_cache(cache_path, real_feature_payload)

        assert cache_path.exists()
        result = storage.read_json(str(cache_path), compressed=compress)
        assert result == real_feature_payload

    def test_read_cached_response_returns_none_for_corrupt_entry(self, tmp_path):
        """A truncated/corrupt entry is a miss (logged), never an exception."""
        api = FeatureApi(api_key="dummy", cache_dir=tmp_path, compress_cache=True)
        cache_path = tmp_path / "corrupt.json.gz"
        cache_path.write_bytes(b"not gzip at all")

        assert api._read_cached_response(str(cache_path)) is None

    def test_read_cached_response_returns_none_for_legacy_nan_entry(self, tmp_path):
        """stdlib json wrote non-standard NaN literals, which orjson rejects.

        Such legacy entries must degrade to a cache miss (refetch and rewrite),
        not an exception.
        """
        api = FeatureApi(api_key="dummy", cache_dir=tmp_path, compress_cache=True)
        cache_path = tmp_path / "legacy_nan.json.gz"
        legacy = json.dumps({"features": [], "score": float("nan")})  # emits NaN literal
        cache_path.write_bytes(gzip.compress(legacy.encode("utf-8")))

        assert api._read_cached_response(str(cache_path)) is None


class TestInflightCoalescing:
    """Concurrent byte-identical requests must result in exactly one API fetch."""

    N_THREADS = 8

    def _run_concurrent(self, api, geometries):
        with ThreadPoolExecutor(max_workers=len(geometries)) as pool:
            futures = [pool.submit(api._get_results, geometry=g, region="us", packs=["building"]) for g in geometries]
            return [f.result() for f in futures]

    def test_duplicate_requests_fetch_once(self, tmp_path, real_feature_payload):
        api = FeatureApi(api_key="dummy", cache_dir=tmp_path, compress_cache=True)
        post_count = [0]
        count_lock = threading.Lock()

        def fake_post(self, url, *args, **kwargs):
            with count_lock:
                post_count[0] += 1
            time.sleep(0.2)  # hold the fetch in flight so duplicates pile up
            return _CannedResponse(200, real_feature_payload)

        aoi = _square(-111.926, 33.414)
        with patch("requests.Session.post", new=fake_post):
            results = self._run_concurrent(api, [aoi] * self.N_THREADS)

        assert post_count[0] == 1
        assert api._cache_misses == 1
        assert api._cache_hits == self.N_THREADS - 1
        assert all(r == real_feature_payload for r in results)
        assert api._inflight_requests == {}, "lock map must drain when requests complete"

    def test_distinct_requests_do_not_coalesce(self, tmp_path, real_feature_payload):
        api = FeatureApi(api_key="dummy", cache_dir=tmp_path, compress_cache=True)
        post_count = [0]
        count_lock = threading.Lock()

        def fake_post(self, url, *args, **kwargs):
            with count_lock:
                post_count[0] += 1
            time.sleep(0.2)
            return _CannedResponse(200, real_feature_payload)

        geometries = [_square(-111.926, 33.414), _square(-111.936, 33.424)]
        with patch("requests.Session.post", new=fake_post):
            self._run_concurrent(api, geometries)

        assert post_count[0] == 2
        assert api._cache_misses == 2
        assert api._cache_hits == 0
        assert api._inflight_requests == {}

    def test_creator_reprobes_cache_after_entry_drains(self, tmp_path, real_feature_payload):
        """A thread parked between its fast-path cache miss and entering the
        coalescer can find the in-flight entry already drained (a competing
        thread fetched, cached, and left). It becomes a fresh entry creator and
        must re-probe the cache under the lock instead of re-fetching a warm key."""
        api = FeatureApi(api_key="dummy", cache_dir=tmp_path, compress_cache=True)
        posts = []
        post_lock = threading.Lock()

        def fake_post(self, url, *args, **kwargs):
            with post_lock:
                posts.append(threading.current_thread().name)
            return _CannedResponse(200, real_feature_payload)

        a_parked = threading.Event()
        b_done = threading.Event()
        real_read = FeatureApi._read_cached_response

        def parking_read(self, cache_path):
            # Park thread A after its first (fast-path) cache read, while B
            # runs a complete fetch-and-cache cycle and drains its entry.
            result = real_read(self, cache_path)
            if threading.current_thread().name == "park-A" and not a_parked.is_set():
                a_parked.set()
                assert b_done.wait(timeout=10)
            return result

        aoi = _square(-111.926, 33.414)

        def run_a():
            threading.current_thread().name = "park-A"
            return api._get_results(geometry=aoi, region="us", packs=["building"])

        def run_b():
            threading.current_thread().name = "run-B"
            assert a_parked.wait(timeout=10)
            result = api._get_results(geometry=aoi, region="us", packs=["building"])
            b_done.set()
            return result

        with patch("requests.Session.post", new=fake_post):
            with patch.object(FeatureApi, "_read_cached_response", parking_read):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    future_a = pool.submit(run_a)
                    future_b = pool.submit(run_b)
                    result_a, result_b = future_a.result(), future_b.result()

        assert posts == ["run-B"], "thread A must hit the cache B populated, not re-fetch it"
        assert api._cache_hits == 1
        assert api._cache_misses == 1
        assert result_a == result_b == real_feature_payload
        assert api._inflight_requests == {}

    def test_overwrite_cache_mode_does_not_coalesce(self, tmp_path, real_feature_payload):
        """Overwrite mode must refetch every request by design."""
        api = FeatureApi(api_key="dummy", cache_dir=tmp_path, compress_cache=True, overwrite_cache=True)
        post_count = [0]
        count_lock = threading.Lock()

        def fake_post(self, url, *args, **kwargs):
            with count_lock:
                post_count[0] += 1
            time.sleep(0.1)
            return _CannedResponse(200, real_feature_payload)

        aoi = _square(-111.926, 33.414)
        with patch("requests.Session.post", new=fake_post):
            self._run_concurrent(api, [aoi] * 4)

        assert post_count[0] == 4
        assert api._cache_misses == 4
        assert api._cache_hits == 0

    def test_failed_fetch_does_not_poison_the_key(self, tmp_path, real_feature_payload):
        """If the first fetch errors, a waiter proceeds with its own fetch."""
        api = FeatureApi(api_key="dummy", cache_dir=tmp_path, compress_cache=True)
        post_count = [0]
        count_lock = threading.Lock()

        def fake_post(self, url, *args, **kwargs):
            with count_lock:
                post_count[0] += 1
                calls = post_count[0]
            time.sleep(0.2)
            if calls == 1:
                return _CannedResponse(404, {"message": "not found"})
            return _CannedResponse(200, real_feature_payload)

        aoi = _square(-111.926, 33.414)
        outcomes = []
        with patch("requests.Session.post", new=fake_post):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(api._get_results, geometry=aoi, region="us", packs=["building"]) for _ in range(2)
                ]
                for f in futures:
                    try:
                        outcomes.append(("ok", f.result()))
                    except AIFeatureAPIError:
                        outcomes.append(("error", None))

        assert post_count[0] == 2
        statuses = sorted(status for status, _ in outcomes)
        assert statuses == ["error", "ok"]
        payloads = [payload for status, payload in outcomes if status == "ok"]
        assert payloads == [real_feature_payload]
        assert api._inflight_requests == {}

    def test_waiters_fail_in_parallel_after_leader_failure(self, tmp_path, real_feature_payload):
        """After a failed fetch, queued duplicates fetch independently in
        parallel (legacy behaviour) rather than serializing behind the per-key
        lock — serialized, four 0.25s fetches would take >= 1.0s; the leader
        plus three parallel waiters take ~0.5s."""
        api = FeatureApi(api_key="dummy", cache_dir=tmp_path, compress_cache=True)
        post_count = [0]
        count_lock = threading.Lock()
        fetch_seconds = 0.25

        def fake_post(self, url, *args, **kwargs):
            with count_lock:
                post_count[0] += 1
                calls = post_count[0]
            time.sleep(fetch_seconds)
            if calls == 1:
                return _CannedResponse(404, {"message": "not found"})
            return _CannedResponse(200, real_feature_payload)

        aoi = _square(-111.926, 33.414)
        statuses = []
        t0 = time.monotonic()
        with patch("requests.Session.post", new=fake_post):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(api._get_results, geometry=aoi, region="us", packs=["building"]) for _ in range(4)
                ]
                for f in futures:
                    try:
                        f.result()
                        statuses.append("ok")
                    except AIFeatureAPIError:
                        statuses.append("error")
        elapsed = time.monotonic() - t0

        # Between 2 and 4 fetches: normally all four threads fetch, but a worker
        # scheduled late enough may legitimately hit the cache a successful
        # waiter populated. The timing bound is the serialization check: four
        # serialized 0.25s fetches would take >= 1.0s.
        assert 2 <= post_count[0] <= 4
        assert sorted(statuses) == ["error", "ok", "ok", "ok"]
        assert elapsed < 3 * fetch_seconds + 0.1, f"waiters serialized after leader failure ({elapsed:.2f}s)"
        assert api._inflight_requests == {}


class TestTolerantCacheWrite:
    """Cache writes are best-effort: an externally-deleted cache dir must not fail the AOI."""

    def test_write_recreates_externally_deleted_cache_dir(self, tmp_path, real_feature_payload):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        api = FeatureApi(api_key="dummy", cache_dir=cache_dir, compress_cache=True)
        cache_path = cache_dir / "151" / "-33" / "entry.json.gz"

        shutil.rmtree(cache_dir)  # external actor deletes the whole cache tree mid-run
        api._write_to_cache(cache_path, real_feature_payload)

        assert cache_path.exists()
        assert storage.read_json(str(cache_path), compressed=True) == real_feature_payload

    def test_unwritable_cache_path_warns_and_skips(self, tmp_path, real_feature_payload, caplog):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        api = FeatureApi(api_key="dummy", cache_dir=cache_dir, compress_cache=True)
        (cache_dir / "blocker").write_text("a file where a directory must go")
        cache_path = cache_dir / "blocker" / "entry.json.gz"

        # The nmaipy logger has propagate=False, so attach caplog's handler directly.
        nmaipy_logger = logging.getLogger("nmaipy")
        nmaipy_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING, logger="nmaipy"):
                api._write_to_cache(cache_path, real_feature_payload)  # must not raise
        finally:
            nmaipy_logger.removeHandler(caplog.handler)

        assert not cache_path.exists()
        assert any("Failed to write cache entry" in r.message for r in caplog.records)
