"""
Tests for latency statistics tracking and aggregation.

These tests verify:
- Percentile calculation from histograms
- Global stats aggregation with bootstrap CIs
- Per-chunk stats collection from API clients
- CSV/parquet I/O for latency data
- Reproducibility of bootstrap sampling with seeded RNG
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from nmaipy.api_common import (
    LATENCY_BUCKETS,
    collect_latency_stats_from_apis,
    combine_chunk_latency_stats,
    compute_global_latency_stats,
    percentile_from_histogram,
    read_latency_csv,
    save_chunk_latency_stats,
    write_latency_csv,
    _get_latency_bucket_names,
    _stats_to_row,
)


class TestPercentileFromHistogram:
    """Tests for percentile_from_histogram function."""

    def test_empty_histogram_returns_zero(self):
        """Empty histogram should return 0.0 for any percentile."""
        hist = np.array([0] * (len(LATENCY_BUCKETS) - 1))
        assert percentile_from_histogram(hist, LATENCY_BUCKETS, 50) == 0.0
        assert percentile_from_histogram(hist, LATENCY_BUCKETS, 99) == 0.0

    def test_all_in_first_bucket(self):
        """All samples in first bucket [0, 50) should give p50 around 25."""
        hist = np.array([100] + [0] * (len(LATENCY_BUCKETS) - 2))
        p50 = percentile_from_histogram(hist, LATENCY_BUCKETS, 50)
        assert 0 <= p50 <= 50

    def test_all_in_last_bucket_infinity(self):
        """All samples in infinity bucket should return lower bound (60000)."""
        hist = np.array([0] * (len(LATENCY_BUCKETS) - 2) + [100])
        p50 = percentile_from_histogram(hist, LATENCY_BUCKETS, 50)
        # Last bucket is [60000, inf), so should return 60000
        assert p50 == 60000.0

    def test_uniform_distribution(self):
        """Uniform distribution across buckets should give reasonable percentiles."""
        # Put 10 samples in each bucket (except infinity)
        hist = np.array([10] * (len(LATENCY_BUCKETS) - 2) + [0])
        p50 = percentile_from_histogram(hist, LATENCY_BUCKETS, 50)
        # Should be somewhere in the middle buckets
        assert 100 < p50 < 10000

    def test_known_distribution(self):
        """Test with a known distribution for exact verification."""
        # All 100 samples in the [100, 150) bucket
        bucket_idx = LATENCY_BUCKETS.index(100)  # Find the bucket starting at 100ms
        hist = np.array([0] * (len(LATENCY_BUCKETS) - 1))
        hist[bucket_idx] = 100

        p50 = percentile_from_histogram(hist, LATENCY_BUCKETS, 50)
        # P50 should be in [100, 150) range
        assert 100 <= p50 <= 150


class TestComputeGlobalLatencyStats:
    """Tests for compute_global_latency_stats function."""

    def test_empty_chunk_stats_returns_empty(self):
        """Empty input should return empty dict."""
        assert compute_global_latency_stats([]) == {}

    def test_single_chunk(self):
        """Single chunk should return its stats."""
        chunk_stats = [{
            "mean": 100.0,
            "count": 50,
            "histogram": [10, 20, 15, 5] + [0] * (len(LATENCY_BUCKETS) - 5),
        }]
        result = compute_global_latency_stats(chunk_stats)

        assert result["mean"] == 100.0
        assert result["count"] == 50
        assert "p50" in result
        assert "p50_ci" in result

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical results."""
        chunk_stats = [
            {"mean": 100.0, "count": 50, "histogram": [10, 20, 15, 5] + [0] * (len(LATENCY_BUCKETS) - 5)},
            {"mean": 150.0, "count": 30, "histogram": [5, 10, 10, 5] + [0] * (len(LATENCY_BUCKETS) - 5)},
        ]

        result1 = compute_global_latency_stats(chunk_stats, seed=42)
        result2 = compute_global_latency_stats(chunk_stats, seed=42)

        assert result1["p50_ci"] == result2["p50_ci"]
        assert result1["p90_ci"] == result2["p90_ci"]

    def test_different_seeds_produce_different_cis(self):
        """Different seeds should produce (slightly) different CIs with enough chunks."""
        # Need more chunks for bootstrap variability to manifest
        chunk_stats = [
            {"mean": 80.0, "count": 40, "histogram": [15, 15, 8, 2] + [0] * (len(LATENCY_BUCKETS) - 5)},
            {"mean": 100.0, "count": 50, "histogram": [10, 20, 15, 5] + [0] * (len(LATENCY_BUCKETS) - 5)},
            {"mean": 120.0, "count": 45, "histogram": [5, 15, 18, 7] + [0] * (len(LATENCY_BUCKETS) - 5)},
            {"mean": 150.0, "count": 30, "histogram": [5, 10, 10, 5] + [0] * (len(LATENCY_BUCKETS) - 5)},
            {"mean": 180.0, "count": 35, "histogram": [2, 8, 15, 10] + [0] * (len(LATENCY_BUCKETS) - 5)},
        ]

        result1 = compute_global_latency_stats(chunk_stats, seed=42)
        result2 = compute_global_latency_stats(chunk_stats, seed=999)

        # With 5 chunks and different seeds, CIs should differ
        # Using p99 which is more sensitive to tail resampling
        assert result1["p99_ci"] != result2["p99_ci"]

    def test_weighted_mean_calculation(self):
        """Global mean should be weighted by count."""
        chunk_stats = [
            {"mean": 100.0, "count": 100, "histogram": [100] + [0] * (len(LATENCY_BUCKETS) - 2)},
            {"mean": 200.0, "count": 100, "histogram": [0, 0, 0, 100] + [0] * (len(LATENCY_BUCKETS) - 5)},
        ]

        result = compute_global_latency_stats(chunk_stats)

        # Weighted mean: (100*100 + 200*100) / 200 = 150
        assert result["mean"] == 150.0

    def test_zero_count_chunks_handled(self):
        """Chunks with zero count should be handled gracefully."""
        chunk_stats = [
            {"mean": 0, "count": 0, "histogram": [0] * (len(LATENCY_BUCKETS) - 1)},
        ]

        result = compute_global_latency_stats(chunk_stats)
        assert result["count"] == 0


class TestCollectLatencyStatsFromApis:
    """Tests for collect_latency_stats_from_apis function."""

    def _create_mock_api(self, latencies, retry_count=0, timeout_count=0, cache_hits=0, cache_misses=0):
        """Create a mock API client with specified latency data."""
        api = MagicMock()
        api._latencies = latencies
        api._retry_count = retry_count
        api._timeout_count = timeout_count
        api._cache_hits = cache_hits
        api._cache_misses = cache_misses
        return api

    def test_no_apis_returns_none(self):
        """No API clients should return None."""
        result = collect_latency_stats_from_apis(
            [], "chunk_0", "2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", 60000
        )
        assert result is None

    def test_none_apis_returns_none(self):
        """All None API clients should return None."""
        result = collect_latency_stats_from_apis(
            [None, None], "chunk_0", "2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", 60000
        )
        assert result is None

    def test_single_api_with_latencies(self):
        """Single API with latencies should return stats."""
        api = self._create_mock_api([100, 150, 200], retry_count=1, cache_misses=3)

        result = collect_latency_stats_from_apis(
            [api], "chunk_0", "2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", 60000
        )

        assert result is not None
        assert result["chunk_id"] == "chunk_0"
        assert result["count"] == 3
        assert result["mean"] == 150.0
        assert result["retry_count"] == 1
        assert result["cache_misses"] == 3
        assert result["total_duration_ms"] == 60000

    def test_multiple_apis_combined(self):
        """Multiple APIs should have their stats combined."""
        api1 = self._create_mock_api([100, 200], retry_count=1, cache_hits=5)
        api2 = self._create_mock_api([150, 250], retry_count=2, cache_hits=3)

        result = collect_latency_stats_from_apis(
            [api1, api2], "chunk_0", "2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", 60000
        )

        assert result["count"] == 4
        assert result["mean"] == 175.0  # (100+200+150+250)/4
        assert result["retry_count"] == 3  # 1+2
        assert result["cache_hits"] == 8  # 5+3

    def test_mixed_none_and_real_apis(self):
        """Mix of None and real APIs should work."""
        api = self._create_mock_api([100, 200, 300])

        result = collect_latency_stats_from_apis(
            [None, api, None], "chunk_0", "2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", 60000
        )

        assert result["count"] == 3

    def test_empty_latencies_returns_none(self):
        """API with empty latencies should return None."""
        api = self._create_mock_api([])

        result = collect_latency_stats_from_apis(
            [api], "chunk_0", "2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", 60000
        )

        assert result is None


class TestLatencyIO:
    """Tests for latency stats I/O functions."""

    def test_stats_to_row_none_returns_none(self):
        """None stats should return None."""
        assert _stats_to_row(None) is None

    def test_stats_to_row_basic(self):
        """Basic stats should convert to row dict."""
        stats = {
            "chunk_id": "test_chunk",
            "count": 100,
            "mean": 150.0,
            "p50": 140.0,
            "p90": 200.0,
            "p95": 250.0,
            "p99": 300.0,
            "min": 50.0,
            "max": 500.0,
            "histogram": [10, 20, 30, 40] + [0] * (len(LATENCY_BUCKETS) - 5),
            "retry_count": 5,
            "timeout_count": 1,
            "cache_hits": 80,
            "cache_misses": 20,
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T00:01:00Z",
            "total_duration_ms": 60000,
        }

        row = _stats_to_row(stats)

        assert row["chunk_id"] == "test_chunk"
        assert row["count"] == 100
        assert row["rps"] == pytest.approx(100 / 60, rel=0.01)

    def test_save_and_combine_chunk_latency_stats(self):
        """Save per-chunk stats and combine them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_path = Path(tmpdir)

            # Save two chunks
            stats1 = {
                "chunk_id": "file_0000",
                "count": 50,
                "mean": 100.0,
                "p50": 90.0,
                "p90": 150.0,
                "p95": 180.0,
                "p99": 200.0,
                "min": 20.0,
                "max": 250.0,
                "histogram": [10, 20, 15, 5] + [0] * (len(LATENCY_BUCKETS) - 5),
                "retry_count": 2,
                "timeout_count": 0,
                "cache_hits": 40,
                "cache_misses": 10,
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T00:00:30Z",
                "total_duration_ms": 30000,
            }
            stats2 = {
                "chunk_id": "file_0001",
                "count": 30,
                "mean": 120.0,
                "p50": 110.0,
                "p90": 170.0,
                "p95": 200.0,
                "p99": 220.0,
                "min": 30.0,
                "max": 280.0,
                "histogram": [5, 15, 8, 2] + [0] * (len(LATENCY_BUCKETS) - 5),
                "retry_count": 1,
                "timeout_count": 1,
                "cache_hits": 25,
                "cache_misses": 5,
                "start_time": "2024-01-01T00:00:30Z",
                "end_time": "2024-01-01T00:01:00Z",
                "total_duration_ms": 30000,
            }

            save_chunk_latency_stats(stats1, chunk_path, "file_0000")
            save_chunk_latency_stats(stats2, chunk_path, "file_0001")

            # Verify files exist
            assert (chunk_path / "latency_file_0000.parquet").exists()
            assert (chunk_path / "latency_file_0001.parquet").exists()

            # Combine them
            output_csv = chunk_path / "combined_latency.csv"
            combined = combine_chunk_latency_stats(chunk_path, "file", output_csv)

            assert len(combined) == 2
            assert output_csv.exists()

            # Verify CSV can be read back
            read_back = read_latency_csv(output_csv)
            assert len(read_back) == 2

    def test_write_and_read_latency_csv(self):
        """Write latency CSV and read it back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "latency.csv"

            chunk_stats = [
                {
                    "chunk_id": "chunk_0",
                    "count": 100,
                    "mean": 150.0,
                    "p50": 140.0,
                    "p90": 200.0,
                    "p95": 250.0,
                    "p99": 300.0,
                    "min": 50.0,
                    "max": 500.0,
                    "histogram": [10, 20, 30, 40] + [0] * (len(LATENCY_BUCKETS) - 5),
                    "retry_count": 5,
                    "timeout_count": 1,
                    "cache_hits": 80,
                    "cache_misses": 20,
                    "start_time": "2024-01-01T00:00:00Z",
                    "end_time": "2024-01-01T00:01:00Z",
                    "total_duration_ms": 60000,
                },
            ]

            write_latency_csv(chunk_stats, csv_path)
            assert csv_path.exists()

            read_back = read_latency_csv(csv_path)
            assert len(read_back) == 1
            assert read_back[0]["chunk_id"] == "chunk_0"
            assert read_back[0]["count"] == 100

    def test_read_nonexistent_csv_returns_empty(self):
        """Reading nonexistent CSV should return empty list."""
        result = read_latency_csv("/nonexistent/path/file.csv")
        assert result == []


class TestLatencyBuckets:
    """Tests for LATENCY_BUCKETS configuration."""

    def test_buckets_are_sorted(self):
        """Buckets should be in ascending order."""
        for i in range(len(LATENCY_BUCKETS) - 1):
            assert LATENCY_BUCKETS[i] < LATENCY_BUCKETS[i + 1]

    def test_buckets_start_at_zero(self):
        """First bucket should start at 0."""
        assert LATENCY_BUCKETS[0] == 0

    def test_buckets_end_with_infinity(self):
        """Last bucket should end with infinity."""
        assert LATENCY_BUCKETS[-1] == float("inf")

    def test_bucket_names_match_bucket_count(self):
        """Number of bucket names should match number of buckets."""
        bucket_names = _get_latency_bucket_names()
        assert len(bucket_names) == len(LATENCY_BUCKETS) - 1

    def test_sla_boundary_present(self):
        """SLA boundary at 2000ms should be present."""
        assert 2000 in LATENCY_BUCKETS
