from unittest.mock import patch

import psutil

from nmaipy.base_exporter import BaseExporter
from nmaipy.cgroup_memory import (
    get_cgroup_cpu_limit,
    get_cpu_info_cgroup_aware,
)


class TestGetCgroupCpuLimitV2:
    def test_parses_standard_limit(self, tmp_path, monkeypatch):
        """200000/100000 = 2.0 CPUs"""
        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("200000 100000\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_CPU_MAX", str(cpu_max))
        assert get_cgroup_cpu_limit() == 2.0

    def test_parses_fractional_limit(self, tmp_path, monkeypatch):
        """150000/100000 = 1.5 CPUs"""
        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("150000 100000\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_CPU_MAX", str(cpu_max))
        assert get_cgroup_cpu_limit() == 1.5

    def test_unlimited_returns_none(self, tmp_path, monkeypatch):
        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("max 100000\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_CPU_MAX", str(cpu_max))
        assert get_cgroup_cpu_limit() is None

    def test_malformed_file_returns_none(self, tmp_path, monkeypatch):
        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("garbage\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_CPU_MAX", str(cpu_max))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_CPU_QUOTA", str(tmp_path / "nonexistent"))
        assert get_cgroup_cpu_limit() is None


class TestGetCgroupCpuLimitV1:
    def test_parses_quota_and_period(self, tmp_path, monkeypatch):
        """400000/100000 = 4.0 CPUs"""
        quota_file = tmp_path / "cpu.cfs_quota_us"
        period_file = tmp_path / "cpu.cfs_period_us"
        quota_file.write_text("400000\n")
        period_file.write_text("100000\n")
        # Ensure v2 path doesn't exist
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_CPU_MAX", str(tmp_path / "nonexistent"))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_CPU_QUOTA", str(quota_file))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_CPU_PERIOD", str(period_file))
        assert get_cgroup_cpu_limit() == 4.0

    def test_unlimited_returns_none(self, tmp_path, monkeypatch):
        quota_file = tmp_path / "cpu.cfs_quota_us"
        quota_file.write_text("-1\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_CPU_MAX", str(tmp_path / "nonexistent"))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_CPU_QUOTA", str(quota_file))
        assert get_cgroup_cpu_limit() is None

    def test_uses_default_period_when_missing(self, tmp_path, monkeypatch):
        """If period file missing, defaults to 100000"""
        quota_file = tmp_path / "cpu.cfs_quota_us"
        quota_file.write_text("200000\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_CPU_MAX", str(tmp_path / "nonexistent"))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_CPU_QUOTA", str(quota_file))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_CPU_PERIOD", str(tmp_path / "nonexistent"))
        assert get_cgroup_cpu_limit() == 2.0


class TestGetCpuInfoCgroupAware:
    def test_returns_tuple_of_floats(self, monkeypatch):
        # Force bare-metal path
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_MAX", "/nonexistent")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_MEMORY_LIMIT", "/nonexistent")
        cpu_pct, cpu_count = get_cpu_info_cgroup_aware()
        assert isinstance(cpu_pct, float)
        assert isinstance(cpu_count, float)
        assert 0.0 <= cpu_pct <= 100.0
        assert cpu_count >= 1.0

    def test_bare_metal_uses_psutil_cpu_count(self, monkeypatch):
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_MAX", "/nonexistent")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_MEMORY_LIMIT", "/nonexistent")
        _, cpu_count = get_cpu_info_cgroup_aware()
        assert cpu_count == float(psutil.cpu_count())

    def test_container_uses_cgroup_limit(self, tmp_path, monkeypatch):
        # Set up memory paths so is_running_in_container() returns True
        mem_max = tmp_path / "memory.max"
        mem_max.write_text("8589934592\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_MAX", str(mem_max))
        # Set up CPU limit
        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("400000 100000\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_CPU_MAX", str(cpu_max))
        _, cpu_count = get_cpu_info_cgroup_aware()
        assert cpu_count == 4.0


class TestFormatProgressDescription:
    def test_contains_memory_and_cpu(self):
        with patch("nmaipy.base_exporter.get_memory_info_cgroup_aware", return_value=(8.5, 32.0)), \
             patch("nmaipy.base_exporter.get_cpu_info_cgroup_aware", return_value=(65.0, 16.0)):
            desc = BaseExporter._format_progress_description(3, 10, lat_str="P50=42ms")
        assert "8.5/32.0GB" in desc
        assert "CPU 65% (16)" in desc
        assert "P50=42ms" in desc
        assert "Chunks: 3/10" in desc

    def test_warmup_default(self):
        with patch("nmaipy.base_exporter.get_memory_info_cgroup_aware", return_value=(1.0, 8.0)), \
             patch("nmaipy.base_exporter.get_cpu_info_cgroup_aware", return_value=(10.0, 4.0)):
            desc = BaseExporter._format_progress_description(0, 5)
        assert "Warmup" in desc
        assert "CPU 10% (4)" in desc

    def test_fractional_cpu_count(self):
        with patch("nmaipy.base_exporter.get_memory_info_cgroup_aware", return_value=(1.0, 8.0)), \
             patch("nmaipy.base_exporter.get_cpu_info_cgroup_aware", return_value=(50.0, 2.5)):
            desc = BaseExporter._format_progress_description(1, 5)
        assert "CPU 50% (2.5)" in desc
