from unittest.mock import patch

from nmaipy.cgroup_memory import (
    get_cgroup_inactive_file_bytes,
    get_cgroup_working_set_bytes,
    get_cgroup_working_set_gb,
    get_memory_info_cgroup_aware,
)

# memory.current / memory.usage_in_bytes only carries the usage counter; the
# reclaimable cache figure lives in memory.stat (inactive_file on v2,
# total_inactive_file on v1). Working set = usage - inactive_file.


class TestGetCgroupInactiveFileBytes:
    def test_reads_v2_inactive_file(self, tmp_path, monkeypatch):
        stat = tmp_path / "memory.stat"
        stat.write_text("anon 500\ninactive_file 200\nactive_file 10\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", str(stat))
        assert get_cgroup_inactive_file_bytes() == 200

    def test_reads_v1_total_inactive_file(self, tmp_path, monkeypatch):
        stat = tmp_path / "memory.stat"
        stat.write_text("cache 1000\ntotal_inactive_file 200\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", str(tmp_path / "nonexistent"))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_MEMORY_STAT", str(stat))
        assert get_cgroup_inactive_file_bytes() == 200

    def test_missing_stat_returns_none(self, monkeypatch):
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", "/nonexistent")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_MEMORY_STAT", "/nonexistent")
        assert get_cgroup_inactive_file_bytes() is None

    def test_field_absent_returns_none(self, tmp_path, monkeypatch):
        stat = tmp_path / "memory.stat"
        stat.write_text("anon 500\nactive_file 10\n")  # no inactive_file line
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", str(stat))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_MEMORY_STAT", "/nonexistent")
        assert get_cgroup_inactive_file_bytes() is None


class TestGetCgroupWorkingSetBytes:
    def test_v2_working_set(self, tmp_path, monkeypatch):
        """current=1000, inactive_file=200 -> working set 800."""
        current = tmp_path / "memory.current"
        current.write_text("1000\n")
        stat = tmp_path / "memory.stat"
        stat.write_text("inactive_file 200\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_CURRENT", str(current))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", str(stat))
        assert get_cgroup_working_set_bytes() == 800

    def test_v1_working_set(self, tmp_path, monkeypatch):
        """v1: usage_in_bytes=1000, total_inactive_file=200 -> 800."""
        usage = tmp_path / "memory.usage_in_bytes"
        usage.write_text("1000\n")
        stat = tmp_path / "memory.stat"
        stat.write_text("total_inactive_file 200\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_CURRENT", str(tmp_path / "nonexistent"))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", str(tmp_path / "nonexistent"))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_MEMORY_USAGE", str(usage))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_MEMORY_STAT", str(stat))
        assert get_cgroup_working_set_bytes() == 800

    def test_falls_back_to_raw_usage_when_stat_missing(self, tmp_path, monkeypatch):
        """No memory.stat -> conservative: return raw usage unchanged."""
        current = tmp_path / "memory.current"
        current.write_text("1000\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_CURRENT", str(current))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", "/nonexistent")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_MEMORY_STAT", "/nonexistent")
        assert get_cgroup_working_set_bytes() == 1000

    def test_inactive_exceeds_usage_clamps_to_zero(self, tmp_path, monkeypatch):
        """Pathological inactive_file > usage must not go negative."""
        current = tmp_path / "memory.current"
        current.write_text("100\n")
        stat = tmp_path / "memory.stat"
        stat.write_text("inactive_file 200\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_CURRENT", str(current))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", str(stat))
        assert get_cgroup_working_set_bytes() == 0

    def test_no_usage_returns_none(self, monkeypatch):
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_CURRENT", "/nonexistent")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_MEMORY_USAGE", "/nonexistent")
        assert get_cgroup_working_set_bytes() is None

    def test_working_set_gb_conversion(self, tmp_path, monkeypatch):
        current = tmp_path / "memory.current"
        current.write_text(f"{4 * 1024**3}\n")
        stat = tmp_path / "memory.stat"
        stat.write_text(f"inactive_file {1 * 1024**3}\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_CURRENT", str(current))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", str(stat))
        assert get_cgroup_working_set_gb() == 3.0

    def test_reproduces_measured_live_values(self, tmp_path, monkeypatch):
        """Real block export sample riding the limit: 124.2GB current, 76.7GB cache -> 47.4GB."""
        current = tmp_path / "memory.current"
        current.write_text("133303267328\n")
        stat = tmp_path / "memory.stat"
        stat.write_text("anon 46611353600\ninactive_file 82357862400\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_CURRENT", str(current))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", str(stat))
        assert get_cgroup_working_set_bytes() == 50945404928
        assert round(get_cgroup_working_set_gb(), 1) == 47.4


class TestGetMemoryInfoCgroupAware:
    def test_container_returns_working_set_and_limit(self, tmp_path, monkeypatch):
        """In-container used figure is the working set, not raw current."""
        mem_max = tmp_path / "memory.max"
        mem_max.write_text(f"{8 * 1024**3}\n")
        current = tmp_path / "memory.current"
        current.write_text(f"{6 * 1024**3}\n")
        stat = tmp_path / "memory.stat"
        stat.write_text(f"inactive_file {2 * 1024**3}\n")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_MAX", str(mem_max))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_CURRENT", str(current))
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_STAT", str(stat))
        used_gb, total_gb = get_memory_info_cgroup_aware()
        assert used_gb == 4.0  # 6 current - 2 cache, not 6
        assert total_gb == 8.0

    def test_bare_metal_uses_psutil(self, monkeypatch):
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_MAX", "/nonexistent")
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V1_MEMORY_LIMIT", "/nonexistent")
        used_gb, total_gb = get_memory_info_cgroup_aware()
        assert used_gb > 0.0
        assert total_gb > 0.0
        assert used_gb <= total_gb

    def test_container_falls_back_to_psutil_when_limit_unreadable(self, tmp_path, monkeypatch):
        """is_running_in_container True (limit file exists) but unparseable -> psutil."""
        mem_max = tmp_path / "memory.max"
        mem_max.write_text("max\n")  # exists -> container detected, but no numeric limit
        monkeypatch.setattr("nmaipy.cgroup_memory.CGROUP_V2_MEMORY_MAX", str(mem_max))
        with patch("nmaipy.cgroup_memory.psutil.virtual_memory") as vm:
            vm.return_value.total = 16 * 1024**3
            vm.return_value.available = 4 * 1024**3
            used_gb, total_gb = get_memory_info_cgroup_aware()
        assert total_gb == 16.0
        assert used_gb == 12.0
