"""
Tests for the storage abstraction module.

Tests cover both local filesystem operations (using tmp_path) and S3 operations
(mocking the cached S3 filesystem to avoid requiring real AWS credentials).
"""

import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pyarrow.parquet as pq
import pytest

from nmaipy import storage

# ---------------------------------------------------------------------------
# is_s3_path
# ---------------------------------------------------------------------------


class TestIsS3Path:
    def test_s3_uri(self):
        assert storage.is_s3_path("s3://bucket/key") is True

    def test_s3_uri_with_trailing_slash(self):
        assert storage.is_s3_path("s3://bucket/key/") is True

    def test_local_unix_path(self):
        assert storage.is_s3_path("/tmp/data") is False

    def test_local_relative_path(self):
        assert storage.is_s3_path("data/output") is False

    def test_path_object(self):
        assert storage.is_s3_path(Path("/tmp/data")) is False

    def test_empty_string(self):
        assert storage.is_s3_path("") is False


# ---------------------------------------------------------------------------
# join_path
# ---------------------------------------------------------------------------


class TestJoinPath:
    def test_local_join(self):
        result = storage.join_path("/tmp/output", "chunks", "file.parquet")
        assert result == str(Path("/tmp/output/chunks/file.parquet"))

    def test_local_join_single_part(self):
        result = storage.join_path("/tmp/output", "final")
        assert result == str(Path("/tmp/output/final"))

    def test_s3_join(self):
        result = storage.join_path("s3://bucket/prefix", "chunks", "file.parquet")
        assert result == "s3://bucket/prefix/chunks/file.parquet"

    def test_s3_join_strips_trailing_slash(self):
        result = storage.join_path("s3://bucket/prefix/", "chunks")
        assert result == "s3://bucket/prefix/chunks"

    def test_s3_join_strips_part_slashes(self):
        result = storage.join_path("s3://bucket/prefix", "/chunks/", "/file.parquet/")
        assert result == "s3://bucket/prefix/chunks/file.parquet"

    def test_s3_join_single_part(self):
        result = storage.join_path("s3://bucket/prefix", "final")
        assert result == "s3://bucket/prefix/final"


# ---------------------------------------------------------------------------
# ensure_directory
# ---------------------------------------------------------------------------


class TestEnsureDirectory:
    def test_local_creates_directory(self, tmp_path):
        new_dir = str(tmp_path / "a" / "b" / "c")
        storage.ensure_directory(new_dir)
        assert Path(new_dir).is_dir()

    def test_local_existing_directory_no_error(self, tmp_path):
        storage.ensure_directory(str(tmp_path))

    def test_s3_is_noop(self):
        # Should not raise or try to create anything
        storage.ensure_directory("s3://bucket/prefix/chunks")


# ---------------------------------------------------------------------------
# file_exists
# ---------------------------------------------------------------------------


class TestFileExists:
    def test_local_exists(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert storage.file_exists(str(f)) is True

    def test_local_not_exists(self, tmp_path):
        assert storage.file_exists(str(tmp_path / "nope.txt")) is False

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_exists(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        mock_get_fs.return_value = mock_fs

        assert storage.file_exists("s3://bucket/key.parquet") is True
        mock_fs.exists.assert_called_with("s3://bucket/key.parquet")

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_not_exists(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        mock_get_fs.return_value = mock_fs

        assert storage.file_exists("s3://bucket/missing.parquet") is False


# ---------------------------------------------------------------------------
# glob_files
# ---------------------------------------------------------------------------


class TestGlobFiles:
    def test_local_glob(self, tmp_path):
        (tmp_path / "rollup_001.parquet").write_text("data")
        (tmp_path / "rollup_002.parquet").write_text("data")
        (tmp_path / "metadata_001.parquet").write_text("data")

        results = storage.glob_files(str(tmp_path), "rollup_*.parquet")
        assert len(results) == 2
        assert all("rollup_" in r for r in results)

    def test_local_glob_no_match(self, tmp_path):
        results = storage.glob_files(str(tmp_path), "*.parquet")
        assert results == []

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_glob(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_fs.glob.return_value = [
            "bucket/prefix/chunks/rollup_001.parquet",
            "bucket/prefix/chunks/rollup_002.parquet",
        ]
        mock_get_fs.return_value = mock_fs

        results = storage.glob_files("s3://bucket/prefix/chunks", "rollup_*.parquet")
        assert results == [
            "s3://bucket/prefix/chunks/rollup_001.parquet",
            "s3://bucket/prefix/chunks/rollup_002.parquet",
        ]
        mock_fs.glob.assert_called_with("s3://bucket/prefix/chunks/rollup_*.parquet")

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_glob_invalidates_cache_before_listing(self, mock_get_fs):
        # A stale s3fs dircache can make glob miss present files (it dropped all
        # feature/per-class chunks from the merge, producing an empty
        # features.parquet). glob_files must drop the cache before listing.
        manager = MagicMock()
        mock_get_fs.return_value = manager.fs
        manager.fs.glob.return_value = ["bucket/prefix/chunks/features_0001.parquet"]

        results = storage.glob_files("s3://bucket/prefix/chunks", "features_*.parquet")

        manager.fs.invalidate_cache.assert_called_once_with("s3://bucket/prefix/chunks")
        # invalidate_cache must happen *before* glob.
        call_names = [c[0] for c in manager.fs.mock_calls if c[0] in ("invalidate_cache", "glob")]
        assert call_names == ["invalidate_cache", "glob"], call_names
        assert results == ["s3://bucket/prefix/chunks/features_0001.parquet"]

    @patch("nmaipy.storage.time.sleep", lambda *_: None)
    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_glob_retries_transient_then_succeeds(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_fs.glob.side_effect = [OSError("timeout"), ["bucket/p/chunks/features_0001.parquet"]]
        mock_get_fs.return_value = mock_fs

        results = storage.glob_files("s3://bucket/p/chunks", "features_*.parquet")
        assert results == ["s3://bucket/p/chunks/features_0001.parquet"]
        assert mock_fs.glob.call_count == 2


# ---------------------------------------------------------------------------
# open_file
# ---------------------------------------------------------------------------


class TestOpenFile:
    def test_local_write_and_read(self, tmp_path):
        fpath = str(tmp_path / "test.txt")
        with storage.open_file(fpath, "w") as f:
            f.write("hello world")
        with storage.open_file(fpath, "r") as f:
            assert f.read() == "hello world"

    def test_local_binary_write_and_read(self, tmp_path):
        fpath = str(tmp_path / "test.bin")
        with storage.open_file(fpath, "wb") as f:
            f.write(b"\x00\x01\x02")
        with storage.open_file(fpath, "rb") as f:
            assert f.read() == b"\x00\x01\x02"

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_open_uses_fork_safe_filesystem(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_file = MagicMock()
        mock_fs.open.return_value = mock_file
        mock_get_fs.return_value = mock_fs

        result = storage.open_file("s3://bucket/key.txt", "r")
        mock_fs.open.assert_called_with("s3://bucket/key.txt", "r")
        assert result is mock_file


# ---------------------------------------------------------------------------
# file_size
# ---------------------------------------------------------------------------


class TestFileSize:
    def test_local_file_size(self, tmp_path):
        fpath = tmp_path / "test.txt"
        content = "hello world"
        fpath.write_text(content)
        assert storage.file_size(str(fpath)) == len(content.encode())

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_file_size(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_fs.size.return_value = 1024
        mock_get_fs.return_value = mock_fs

        assert storage.file_size("s3://bucket/key.parquet") == 1024
        mock_fs.size.assert_called_with("s3://bucket/key.parquet")


# ---------------------------------------------------------------------------
# upload_file
# ---------------------------------------------------------------------------


class TestUploadFile:
    def test_local_to_local_copy(self, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("data")

        storage.upload_file(str(src), str(dst))
        assert dst.read_text() == "data"

    def test_local_same_path_noop(self, tmp_path):
        fpath = str(tmp_path / "file.txt")
        Path(fpath).write_text("data")
        # Should not raise
        storage.upload_file(fpath, fpath)

    @patch("nmaipy.storage._get_s3_boto3_client")
    def test_upload_to_s3(self, mock_get_client, tmp_path):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create a real local file so Path.stat() works
        src = tmp_path / "local.parquet"
        src.write_bytes(b"data")

        storage.upload_file(str(src), "s3://bucket/key.parquet")
        mock_client.upload_file.assert_called_once_with(
            str(src),
            "bucket",
            "key.parquet",
            Config=None,
        )


# ---------------------------------------------------------------------------
# remove_file
# ---------------------------------------------------------------------------


class TestRemoveFile:
    def test_local_remove(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        storage.remove_file(str(f))
        assert not f.exists()

    def test_local_remove_missing_is_noop(self, tmp_path):
        storage.remove_file(str(tmp_path / "nonexistent.txt"))

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_remove(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_get_fs.return_value = mock_fs

        storage.remove_file("s3://bucket/key.parquet")
        mock_fs.rm.assert_called_with("s3://bucket/key.parquet")


# ---------------------------------------------------------------------------
# read_json / write_json
# ---------------------------------------------------------------------------


class TestJsonIO:
    def test_write_and_read_json(self, tmp_path):
        fpath = str(tmp_path / "data.json")
        data = {"key": "value", "count": 42}

        storage.write_json(fpath, data)
        result = storage.read_json(fpath)
        assert result == data

    def test_write_and_read_json_with_indent(self, tmp_path):
        fpath = str(tmp_path / "data.json")
        data = {"key": "value"}

        storage.write_json(fpath, data, indent=2)
        raw = Path(fpath).read_text()
        assert "\n" in raw  # indented output has newlines
        assert storage.read_json(fpath) == data

    def test_write_and_read_compressed_json(self, tmp_path):
        fpath = str(tmp_path / "data.json.gz")
        data = {"key": "value", "nested": [1, 2, 3]}

        storage.write_json(fpath, data, compressed=True)
        result = storage.read_json(fpath, compressed=True)
        assert result == data

        # Verify it's actually gzip compressed
        with open(fpath, "rb") as f:
            assert f.read(2) == b"\x1f\x8b"  # gzip magic number

    def test_write_json_default_str(self, tmp_path):
        """write_json with default=str handles non-serializable types."""
        fpath = str(tmp_path / "data.json")
        data = {"path": Path("/tmp/test")}

        storage.write_json(fpath, data, default=str)
        result = storage.read_json(fpath)
        assert result["path"] == str(Path("/tmp/test"))

    def test_write_compressed_json_default_str(self, tmp_path):
        """Compressed write_json with default=str also handles non-serializable types."""
        fpath = str(tmp_path / "data.json.gz")
        data = {"path": Path("/tmp/test")}

        storage.write_json(fpath, data, compressed=True, default=str)
        result = storage.read_json(fpath, compressed=True)
        assert result["path"] == str(Path("/tmp/test"))


# ---------------------------------------------------------------------------
# basename
# ---------------------------------------------------------------------------


class TestBasename:
    def test_local_basename(self):
        assert storage.basename("/tmp/output/final/file.parquet") == "file.parquet"

    def test_local_basename_dir(self):
        assert storage.basename("/tmp/output/final") == "final"

    def test_s3_basename(self):
        assert storage.basename("s3://bucket/prefix/file.parquet") == "file.parquet"

    def test_s3_basename_with_trailing_slash(self):
        assert storage.basename("s3://bucket/prefix/dir/") == "dir"

    def test_s3_basename_bucket_only(self):
        assert storage.basename("s3://bucket") == "bucket"


# ---------------------------------------------------------------------------
# validate_parquet
# ---------------------------------------------------------------------------


class TestValidateParquet:
    def test_valid_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = str(tmp_path / "valid.parquet")
        df.to_parquet(path)
        assert storage.validate_parquet(path) is True

    def test_corrupted_file(self, tmp_path):
        path = str(tmp_path / "corrupted.parquet")
        Path(path).write_text("this is not a parquet file")
        assert storage.validate_parquet(path) is False

    def test_truncated_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        valid_path = tmp_path / "valid.parquet"
        df.to_parquet(valid_path)
        data = valid_path.read_bytes()
        truncated_path = str(tmp_path / "truncated.parquet")
        Path(truncated_path).write_bytes(data[: len(data) // 2])
        assert storage.validate_parquet(truncated_path) is False

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.parquet")
        Path(path).write_bytes(b"")
        assert storage.validate_parquet(path) is False

    def test_nonexistent_file(self, tmp_path):
        assert storage.validate_parquet(str(tmp_path / "missing.parquet")) is False


# ---------------------------------------------------------------------------
# write_parquet
# ---------------------------------------------------------------------------


class TestWriteParquet:
    def test_local_write_and_read(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        fpath = str(tmp_path / "test.parquet")
        storage.write_parquet(df, fpath)
        result = pd.read_parquet(fpath)
        assert list(result["a"]) == [1, 2, 3]
        assert list(result["b"]) == ["x", "y", "z"]

    def test_local_write_with_kwargs(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.index.name = "idx"
        fpath = str(tmp_path / "test.parquet")
        storage.write_parquet(df, fpath, index=False)
        result = pd.read_parquet(fpath)
        assert "idx" not in result.columns

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_write_routes_through_fork_safe_fs(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_file = MagicMock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        mock_fs.open.return_value = mock_file
        mock_get_fs.return_value = mock_fs

        df = MagicMock()
        storage.write_parquet(df, "s3://bucket/test.parquet", index=False)
        mock_fs.open.assert_called_with("s3://bucket/test.parquet", "wb")
        df.to_parquet.assert_called_once()
        call_args, call_kwargs = df.to_parquet.call_args
        assert call_args == (mock_file,)
        assert call_kwargs.get("index") is False

    def test_default_compression_is_zstd(self, tmp_path):
        df = pd.DataFrame({"a": list(range(100))})
        fpath = str(tmp_path / "test.parquet")
        storage.write_parquet(df, fpath)
        md = pq.read_metadata(fpath)
        assert md.row_group(0).column(0).compression == "ZSTD"

    def test_explicit_compression_overrides_default(self, tmp_path):
        df = pd.DataFrame({"a": list(range(100))})
        fpath = str(tmp_path / "test.parquet")
        storage.write_parquet(df, fpath, compression="snappy")
        md = pq.read_metadata(fpath)
        assert md.row_group(0).column(0).compression == "SNAPPY"


# ---------------------------------------------------------------------------
# move_file
# ---------------------------------------------------------------------------


class TestMoveFile:
    def test_local_move_replaces_atomically(self, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("hello")
        storage.move_file(str(src), str(dst))
        assert not src.exists()
        assert dst.read_text() == "hello"

    def test_local_move_overwrites_existing_destination(self, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("new")
        dst.write_text("old")
        storage.move_file(str(src), str(dst))
        assert dst.read_text() == "new"

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_move_routes_through_fs_mv(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_get_fs.return_value = mock_fs
        storage.move_file("s3://bucket/src", "s3://bucket/dst")
        mock_fs.mv.assert_called_once_with("s3://bucket/src", "s3://bucket/dst")

    def test_cross_scheme_move_rejected(self, tmp_path):
        local = str(tmp_path / "local.txt")
        Path(local).write_text("x")
        with pytest.raises(ValueError, match="cross-scheme"):
            storage.move_file(local, "s3://bucket/dst")
        with pytest.raises(ValueError, match="cross-scheme"):
            storage.move_file("s3://bucket/src", local)


# ---------------------------------------------------------------------------
# Transient-read resilience (file_exists / validate_parquet / invalidate_cache)
#
# A transient S3 read or a stale s3fs directory listing must not be mistaken for
# a missing/corrupt chunk — that previously aborted the whole export during the
# rollup merge even though the chunk was present and valid on S3.
# ---------------------------------------------------------------------------


class TestFileExistsResilience:
    @patch("nmaipy.storage.time.sleep", lambda *_: None)
    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_retries_transient_then_succeeds(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_fs.exists.side_effect = [OSError("connection reset"), OSError("timeout"), True]
        mock_get_fs.return_value = mock_fs

        assert storage.file_exists("s3://bucket/key.parquet") is True
        assert mock_fs.exists.call_count == 3
        assert mock_fs.invalidate_cache.call_count == 2  # once per retry

    @patch("nmaipy.storage.time.sleep", lambda *_: None)
    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_persistent_error_raises_not_false(self, mock_get_fs):
        # A persistent outage must surface as the real error, never be silently
        # reported as "file absent".
        mock_fs = MagicMock()
        mock_fs.exists.side_effect = OSError("s3 down")
        mock_get_fs.return_value = mock_fs

        with pytest.raises(OSError, match="s3 down"):
            storage.file_exists("s3://bucket/key.parquet")

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_clean_negative_is_not_retried(self, mock_get_fs):
        # A clean False from s3fs is a genuine "absent" — return immediately.
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        mock_get_fs.return_value = mock_fs

        assert storage.file_exists("s3://bucket/missing.parquet") is False
        assert mock_fs.exists.call_count == 1


class TestValidateParquetResilience:
    @patch("nmaipy.storage.time.sleep", lambda *_: None)
    def test_transient_read_retries_then_true(self, tmp_path, monkeypatch):
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = str(tmp_path / "valid.parquet")
        df.to_parquet(path)

        real_open = storage.open_file
        calls = {"n": 0}

        def flaky_open(p, mode="r", **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("transient read")
            return real_open(p, mode, **kw)

        monkeypatch.setattr(storage, "open_file", flaky_open)
        assert storage.validate_parquet(path) is True
        assert calls["n"] == 2  # failed once, succeeded on retry

    @patch("nmaipy.storage.time.sleep", lambda *_: None)
    def test_persistent_transient_returns_false_after_retrying(self, monkeypatch):
        calls = {"n": 0}

        def always_raise(p, mode="r", **kw):
            calls["n"] += 1
            raise OSError("s3 unreadable")

        monkeypatch.setattr(storage, "open_file", always_raise)
        # Persistent transient error → False, but only after exhausting retries
        # (not silently on the first read).
        assert storage.validate_parquet("s3://bucket/x.parquet") is False
        assert calls["n"] == storage._S3_READ_RETRIES + 1

    def test_corruption_is_not_retried(self, tmp_path, monkeypatch):
        path = str(tmp_path / "corrupt.parquet")
        Path(path).write_text("not a parquet")
        calls = {"n": 0}
        real_open = storage.open_file

        def counting_open(p, mode="r", **kw):
            calls["n"] += 1
            return real_open(p, mode, **kw)

        monkeypatch.setattr(storage, "open_file", counting_open)
        assert storage.validate_parquet(path) is False
        assert calls["n"] == 1  # ArrowInvalid → no retry


class TestInvalidateCache:
    def test_local_is_noop(self):
        with patch("nmaipy.storage._get_s3_filesystem") as mock_get_fs:
            storage.invalidate_cache("/tmp/some/dir")
            mock_get_fs.assert_not_called()

    @patch("nmaipy.storage._get_s3_filesystem")
    def test_s3_invalidates(self, mock_get_fs):
        mock_fs = MagicMock()
        mock_get_fs.return_value = mock_fs
        storage.invalidate_cache("s3://bucket/dir")
        mock_fs.invalidate_cache.assert_called_once_with("s3://bucket/dir")
