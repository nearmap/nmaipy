"""
Tests for the storage abstraction module.

Tests cover both local filesystem operations (using tmp_path) and S3 operations
(mocking fsspec/s3fs to avoid requiring real AWS credentials).
"""

import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

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

    @patch("nmaipy.storage.fsspec")
    def test_s3_exists(self, mock_fsspec):
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        mock_fsspec.filesystem.return_value = mock_fs

        assert storage.file_exists("s3://bucket/key.parquet") is True
        mock_fsspec.filesystem.assert_called_with("s3")
        mock_fs.exists.assert_called_with("s3://bucket/key.parquet")

    @patch("nmaipy.storage.fsspec")
    def test_s3_not_exists(self, mock_fsspec):
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        mock_fsspec.filesystem.return_value = mock_fs

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

    @patch("nmaipy.storage.fsspec")
    def test_s3_glob(self, mock_fsspec):
        mock_fs = MagicMock()
        mock_fs.glob.return_value = [
            "bucket/prefix/chunks/rollup_001.parquet",
            "bucket/prefix/chunks/rollup_002.parquet",
        ]
        mock_fsspec.filesystem.return_value = mock_fs

        results = storage.glob_files("s3://bucket/prefix/chunks", "rollup_*.parquet")
        assert results == [
            "s3://bucket/prefix/chunks/rollup_001.parquet",
            "s3://bucket/prefix/chunks/rollup_002.parquet",
        ]
        mock_fs.glob.assert_called_with("s3://bucket/prefix/chunks/rollup_*.parquet")


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

    @patch("nmaipy.storage.fsspec")
    def test_s3_open(self, mock_fsspec):
        mock_file = MagicMock()
        mock_open_obj = MagicMock()
        mock_open_obj.open.return_value = mock_file
        mock_fsspec.open.return_value = mock_open_obj

        result = storage.open_file("s3://bucket/key.txt", "r")
        mock_fsspec.open.assert_called_with("s3://bucket/key.txt", "r")
        assert result == mock_file


# ---------------------------------------------------------------------------
# file_size
# ---------------------------------------------------------------------------


class TestFileSize:
    def test_local_file_size(self, tmp_path):
        fpath = tmp_path / "test.txt"
        content = "hello world"
        fpath.write_text(content)
        assert storage.file_size(str(fpath)) == len(content.encode())

    @patch("nmaipy.storage.fsspec")
    def test_s3_file_size(self, mock_fsspec):
        mock_fs = MagicMock()
        mock_fs.size.return_value = 1024
        mock_fsspec.filesystem.return_value = mock_fs

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

    @patch("nmaipy.storage.fsspec")
    def test_upload_to_s3(self, mock_fsspec):
        mock_fs = MagicMock()
        mock_fsspec.filesystem.return_value = mock_fs

        storage.upload_file("/tmp/local.parquet", "s3://bucket/key.parquet")
        mock_fsspec.filesystem.assert_called_with("s3")
        mock_fs.put.assert_called_with("/tmp/local.parquet", "s3://bucket/key.parquet")


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
        """write_json uses default=str for non-serializable types."""
        fpath = str(tmp_path / "data.json")
        data = {"path": Path("/tmp/test")}

        storage.write_json(fpath, data)
        result = storage.read_json(fpath)
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
