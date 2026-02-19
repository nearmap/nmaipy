"""
Storage abstraction for local and S3 filesystem operations.

Provides a thin layer over local Path operations and fsspec/s3fs so that
the exporter pipeline can write to either local directories or S3 URIs
without changing calling code.

S3 detection is implicit: any path starting with "s3://" is treated as S3.
AWS credentials are picked up automatically by s3fs from environment variables
or ~/.aws/credentials.
"""

import functools
import gzip
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fsspec


@functools.lru_cache(maxsize=1)
def _get_s3_filesystem():
    """Return a cached S3 filesystem instance to avoid repeated instantiation."""
    return fsspec.filesystem("s3")


def is_s3_path(path: Union[str, Path]) -> bool:
    """Check if a path is an S3 URI."""
    return str(path).startswith("s3://")


def join_path(base: str, *parts: str) -> str:
    """
    Join path components. Works for both local paths and S3 URIs.

    Args:
        base: Base path (local or S3 URI)
        *parts: Path components to append

    Returns:
        Joined path as string
    """
    if is_s3_path(base):
        result = base.rstrip("/")
        for part in parts:
            result = result + "/" + part.strip("/")
        return result
    else:
        return str(Path(base).joinpath(*parts))


def ensure_directory(path: str) -> None:
    """
    Create directory if local. No-op for S3 (directories are virtual).

    Args:
        path: Directory path to create
    """
    if not is_s3_path(path):
        Path(path).mkdir(parents=True, exist_ok=True)


def file_exists(path: str) -> bool:
    """
    Check if file exists. Works for both local and S3.

    Args:
        path: File path to check

    Returns:
        True if the file exists
    """
    if is_s3_path(path):
        fs = _get_s3_filesystem()
        return fs.exists(path)
    else:
        return Path(path).exists()


def glob_files(directory: str, pattern: str) -> List[str]:
    """
    Glob for files in a directory. Works for both local and S3.

    Args:
        directory: Directory to search in
        pattern: Glob pattern (e.g. "rollup_*.parquet")

    Returns:
        List of matching file paths as strings
    """
    if is_s3_path(directory):
        fs = _get_s3_filesystem()
        full_pattern = join_path(directory, pattern)
        # fsspec glob returns paths without the s3:// prefix
        results = fs.glob(full_pattern)
        return ["s3://" + r for r in results]
    else:
        return [str(p) for p in Path(directory).glob(pattern)]


def open_file(path: str, mode: str = "r", **kwargs):
    """
    Open a file for reading or writing. Works for both local and S3.

    Returns a context manager that properly handles cleanup for both local
    files and S3 (via fsspec's OpenFile wrapper).

    Args:
        path: File path to open
        mode: File mode (e.g. 'r', 'w', 'rb', 'wb')
        **kwargs: Additional arguments passed to open/fsspec.open

    Returns:
        File-like context manager
    """
    if is_s3_path(path):
        # Return the fsspec OpenFile directly — it is a context manager that
        # opens the underlying file on __enter__ and flushes/closes on __exit__.
        return fsspec.open(path, mode, **kwargs)
    else:
        return open(path, mode, **kwargs)


def file_size(path: str) -> int:
    """
    Get file size in bytes.

    Args:
        path: File path

    Returns:
        File size in bytes
    """
    if is_s3_path(path):
        fs = _get_s3_filesystem()
        return fs.size(path)
    else:
        return Path(path).stat().st_size


def upload_file(local_path: str, remote_path: str) -> None:
    """
    Upload a local file to a remote path. If remote_path is local,
    copies the file instead.

    Args:
        local_path: Source file on local filesystem
        remote_path: Destination path (local or S3)
    """
    if is_s3_path(remote_path):
        fs = _get_s3_filesystem()
        fs.put(str(local_path), remote_path)
    else:
        if str(local_path) != str(remote_path):
            shutil.copy2(str(local_path), str(remote_path))


def remove_file(path: str) -> None:
    """
    Remove a file. Works for both local and S3. Silently ignores missing files.

    Args:
        path: File path to remove
    """
    try:
        if is_s3_path(path):
            fs = _get_s3_filesystem()
            fs.rm(path)
        else:
            os.remove(path)
    except (OSError, FileNotFoundError):
        pass


def read_json(path: str, compressed: bool = False) -> Optional[Dict]:
    """
    Read a JSON file, optionally gzip-compressed. Works for both local and S3.

    Args:
        path: File path to read
        compressed: If True, read as gzip-compressed JSON

    Returns:
        Parsed JSON data, or None on failure
    """
    if compressed:
        with open_file(path, "rb") as raw_f:
            with gzip.GzipFile(fileobj=raw_f) as gz_f:
                return json.loads(gz_f.read().decode("utf-8"))
    else:
        with open_file(path, "r") as f:
            return json.load(f)


def write_json(path: str, data: Any, compressed: bool = False, indent: Optional[int] = None) -> None:
    """
    Write data as JSON, optionally gzip-compressed. Works for both local and S3.

    Args:
        path: File path to write
        data: Data to serialize as JSON
        compressed: If True, write as gzip-compressed JSON
        indent: JSON indentation level (None for compact)
    """
    if compressed:
        with open_file(path, "wb") as raw_f:
            with gzip.GzipFile(fileobj=raw_f, mode="wb") as gz_f:
                gz_f.write(json.dumps(data, default=str).encode("utf-8"))
    else:
        with open_file(path, "w") as f:
            json.dump(data, f, indent=indent, default=str)


def basename(path: str) -> str:
    """
    Get the filename component of a path. Works for both local and S3.

    Args:
        path: File path

    Returns:
        Filename component
    """
    if is_s3_path(path):
        return path.rstrip("/").rsplit("/", 1)[-1]
    else:
        return Path(path).name
