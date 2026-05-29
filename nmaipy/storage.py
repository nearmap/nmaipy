"""
Storage abstraction for local and S3 filesystem operations.

Provides a thin layer over local Path operations and fsspec/s3fs so that
the exporter pipeline can write to either local directories or S3 URIs
without changing calling code.

S3 detection is implicit: any path starting with "s3://" is treated as S3.
AWS credentials are picked up automatically by s3fs from environment variables
or ~/.aws/credentials.
"""

import gzip
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)

# Bounded retry for transient S3 reads. file_exists / validate_parquet are used
# both to skip already-processed chunks and to gather chunks during the final
# merge; a transient read or a stale s3fs directory listing must not be mistaken
# for a missing/corrupt chunk (which aborts the whole export — see
# AOIExporter rollup consolidation).
# Total worst-case wait: ~3.5s per call (0.5 + 1.0 + 2.0 — exponential backoff).
_S3_READ_RETRIES = 3
_S3_READ_BACKOFF_SECONDS = 0.5

# Tuned transfer config for large files (>1GB).
# Default boto3 uses 8MB parts / 10 threads — too conservative for multi-GB files.
# 64MB parts with 25 threads saturates typical EC2-to-S3 bandwidth much better.
_LARGE_FILE_THRESHOLD = 1 * 1024**3  # 1 GB
_LARGE_FILE_TRANSFER_CONFIG = TransferConfig(
    multipart_chunksize=64 * 1024**2,  # 64 MB parts
    max_concurrency=25,  # ~1.6 GB peak buffer — fine on export machines
    multipart_threshold=64 * 1024**2,  # Use multipart above 64 MB
)

_s3_boto3_client_cache = {}
_s3_filesystem_cache = {}


# boto3's default max_pool_connections is 10. The chunk merge phase fans out
# hundreds of parallel S3 reads — under that load we hit the pool limit and
# log a flurry of "Connection pool is full, discarding connection" warnings
# on multi-million-parcel exports. Bump to 50 to match a typical
# 32-process × 15-thread runner shape with headroom. Adaptive retry mode
# smooths transient 503s without changing correctness.
_S3_CLIENT_CONFIG = BotoConfig(
    max_pool_connections=50,
    retries={"max_attempts": 10, "mode": "adaptive"},
)


def _get_s3_boto3_client():
    """Return a per-process cached boto3 S3 client.

    Used for uploads where we need TransferConfig control that s3fs
    doesn't expose.  Same fork-safety pattern as _get_s3_filesystem.
    """
    pid = os.getpid()
    if pid not in _s3_boto3_client_cache:
        _s3_boto3_client_cache[pid] = boto3.client("s3", config=_S3_CLIENT_CONFIG)
    return _s3_boto3_client_cache[pid]


def _get_s3_filesystem():
    """Return a per-process cached S3 filesystem instance.

    Creates a new instance after fork to avoid sharing non-fork-safe S3 clients
    across process boundaries (s3fs/botocore connections cannot survive fork).

    Does NOT clear _s3_filesystem_cache or AbstractFileSystem._cache — doing so
    would drop references to the parent's stale S3FileSystem, triggering __del__
    which tries to close the broken aiobotocore event loop and hangs. The stale
    entries are harmless: our instance uses skip_instance_cache=True, and all S3
    I/O is routed through this function (including write_parquet), so the stale
    cached instances are never accessed.
    """
    pid = os.getpid()
    if pid not in _s3_filesystem_cache:
        _s3_filesystem_cache[pid] = fsspec.filesystem("s3", skip_instance_cache=True)
    return _s3_filesystem_cache[pid]


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


def invalidate_cache(path: str) -> None:
    """Drop any cached s3fs directory listing for ``path``.

    s3fs caches directory listings; a stale or partially-populated listing can
    make an object that exists look absent. Call this before an existence sweep
    that must be authoritative (e.g. the chunk-merge phase). No-op for local
    paths.
    """
    if is_s3_path(path):
        _get_s3_filesystem().invalidate_cache(path)


def file_exists(path: str) -> bool:
    """
    Check if file exists. Works for both local and S3.

    For S3, a transient read error (throttle, timeout, connection reset) is
    retried rather than propagated — only a clean negative from s3fs is taken
    as "absent". This prevents a flaky read from being mistaken for a missing
    file by callers that treat absence as fatal.

    Args:
        path: File path to check

    Returns:
        True if the file exists, False if it does not.

    Raises:
        The underlying S3 error if reads still fail after retries. By design —
        a persistent outage surfaces as itself rather than masquerading as
        "file missing" to callers that proceed on a False return.
    """
    if not is_s3_path(path):
        return Path(path).exists()

    fs = _get_s3_filesystem()
    last_exc: Optional[Exception] = None
    for attempt in range(_S3_READ_RETRIES + 1):
        try:
            return fs.exists(path)
        except Exception as e:  # transient S3/network error — retry
            last_exc = e
            if attempt < _S3_READ_RETRIES:
                time.sleep(_S3_READ_BACKOFF_SECONDS * (2**attempt))
                fs.invalidate_cache(path)
    # Exhausted retries: re-raise rather than silently returning False, so a
    # genuine, persistent S3 outage surfaces as itself instead of "file missing".
    raise last_exc


def validate_parquet(path: str) -> bool:
    """
    Check if a parquet file has a valid footer (schema + row group metadata).

    Opens the file and attempts to construct a ``pyarrow.parquet.ParquetFile``,
    which reads only the footer.  This is a lightweight integrity check that
    detects truncated or corrupted files without reading column data.

    A genuinely corrupt/truncated file (``pyarrow.ArrowInvalid``) returns
    ``False`` immediately. A transient read error (S3 throttle/timeout, stale
    s3fs listing) is retried — it must not be silently reported as corruption,
    because callers treat an invalid chunk as missing and may abort the export.

    Args:
        path: File path (local or S3 URI).

    Returns:
        True if the file has a valid parquet footer, False otherwise.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_S3_READ_RETRIES + 1):
        try:
            with open_file(path, "rb") as f:
                pq.ParquetFile(f)
            return True
        except (pa.ArrowInvalid, FileNotFoundError):
            # Genuinely corrupt/truncated, or genuinely absent — retrying won't help.
            return False
        except Exception as e:  # transient read error — retry before giving up
            last_exc = e
            if attempt < _S3_READ_RETRIES:
                time.sleep(_S3_READ_BACKOFF_SECONDS * (2**attempt))
                if is_s3_path(path):
                    _get_s3_filesystem().invalidate_cache(path)
    logger.warning(
        f"validate_parquet: treating {path} as invalid after "
        f"{_S3_READ_RETRIES + 1} read attempts (last error: {last_exc})"
    )
    return False


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

    For S3, routes through _get_s3_filesystem() to ensure fork-safe connections.
    Returns a file-like context manager in both cases.

    Args:
        path: File path to open
        mode: File mode (e.g. 'r', 'w', 'rb', 'wb')
        **kwargs: Additional arguments passed to open/fs.open

    Returns:
        File-like context manager
    """
    if is_s3_path(path):
        fs = _get_s3_filesystem()
        return fs.open(path, mode, **kwargs)
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


def _parse_s3_uri(uri: str) -> tuple:
    """Split 's3://bucket/key' into (bucket, key)."""
    without_scheme = uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def upload_file(local_path: str, remote_path: str) -> None:
    """
    Upload a local file to a remote path. If remote_path is local,
    copies the file instead.

    For S3 destinations, files above 1 GB use a tuned TransferConfig with
    larger multipart chunks and higher concurrency to saturate network
    bandwidth.  Smaller files use boto3 defaults.

    Args:
        local_path: Source file on local filesystem
        remote_path: Destination path (local or S3)
    """
    if is_s3_path(remote_path):
        bucket, key = _parse_s3_uri(remote_path)
        local_size = Path(local_path).stat().st_size
        config = _LARGE_FILE_TRANSFER_CONFIG if local_size >= _LARGE_FILE_THRESHOLD else None
        _get_s3_boto3_client().upload_file(
            str(local_path),
            bucket,
            key,
            Config=config,
        )
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


def move_file(src: str, dst: str) -> None:
    """
    Move a file. Used for atomic-write patterns (write-to-temp + move).

    On local filesystems, uses os.replace which is atomic on POSIX (and best-
    effort atomic on Windows). On S3, uses s3fs server-side COPY+DELETE — the
    destination becomes visible atomically at the S3 object level, even though
    the operation is two API calls.

    Cross-scheme moves (local <-> S3) are not supported.

    Args:
        src: Source path
        dst: Destination path
    """
    src_is_s3 = is_s3_path(src)
    dst_is_s3 = is_s3_path(dst)
    if src_is_s3 != dst_is_s3:
        raise ValueError(f"move_file does not support cross-scheme moves: src={src!r} dst={dst!r}")
    if src_is_s3:
        fs = _get_s3_filesystem()
        fs.mv(src, dst)
    else:
        os.replace(src, dst)


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
        with open_file(path, "r", encoding="utf-8") as f:
            return json.load(f)


def write_json(
    path: str,
    data: Any,
    compressed: bool = False,
    indent: Optional[int] = None,
    default=None,
) -> None:
    """
    Write data as JSON, optionally gzip-compressed. Works for both local and S3.

    Args:
        path: File path to write
        data: Data to serialize as JSON
        compressed: If True, write as gzip-compressed JSON
        indent: JSON indentation level (None for compact)
        default: Function for non-serializable types (e.g. str). Passed to json.dump/json.dumps.
    """
    if compressed:
        with open_file(path, "wb") as raw_f:
            with gzip.GzipFile(fileobj=raw_f, mode="wb") as gz_f:
                gz_f.write(json.dumps(data, default=default).encode("utf-8"))
    else:
        with open_file(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=default)


def write_parquet(data, path: str, **kwargs) -> None:
    """
    Write a DataFrame, GeoDataFrame, or pyarrow Table to parquet. Fork-safe for S3.

    Routes S3 writes through our fork-safe filesystem to avoid hangs from
    pyarrow's non-fork-safe native S3FileSystem. For local paths, delegates
    directly to the appropriate writer.

    Defaults `compression="zstd"` when no compression is specified. zstd at
    pyarrow's default level produces ~27 % smaller files than snappy at the
    same encoding wall time (the structural ~12 s/1.6 M-row encoding cost
    dominates, the codec CPU is in the noise). The win shows up in S3
    upload time (less data to push) and ongoing storage cost. Callers that
    pass `compression=...` explicitly override this.

    Args:
        data: pandas DataFrame, geopandas GeoDataFrame, or pyarrow Table
        path: Output file path (local or S3 URI)
        **kwargs: Additional arguments passed to the underlying writer
    """
    kwargs.setdefault("compression", "zstd")
    if is_s3_path(path):
        with open_file(path, "wb") as f:
            if isinstance(data, pa.Table):
                pq.write_table(data, f, **kwargs)
            else:
                data.to_parquet(f, **kwargs)
    else:
        if isinstance(data, pa.Table):
            pq.write_table(data, path, **kwargs)
        else:
            data.to_parquet(path, **kwargs)


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
