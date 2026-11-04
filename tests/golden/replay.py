"""Run a golden case offline: every HTTP request the export makes is answered by the synthetic fake API.

This is the only module that knows which HTTP library nmaipy uses.
"""

import functools
import json
import os
import re
import socket
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import requests
import responses

from nmaipy import constants
from nmaipy.exporter import NearmapAIExporter

from . import world
from .fake_api import FakeApi, UnknownRequest, key_paths

GOLDEN_DIR = Path(__file__).parents[1] / "data" / "golden"
CLASS_DESCRIPTIONS = Path(constants.__file__).parent / "class_descriptions.json"
COMMON = dict(
    processes=1,
    threads=1,
    chunk_size=100,
    max_retries=0,
    save_features=True,
    class_level_files=True,
    include_parcel_geometry=True,
)
CASES: dict[str, dict] = {
    "us": dict(
        country="us",
        packs=["building", "building_char", "pool", "roof_char", "roof_cond", "solar", "surfaces", "vegetation"],
        roof_age=True,
        until="2025-03-18",
    ),
    "au": dict(country="au", packs=["building", "roof_char", "vegetation"], since="2025-01-20", until="2025-03-06"),
}

_ANY_URL = re.compile(".*")
_REAL_CONNECT = socket.socket.connect
_ACTIVE_MOCKS: list[responses.RequestsMock] = []


def run_case(case: str, out_dir: Path) -> Path:
    """Export a golden case offline into `out_dir` (markers and API cache under it too) and return its `final/`.

    Fails when the fake was not installed in a worker, when a request had no answer, or when the export rewrote
    `nmaipy/class_descriptions.json`.
    """
    marker_dir = out_dir / "markers"
    marker_dir.mkdir(parents=True)
    aoi_file = write_aois(case, out_dir / "aois.csv")
    with pytest.MonkeyPatch.context() as m, _class_descriptions_unchanged():
        _go_offline(m)
        m.setattr("nmaipy.base_exporter.ProcessPoolExecutor", executor_with(install_fake, case, str(marker_dir)))
        with serving(case, marker_dir):
            try:
                NearmapAIExporter(
                    aoi_file=aoi_file, output_dir=out_dir, cache_dir=out_dir / "cache", **COMMON, **CASES[case]
                ).run()
            finally:
                _assert_fake_complete(marker_dir)
    return out_dir / "final"


def write_aois(case: str, path: Path) -> Path:
    """Write a case's AOIs (including its invalid rows) as the exporter's CSV input."""
    rows = [f'{aoi_id},"{wkt}"' if wkt else f"{aoi_id}," for aoi_id, wkt in world.build(case).aois.items()]
    path.write_text("aoi_id,geometry\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def install_fake(case: str, marker_dir: str) -> None:
    """Answer this process's HTTP requests from the case's fake API for the rest of its life (a pool initializer).

    The key paths of every answer go to `<marker_dir>/<pid>.paths`, every unanswered or failed request and every TCP
    connection attempt to `<pid>.misses` (and raises), and `<pid>.installed` marks that the hook ran.
    """
    _ACTIVE_MOCKS.append(_start_fake(case, Path(marker_dir)))
    socket.socket.connect = _connect_blocker(Path(marker_dir))
    (Path(marker_dir) / f"{os.getpid()}.installed").touch()


@contextmanager
def serving(case: str, marker_dir: Path) -> Iterator[None]:
    """Answer the calling process's HTTP requests from the case's fake API, as `install_fake` does, until exit."""
    mock = _start_fake(case, marker_dir)
    socket.socket.connect = _connect_blocker(marker_dir)
    try:
        yield
    finally:
        socket.socket.connect = _REAL_CONNECT
        mock.stop(allow_assert=False)
        mock.reset()


def executor_with(initializer: Callable[..., None], *initargs: Any) -> Callable[..., ProcessPoolExecutor]:
    """A drop-in `ProcessPoolExecutor` factory whose workers run `initializer(*initargs)` first."""
    return functools.partial(ProcessPoolExecutor, initializer=initializer, initargs=initargs)


def emitted_paths(marker_dir: Path) -> dict[str, set[str]]:
    """The key paths the fake emitted per endpoint, from the `.paths` markers."""
    paths: dict[str, set[str]] = {}
    for marker in marker_dir.glob("*.paths"):
        for line in marker.read_text(encoding="utf-8").splitlines():
            endpoint, path = line.split("\t")
            paths.setdefault(endpoint, set()).add(path)
    return paths


def _start_fake(case: str, marker_dir: Path) -> responses.RequestsMock:
    fake = FakeApi(world.build(case))
    pid = os.getpid()

    def respond(request: requests.PreparedRequest) -> tuple[int, dict[str, str], str]:
        path = request.url.split("?")[0]
        try:
            answer = fake.handle(request.method, request.url, request.body)
            paths = sorted(key_paths(json.loads(answer.text)))
            _append(marker_dir / f"{pid}.paths", "\n".join(f"{answer.endpoint}\t{p}" for p in paths))
        except UnknownRequest as error:
            _append(marker_dir / f"{pid}.misses", f"{request.method} {path}: {error}")
            raise requests.exceptions.ConnectionError(f"The golden fake API has no answer: {error}") from error
        except Exception as error:
            _append(marker_dir / f"{pid}.misses", f"{request.method} {path}: fake raised {error!r}")
            raise
        return answer.status, {"Content-Type": answer.content_type}, answer.text

    mock = responses.RequestsMock(assert_all_requests_are_fired=False)
    for method in (
        responses.GET,
        responses.POST,
        responses.PUT,
        responses.PATCH,
        responses.DELETE,
        responses.HEAD,
        responses.OPTIONS,
    ):
        mock.add_callback(method, _ANY_URL, callback=respond, content_type=None)
    mock.start()
    return mock


def _assert_fake_complete(marker_dir: Path) -> None:
    installed = {int(p.stem) for p in marker_dir.glob("*.installed")} - {os.getpid()}
    assert installed, "fake not installed in the workers"
    misses = sorted({line for p in marker_dir.glob("*.misses") for line in p.read_text().splitlines()})
    assert not misses, "requests the golden fake API has no answer for:\n" + "\n".join(misses)


def _go_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)


@contextmanager
def _class_descriptions_unchanged() -> Iterator[None]:
    original = CLASS_DESCRIPTIONS.read_bytes()
    try:
        yield
    finally:
        changed = CLASS_DESCRIPTIONS.read_bytes() != original
        if changed:
            CLASS_DESCRIPTIONS.write_bytes(original)
    assert not changed, "a golden run rewrote nmaipy/class_descriptions.json"


def _connect_blocker(marker_dir: Path) -> Callable[[socket.socket, Any], None]:
    def connect(self: socket.socket, address: Any) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _append(marker_dir / f"{os.getpid()}.misses", f"socket connect {address}")
            raise ConnectionRefusedError(f"The golden fake API blocks network access (connect to {address})")
        return _REAL_CONNECT(self, address)

    return connect


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
