"""Record and replay the HTTP exchanges behind a golden export.

This is the only module that knows which HTTP library nmaipy uses; recordings on disk are library-independent.
"""

import functools
import hashlib
import json
import os
import re
import shutil
import socket
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import requests
import responses
from requests.adapters import HTTPAdapter

_ANY_URL = re.compile(".*")
_REAL_CONNECT = socket.socket.connect
_REAL_SEND = HTTPAdapter.send
_ACTIVE_MOCKS: list[responses.RequestsMock] = []


def request_key(method: str, url: str, body: bytes | str | None) -> str:
    """Identify a request by method, host and path, query pairs (minus any api key) and parsed JSON body."""
    parts = urlsplit(url)
    return _key(method, parts.netloc + parts.path, _query_pairs(parts.query), json.loads(body) if body else None)


def load_exchanges(case_dir: Path) -> dict[str, dict]:
    """Load every recorded exchange of a case, keyed by `request_key`."""
    exchanges = {}
    for path in sorted((case_dir / "exchanges").glob("*.json")):
        exchange = json.loads(path.read_text(encoding="utf-8"))
        request = exchange["request"]
        parts = urlsplit(request["url"])
        exchanges[_key(request["method"], parts.netloc + parts.path, request["query"], request["body"])] = exchange
    return exchanges


def install_replay(case_dir: str, marker_dir: str) -> None:
    """Serve this process's HTTP requests from a case's recordings for the rest of its life (a pool initializer).

    Every hit is logged to `<marker_dir>/<pid>.used`, every miss to `<pid>.misses` (and raises), and
    `<pid>.installed` marks that the hook ran. Any TCP connection attempt raises.
    """
    _ACTIVE_MOCKS.append(_start_replay(Path(case_dir), Path(marker_dir)))
    socket.socket.connect = _blocked_connect
    (Path(marker_dir) / f"{os.getpid()}.installed").touch()


@contextmanager
def serving(case_dir: Path, marker_dir: Path) -> Iterator[None]:
    """Serve the calling process's HTTP requests from a case's recordings, as `install_replay` does, until exit."""
    mock = _start_replay(case_dir, marker_dir)
    socket.socket.connect = _blocked_connect
    try:
        yield
    finally:
        socket.socket.connect = _REAL_CONNECT
        mock.stop(allow_assert=False)
        mock.reset()


def install_recorder(out_dir: str) -> None:
    """Log every live HTTP exchange of this process to `<out_dir>/<pid>.jsonl` (a pool initializer)."""
    HTTPAdapter.send = _recording_send(Path(out_dir))


@contextmanager
def recording(out_dir: Path) -> Iterator[None]:
    """Log the calling process's live HTTP exchanges, as `install_recorder` does, until exit."""
    HTTPAdapter.send = _recording_send(out_dir)
    try:
        yield
    finally:
        HTTPAdapter.send = _REAL_SEND


def executor_with(initializer: Callable[..., None], *initargs: Any) -> Callable[..., ProcessPoolExecutor]:
    """A drop-in `ProcessPoolExecutor` factory whose workers run `initializer(*initargs)` first."""
    return functools.partial(ProcessPoolExecutor, initializer=initializer, initargs=initargs)


def write_exchanges(record_dir: Path, case_dir: Path, packs: list[str]) -> None:
    """Replace `case_dir/exchanges` with the recorder logs in `record_dir`, one file per distinct request.

    The `packs.json` and `classes.json` catalogues are cut to `packs` and their classes; api keys are dropped.
    """
    records = {}
    for log in sorted(record_dir.glob("*.jsonl")):
        for line in log.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            records[request_key(record["method"], record["url"], record["body"])] = record

    exchanges = {key: _exchange(record) for key, record in records.items()}
    _trim_catalogues(list(exchanges.values()), packs)

    out = case_dir / "exchanges"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for key, exchange in exchanges.items():
        name = hashlib.sha1(key.encode()).hexdigest()[:16]
        text = json.dumps(exchange, indent=1, ensure_ascii=False)
        (out / f"{name}.json").write_text(text + "\n", encoding="utf-8")


def _key(method: str, location: str, query: list[list[str]], body: Any) -> str:
    return json.dumps([method, location, [list(pair) for pair in query], body], sort_keys=True)


def _query_pairs(query: str) -> list[list[str]]:
    return sorted([k, v] for k, v in parse_qsl(query, keep_blank_values=True) if k.lower() != "apikey")


def _render(body: Any) -> str:
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False)


def _start_replay(case_dir: Path, marker_dir: Path) -> responses.RequestsMock:
    exchanges = load_exchanges(case_dir)
    pid = os.getpid()

    def respond(request: requests.PreparedRequest) -> tuple[int, dict[str, str], str]:
        key = request_key(request.method, request.url, request.body)
        exchange = exchanges.get(key)
        _append(marker_dir / f"{pid}.{'used' if exchange else 'misses'}", key)
        if exchange is None:
            raise requests.exceptions.ConnectionError(f"No golden recording for {key}")
        response = exchange["response"]
        text = response["body"] if response["body_text"] else _render(response["body"])
        return response["status"], {"Content-Type": response["content_type"]}, text

    mock = responses.RequestsMock(assert_all_requests_are_fired=False)
    for method in (responses.GET, responses.POST):
        mock.add_callback(method, _ANY_URL, callback=respond, content_type=None)
    mock.start()
    return mock


def _blocked_connect(self: socket.socket, address: Any) -> None:
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise ConnectionRefusedError(f"Golden replay blocks network access (connect to {address})")
    return _REAL_CONNECT(self, address)


def _recording_send(out_dir: Path) -> Callable[..., requests.Response]:
    log = out_dir / f"{os.getpid()}.jsonl"

    def send(self: HTTPAdapter, request: requests.PreparedRequest, *args: Any, **kwargs: Any) -> requests.Response:
        response = _REAL_SEND(self, request, *args, **kwargs)
        body = request.body.decode() if isinstance(request.body, bytes) else request.body
        record = {
            "method": request.method,
            "url": request.url,
            "body": body,
            "status": response.status_code,
            "content_type": response.headers.get("Content-Type"),
            "response": response.text,
        }
        _append(log, json.dumps(record, ensure_ascii=False))
        return response

    return send


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _exchange(record: dict) -> dict:
    parts = urlsplit(record["url"])
    text = record["response"]
    try:
        body = json.loads(text)
        # Only error bodies reach outputs verbatim; success bodies are parsed, so they are kept as JSON.
        body_text = record["status"] >= 400 and _render(body) != text
    except json.JSONDecodeError:
        body, body_text = text, True
    return {
        "request": {
            "method": record["method"],
            "url": f"{parts.scheme}://{parts.netloc}{parts.path}",
            "query": _query_pairs(parts.query),
            "body": json.loads(record["body"]) if record["body"] else None,
        },
        "response": {
            "status": record["status"],
            "content_type": record["content_type"],
            "body": text if body_text else body,
            "body_text": body_text,
        },
    }


def _trim_catalogues(exchanges: list[dict], packs: list[str]) -> None:
    pack_bodies = [e["response"]["body"] for e in exchanges if e["request"]["url"].endswith("/packs.json")]
    class_bodies = [e["response"]["body"] for e in exchanges if e["request"]["url"].endswith("/classes.json")]
    assert pack_bodies, "No packs.json exchange was recorded"
    class_ids = set()
    for body in pack_bodies:
        body["packs"] = [p for p in body["packs"] if p["code"] in packs]
        assert {p["code"] for p in body["packs"]} == set(packs), f"packs.json lacks some of {packs}"
        class_ids |= {c["id"] for p in body["packs"] for c in p["featureClasses"]}
    for body in class_bodies:
        fields = {k for c in body["classes"] for k in c}
        body["classes"] = [c for c in body["classes"] if c["id"] in class_ids]
        # The client builds a frame from this list, so a field only dropped classes had must stay as a column.
        dropped = fields - {k for c in body["classes"] for k in c}
        body["classes"] = [{**c, **dict.fromkeys(sorted(dropped))} for c in body["classes"]]
