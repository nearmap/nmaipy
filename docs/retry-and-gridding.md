# Backoff/retry and gridding strategy

A walkthrough of how `nmaipy` handles transient HTTP errors, oversized
requests, and per-parcel CPU-budget skips when talking to the Nearmap AI
Feature API. Written for an engineer who's about to build something
similar against another geospatial API and wants the design rationale,
not just the code.

The goal of this document is not to be a reference for the codebase —
the source is. It's to explain *why* each piece is the shape it is, and
the failure modes each piece is defending against. Several of these
design choices are reactions to specific production incidents; they look
over-engineered at first glance and earn their keep on workloads of
~5–50 million parcels.

---

## The big picture

A single high-level call (`FeatureApi.get_features_gdf(geometry, …)`)
sits on top of several defensive layers. From the outside in:

```
┌──────────────────────────────────────────────────────────────┐
│ Proactive gridding (size-based)                              │
│   "AOI > 1 km² before we even try → split into ~500m cells"  │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Single-request attempt                                       │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ urllib3 Retry — full-jitter exponential backoff        │ │
│   │   429/500/502/503 + connection errors + timeouts       │ │
│   │   max 10 retries, base 0.5s, capped at 60s             │ │
│   └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Reactive gridding — size error                               │
│   413 / 504 → discard, split into grid, query each cell      │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Reactive gridding — include skip                             │
│   any `*_skipped: true` column → discard, split, requery     │
│   (CPU budget exceeded per-parcel; sub-parcels fit)          │
└──────────────────────────────────────────────────────────────┘
```

Each layer is independent: an AOI can hit zero, one, or several of these
in a single call. The whole thing is then wrapped in a per-AOI
`process_chunk` running across a `ProcessPoolExecutor` with chunked
parallelism — orthogonal to this document.

---

## Layer 1: per-request retry (urllib3 + full-jitter backoff)

Everything starts with a single HTTP request. The first defence is a
custom `urllib3.Retry` subclass that retries with **full-jitter
exponential backoff** plus broad connection-error coverage.

### Status codes that retry

```python
# Base set — retried in normal operation
RETRY_STATUS_CODES_BASE = [
    429,  # rate-limited
    502,  # bad gateway
    503,  # service unavailable
    500,  # internal server error
]

# Extended set — used when we're already in a gridding sub-query
RETRY_STATUS_CODES_WITH_TIMEOUT = RETRY_STATUS_CODES_BASE + [504]
```

The dual set is deliberate. At the top level a `504 Gateway Timeout` is
a *signal* that the parcel was too big — we want to surface it as
`APIRequestSizeError` and trigger reactive gridding (see Layer 3). But
once we're inside a gridded sub-query, a 504 is just a transient
problem worth retrying — there's no smaller grid to drop to.
([feature_api.py:138-146](../nmaipy/feature_api.py#L138-L146))

### Backoff formula

Full jitter, scaled by retry count:

```python
sleep = uniform(0, min(BACKOFF_MAX, BACKOFF_FACTOR * 2^(n-1)))
```

Where `BACKOFF_FACTOR = 0.5s`, `BACKOFF_MAX = 60s`, `MAX_RETRIES = 10`.
([api_common.py:214-225](../nmaipy/api_common.py#L214-L225),
[constants.py:127-138](../nmaipy/constants.py#L127-L138))

**Why full jitter and not the usual exponential-with-small-jitter?**
Because we run hundreds of parallel workers and they all see the same
upstream incident at the same time. Without jitter, all 200+ workers
sleep for *exactly* `backoff_factor * 2^n` and then hit the API
simultaneously — a thundering herd that often re-triggers the same 503.
Full jitter desynchronises the retry storm; the trade-off is slightly
slower individual recovery for materially better aggregate behaviour.

The jitter also *scales with retry count*: early retries (n=1,2) have
a small jitter window (~0.5–1s), recovering fast from genuinely transient
issues. Late retries (n=8,9,10) jitter across the full 0–60s, spreading
load across the whole minute — this is when you actually need
desynchronisation, because you're persisting through a real outage.

### Connection errors retry too

`urllib3.Retry` only retries HTTP status codes by default. We extend it
to also retry on the connection-level exceptions that show up under high
concurrency:

```python
ChunkedEncodingError, ConnectionError, ReadTimeout, RemoteDisconnected,
ProxyError, SSLError (both requests and urllib3 flavours), Timeout,
ProtocolError, EOFError, ConnectionResetError, ssl.SSLEOFError
```

([api_common.py:197-212](../nmaipy/api_common.py#L197-L212))

These all reduce to "the connection died mid-flight." A naive
implementation would let them propagate as exceptions and the export
would crash 30 minutes in. Treating them as retryable transients
recovers them at the cost of one extra request.

### Per-request timeouts

```python
TIMEOUT_SECONDS = 120        # connect timeout
READ_TIMEOUT_SECONDS = 90    # read timeout
```

The asymmetry is intentional: we accept that a connect can be slow under
load, but a read shouldn't be — if the server has accepted the request
but isn't streaming bytes, it's probably stuck. We surface
`READ_TIMEOUT_SECONDS` as a synthetic 504 and let the gridding layer
decide whether to retry or split.

### Connection pooling

Per-thread `HTTPAdapter` with `pool_maxsize = min(max(threads, 10), 50)`,
keyed by retry config so different policies get separate pools.
([api_common.py:617-637](../nmaipy/api_common.py#L617-L637))

Two things you can mess up here:

- **Shared `fsspec` / `boto3` clients across forked processes.** They're
  not fork-safe; we recreate per PID via `_get_s3_filesystem()`.
- **Adapter pool exhaustion under burst load.** boto3 defaults to 10
  pool connections; we bump to 50 to match a 32-process × 15-thread
  runner shape.

---

## Layer 2: proactive gridding (size-based)

Before any request fires, `should_grid_aoi()` projects the geometry to
the right area-CRS for the region and compares against a threshold:

```python
# constants.py
MAX_AOI_AREA_SQM_BEFORE_GRIDDING = 1_000_000  # 1 km²
GRID_SIZE_DEGREES = 0.005  # ~500m at equator → ~250k m² per cell
```

If `area > MAX_AOI_AREA_SQM_BEFORE_GRIDDING`, we skip the single-request
attempt entirely and go straight to gridding. The "client threshold"
language in the log message is deliberate — this is *our* threshold,
chosen conservatively after backend issues at 25 km², not a published
API limit.
([feature_api.py:1216-1232](../nmaipy/feature_api.py#L1216-L1232))

### Why proactive at all?

If reactive gridding (Layer 3) can recover from oversized errors anyway,
why grid proactively?

1. **Wasted request cost.** A 1.5 km² parcel that we know will return
   504 doesn't need to make the call first. The proactive path saves
   one round trip and the server's wasted CPU.
2. **Latency.** The 504 response time is dominated by the server
   discovering it can't compute in budget — often 60+ seconds. Going
   straight to grid is faster.
3. **Cleaner failure mode under high concurrency.** When 50 workers
   simultaneously fire requests that will all 504, the resulting retry
   storm + reactive grid storm is much worse than just gridding up
   front.

---

## Layer 3: reactive gridding — size error

If a request slips past the proactive check (because it was under 1 km²
but still too complex for the backend), we get back a
`413 Request Entity Too Large` or `504 Gateway Timeout`. Both are mapped
to `APIRequestSizeError` and trigger the reactive grid path.
([feature_api.py:1286-1317](../nmaipy/feature_api.py#L1286-L1317))

### Recursion guard

The single most important thing in this layer:

```python
if fail_hard_regrid or geometry is None:
    # Do not re-grid. Surface the error.
```

Gridded sub-queries set `fail_hard_regrid=True` when they call back
into the same function. Without this, a sub-cell that itself returns
413/504 would trigger another grid → another set of sub-cells → infinite
recursion that ends with the process dying after the entire AOI has been
queried at 10 m² resolution.

The `in_gridding_mode` flag does the same job in a slightly different
context (suppressing reactive *skip* gridding inside an already-gridded
query). The two flags exist because they're set in different paths and
suppress different downstream behaviour, but the principle is the same:
**always set a "we're already at the smallest unit we're going to try"
flag when recursing**, and never let downstream code re-grid past it.

### What gets passed to the grid?

The sub-cells are queried with **parcel mode disabled**
(`disable_parcel_mode=True`). This matters: parcel mode tells the
backend to clip results to the parcel boundary and compute aggregate
include parameters at the parcel level. When you're querying a 500 m × 500 m
sub-cell of a parcel, the parcel-level aggregations are meaningless,
and clipping at the grid boundary would corrupt per-roof scores. We
warn loudly and force parcel mode off for sub-queries.
([feature_api.py:1440-1447](../nmaipy/feature_api.py#L1440-L1447))

### Concurrency control

Gridded queries fan out into N sub-requests where N can be 50+. With
the main pool already running 10–20 threads per process, an unbounded
grid fan-out can saturate the backend or our own connection pool. A
semaphore caps concurrent gridding *operations* per process:

```python
max_concurrent_gridding = max(1, threads // 5)  # ~2-4 at typical thread counts
```

([api_common.py:951-953](../nmaipy/api_common.py#L951-L953))

The semaphore is acquired before the grid fan-out and released after,
so the *sub-requests* within one grid still all run in parallel —
we're only limiting how many *separate AOIs* can be in gridded state
at once.

### Result combination

Sub-cell responses are concatenated and then `combine_features_from_grid`
deduplicates and merges:

1. **Drop exact duplicates** (`feature_id, geometry` pairs) — discrete
   features (buildings, pools) that happened to fall in multiple cells.
2. **Dissolve by `feature_id`** — connected features (vegetation,
   surfaces) split across grid lines get their geometry merged.
3. **Sum clipped-area columns** — areas need re-summing across the
   merged pieces, not "first value wins."
4. **For other columns**, the value from the **largest area portion**
   wins. Surfaces a warning if any column has conflicting values
   across cells.

([geometry_utils.py:192-303](../nmaipy/geometry_utils.py#L192-L303))

Of these, (3) and (4) are the subtle ones — they're easy to get wrong
and the test suite will pass without exercising them. We learned (3)
the hard way after roof area totals on gridded parcels came in
mysteriously low.

---

## Layer 4: reactive gridding — include skip

This is the newest layer (v5.0.14) and the most defensible-in-isolation.

### The problem

The Feature API can return a `{"skipped": true}` block for individual
include sections (`hurricaneScore`, `defensibleSpace`,
`roofSpotlightIndex`, etc.) when computing them would exceed the
per-parcel CPU budget. The response itself succeeds — the parcel's
shape and basic feature data come back fine — but the include columns
arrive as null. From the client side, you get a 200 OK with quietly
incomplete data.

This is genuinely insidious. A naive consumer doesn't notice anything
is wrong; they just see null peril scores on certain (usually large or
feature-dense) parcels.

### Detection

A flat scan for any column ending in `_skipped` set to True:

```python
def _response_has_include_skips(features_gdf):
    skipped = []
    if features_gdf is not None and len(features_gdf) > 0:
        for col in features_gdf.columns:
            if not col.endswith("_skipped"):
                continue
            if (features_gdf[col] == True).any():
                skipped.append(col)
    return (len(skipped) > 0, skipped)
```

Duck-typed by column name. We considered an explicit whitelist of known
skip columns; rejected because the API is still adding include types
and a stale whitelist is a maintenance bug we'd inevitably miss.
False-positive risk is low — any future API addition introducing an
unrelated `*_skipped` column would have to be a real concern for our
code anyway.

### Trigger conditions

```python
if not in_gridding_mode and geometry is not None:
    any_skipped, skipped_includes = _response_has_include_skips(features_gdf)
    if any_skipped:
        if self.regrid_on_skip and not fail_hard_regrid:
            # Discard response, attempt grid
            features_gdf, metadata, error, grid_errors_df = self._attempt_gridding(
                geometry=geometry, ...,
                reason="reactive - include skip",
            )
        else:
            # Operator opted out — surface the skip at INFO
            logger.info(f"AOI (id {aoi_id}): include skip detected …")
```

([feature_api.py:1260-1285](../nmaipy/feature_api.py#L1260-L1285))

Four guards:
- `regrid_on_skip` — operator switch, default `True`.
- `not in_gridding_mode` — already gridding, don't recurse.
- `not fail_hard_regrid` — caller has opted out.
- `geometry is not None` — can't grid without a geometry.

If we don't regrid (operator opt-out or `fail_hard_regrid`), we still
log the skip at INFO so the operator who explicitly disabled regridding
sees what the API dropped. The regrid path itself logs at DEBUG —
when regrid is on (the default), the regrid trigger isn't noteworthy
per-AOI; surfacing it at INFO would flood logs at 10k+ chunks.

### Why does it work?

The API's CPU budget is per-parcel. When the parcel is split into 500m
sub-cells with parcel mode disabled, each sub-cell is a separate query
that the API sees as its own mini-parcel — each one comfortably under
the per-parcel budget. The include scores come back populated.

### What it can't recover

Aggregate (parcel-level) defensible space — the field is computed only
in parcel mode, and we disable parcel mode for sub-queries. Per-roof
DS is recovered; the parcel-aggregate DS stays null. We don't try to
client-side reconstruct it; that would conflate "API computed it" with
"we estimated it" in a way that's hard to detect downstream.

---

## How a request actually flows

A real call path for an oversized, complex parcel:

```
get_features_gdf(geometry)
    │
    ├─ should_grid_aoi? → True (area > 1 km²)
    │     │
    │     └─ _attempt_gridding(reason="proactive - area …")
    │           │
    │           ├─ split into 500m cells → 6 sub-queries
    │           ├─ each sub-query: own urllib3 Retry, can 429/503 retry
    │           ├─ combine_features_from_grid → dedupe + dissolve
    │           └─ return
    │
    └─ (if proactive didn't fire)
       single request → 200 OK
            │
            ├─ _response_has_include_skips? → True
            │     │
            │     └─ _attempt_gridding(reason="reactive - include skip")
            │           │ (same as above)
            │           └─ return populated scores
            │
            └─ (if no skips) return as-is
```

`AIFeatureAPIRequestSizeError` is the third entry to `_attempt_gridding`
— the catch clause for `413/504` raised mid-request.
([feature_api.py:1218-1232 (proactive), 1268-1280 (skip), 1305-1317 (size error)](../nmaipy/feature_api.py))

Each entry passes a `reason` string that ends up in the log; this is
load-bearing for debugging — when something's wasted requests, you want
to know *why* the grid fired.

---

## How memory scales: `--processes` and `--chunk-size`

The retry/gridding machinery above sits inside two operator-facing
dials that together set the export's memory footprint. Both stages
of the pipeline — chunk-fetch (during processing) and chunk-stream
(during closeout) — scale linearly with `--processes`:

| Stage | Peak memory |
|---|---|
| Chunk processing (concurrent) | `processes × per-chunk feature table` |
| Closeout streaming (consolidation) | `round(1.5 × processes) × per-chunk feature table` (= `_resolve_prefetch_workers(processes)` prefetched tables resident at once) |

`--chunk-size` is the orthogonal dial — it sets how many AOIs each
worker handles per chunk, which is what makes each "per-chunk feature
table" big or small. Cutting chunk size in half cuts both stages'
per-table footprint in half at the cost of more chunk-boundary
overhead and more files to combine at the end.

For an engineer porting this design, the operational consequence is
that there is no separate `--prefetch-workers` knob — `--processes`
is the single dial that bounds RAM at both stages. We tried adding
an override for the prefetch buffer and rejected it because it
diluted `--processes` as the source of truth (the discussion is in
PR #200's description if you want the long-form rationale).

A subtlety worth knowing if you copy this pattern: the prefetch
buffer scales with `--processes` even though the streaming stage
runs in the *main* process after the worker pool has exited. The
coupling assumes "if you picked a high process count, you also
picked a host with enough RAM for the closeout buffer" — which is
true on typical AWS general-purpose instances but isn't enforced.
If your hosting model has skewed CPU:RAM ratios, add an upper cap or
detach the multiplier.

Memory is monitored as the working set
(`memory.current − inactive_file`), matching kubelet and the kernel
OOM killer. Don't use raw `memory.current` for an OOM-risk figure —
on file-heavy workloads it overstates resident memory by 2-3×
because of reclaimable page cache. See `nmaipy/cgroup_memory.py`.

---

## Pitfalls to avoid (if you're building something similar)

1. **Forget the recursion guard at your peril.** A reactive-grid loop
   that doesn't suppress nested grids will eventually consume your
   entire request budget on a single weird parcel. Make the
   "already gridding" flag part of the function signature, not a global
   or instance variable.

2. **Decide proactive thresholds based on production observation, not
   spec.** Our 1 km² threshold isn't what the API documents as its
   limit; it's what we observed empirically produces stable responses.
   Document the observation in code so a future tuner doesn't bump it
   based on the published number alone.

3. **Test the dissolve/sum logic on connected features.** A test suite
   that uses one parcel × one survey × discrete features will pass
   completely while the gridded vegetation totals come back at 70% of
   true area. Cross-grid-boundary dissolve is the silent failure mode.

4. **Log the regrid `reason`.** When wasted requests show up in
   production, you need to know whether they were size-error-driven,
   skip-driven, or proactive — without that, the trace is unreadable.

5. **Resist the urge to "fix" full jitter to be less random.** Operators
   *will* push back saying "why is this 30s sleep so variable, can we
   make it deterministic?" The variability is the entire point; the
   moment you make late retries deterministic you've reintroduced the
   thundering herd.

6. **Per-thread connection pools, fork-safe boto3/fsspec.** Every
   geospatial pipeline written in Python eventually hits this. Plan
   for `ProcessPoolExecutor` from day one, not "we'll figure it out
   later."

7. **Set a connect/read timeout always.** A missing timeout on a hung
   socket blocks a worker thread forever. We learned this twice — once
   in the main feature API client (set early, easy fix) and once in
   the legacy `coverage_utils` module (set late, after observing
   threaded usage actually hang). Default timeouts are not a thing in
   `requests.Session`.

8. **Discard the partial response on reactive regrid.** Don't try to
   merge "the parcel-mode response with skips" + "the gridded response
   without skips." The shape and clipping semantics are different and
   you'll spend a week chasing edge cases. Throw it away and re-query.

---

## Constants reference (current values)

For the engineer porting this to their own codebase, these are the
specific values we've settled on after a year of production load:

| Constant | Value | Where | Rationale |
|---|---|---|---|
| `MAX_RETRIES` | 10 | constants.py | Mirrors S3 transient-error rates under high concurrency; total worst-case wait at full backoff is ~10 min |
| `BACKOFF_FACTOR` | 0.5 | constants.py | Small enough that the first 2-3 retries recover fast on truly transient errors |
| `BACKOFF_MAX` | 60 | constants.py | Caps the maximum sleep; longer than this and the operator starts noticing |
| `TIMEOUT_SECONDS` | 120 | constants.py | Connect timeout — generous under load |
| `READ_TIMEOUT_SECONDS` | 90 | constants.py | Read timeout — should not exceed the server's per-parcel CPU budget |
| `MAX_AOI_AREA_SQM_BEFORE_GRIDDING` | 1,000,000 | constants.py | Conservative; the API can do larger but the failure mode at the limit is hard to recover from |
| `GRID_SIZE_DEGREES` | 0.005 | constants.py | ~500m at equator, ~200-300k m² per cell depending on latitude |
| `max_concurrent_gridding` | `threads // 5` | api_common.py:951 | Bounds the per-process grid fan-out without throttling the typical case |
| `regrid_on_skip` | `True` | feature_api.py:173 | Default-on; opt-out for callers that prefer null over more requests |

---

## Where to look in the code

- **Retry / backoff:** [`nmaipy/api_common.py`](../nmaipy/api_common.py) — `RetryRequest`, `_session_scope`
- **Proactive gridding:** [`nmaipy/feature_api.py:1216`](../nmaipy/feature_api.py#L1216), `GriddedApiClient.should_grid_aoi` in [`api_common.py:956`](../nmaipy/api_common.py#L956)
- **Reactive gridding (size):** [`feature_api.py:1286`](../nmaipy/feature_api.py#L1286)
- **Reactive gridding (skip):** [`feature_api.py:1260`](../nmaipy/feature_api.py#L1260)
- **Grid execution + combine:** `_attempt_gridding` in [`feature_api.py:1005`](../nmaipy/feature_api.py#L1005), `combine_features_from_grid` in [`geometry_utils.py:192`](../nmaipy/geometry_utils.py#L192)
- **Constants:** [`nmaipy/constants.py`](../nmaipy/constants.py)
