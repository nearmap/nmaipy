# Damage Conflation API (ai/damage/v2)

Evidence tags:
- **[API]**: shown by the bundled response `tests/data/test_damage_conflation_milton_response.json` or by the code that handles it.
- **[nmaipy]**: nmaipy's behaviour (`damage_conflation_api.py`, `damage_conflation_exporter.py`, `parcels.conflation_rollup`, `feature_attributes.flatten_conflated_damage_attributes`, `coverage_utils.discover_event`).
- **[inferred]**: not shown directly.

## Requests and event scoping

- [nmaipy] **Endpoint and body.** Requests are POSTs to `…/ai/damage/v2/events/{event_id}/latest.geojson`. The body holds `aoi` or `address`, plus `cursor`/`limit`.
  - The **event is the only scope**: there are no date, survey or system-version parameters.
- [nmaipy] **Pagination.** nmaipy follows `nextCursor` with pages of 1,000 and stops with an error after 2,000 pages. There is **no gridding**: pagination lets one AOI cover an entire event boundary. `nextCursor` is opaque and pinned to the `resourceId` of the first page; it is sent back unchanged with the same body. A `400` with `code: BAD_CURSOR` means the API can no longer honour the cursor (one issued before a deploy, or whose resource is gone): nmaipy restarts that AOI from page 1 once and treats a second one as a failure. A page whose `resourceId` differs from page 1's is rejected rather than merged.
- [nmaipy] **Errors and retries.** An HTTP 200 with an `error` body is treated as an error. 429 and 5xx are retried per page.
  - This API's rate limit is much lower than the Feature API's; the README recommends `threads=1`.
- [nmaipy] **Finding an event id** (`coverage_utils.discover_event(lat, lon)`).
  - It queries the Coverage API at a point, collects the `postCatEventId` survey tags, and picks the event with the newest `postCatEventDate`, warning if others also cover the point.
  - `event_boundary` unions the event's survey footprints, so the boundary can then be split into AOIs.
  - The event id is on the Coverage API's `postCatEventId` survey tag (not `eventId`, as some older documentation says).
- [API, per README] An AOI outside the event footprint returns zero buildings; that is not an error.

## Response

- [API] **Top level:** `resourceId`, `eventUuid`, `eventName`, `modelVersion` and `presentationVersion`, plus `geomatchedAddress` for address queries. nmaipy copies them onto every row.
- [API] **Each feature is a building polygon.**
  - Its `id` becomes nmaipy's `feature_id`; the separate `hilbertId` is kept as its own column.
  - Properties are `areaSqm`, `areaSqft`, `confidence`, `hilbertId`, `damage.event` and `damage.preEvent`.
- [API] **Each damage block** holds:
  - `rating`: `NoDamage`, `Affected`, `Minor`, `Major` or `Destroyed`;
  - `confidence`;
  - `rawRatings`: the five class scores;
  - `classRatios`: a list of `{classId, description, ratio, confidence}` for AI classes over the footprint.

  `preEvent` also has `latestCaptureDate`.
- [API] **Observed in one sample, not guaranteed:**
  - `rating` is the class with the highest raw score.
  - The block's `confidence` equals that score.
  - Raw scores sum to 1.
  - `preEvent` is missing for some buildings, consistent with construction that has no pre-event baseline.
  - A building can be `Affected` both before and after the event.
- [API, per README] **Map-style results.** Every building intersecting the AOI is returned *whole*, including neighbours' buildings that only clip the boundary.

## Flattening and parcel mode

- [nmaipy] **Fixed column set.** Every row gets:
  - `damage_{event|pre_event}_{rating, confidence, raw_<class>, class_ratios (JSON), latest_capture_date}`;
  - the area in the country's unit (`area_sqft` for the US, `area_sqm` elsewhere), `confidence` and `hilbert_id`.

  A missing block gives nulls.
  - [inferred] `damage_event_latest_capture_date` is normally empty, because only the pre-event block carries a capture date.
- [nmaipy] **`--parcel-mode`** (exporter option, off by default) is a *client-side approximation* of the Feature API's parcelMode.
  - It keeps a building if at least 50% of its area lies inside the AOI, **or** its in-AOI footprint is at least 30 sqm, so terraced and townhouse rows survive.
  - **Nothing is clipped**: kept buildings keep their whole area.
  - It filters both the per-building file and the rollup.
  - It differs from the Feature API's server-side parcelMode, which attributes features with `belongsToParcel` and returns clipped geometry for multiparcel features.

## Rollup (`--rollup`, `parcels.conflation_rollup`)

- [nmaipy] **One row per input AOI.** Input columns come first; on a name clash the input column gets an `_input` suffix.
- [nmaipy] **Columns:** `query_succeeded`, `n_buildings` and `n_no_damage`/`n_affected`/`n_minor`/`n_major`/`n_destroyed`, plus the primary building's attributes.
  - The rating counts use **`damage_event_rating` only**.
  - [inferred] So a building with no event rating counts in `n_buildings` but in no `n_*` rating column.
- [nmaipy] **Failed vs empty.** A failed AOI gets null counts and no primary. A successful AOI with no buildings gets zeros.
- [nmaipy] **Primary building** (`--primary-decision`):
  - `largest` picks by **whole-building** `area_sqm`, so on a small parcel a large neighbour can win unless `--parcel-mode` removed it.
  - `optimal` prefers the building containing the AOI's `lat`/`lon`, falling back to largest when those columns are absent.
- [nmaipy] **Buildings across AOIs.** A building straddling two AOIs appears in each with the same `feature_id`, so deduplicate on it when counting across AOIs.

## Outputs, cache, resume

- [nmaipy] **Files:**
  - `damage_buildings.parquet` (all columns) by default; `damage_buildings.csv` (curated, without `class_ratios`) only with `--output-format csv` or `both`;
  - `damage_rollup.parquet` / `.csv` with `--rollup`, following the same format choice;
  - `damage_metadata.csv` (per-AOI query metadata; event fields only for AOIs with at least one building);
  - `damage_errors.csv` (status code and message, written only when errors occur).
- [nmaipy] **Cache.** The cache lives under `damageconflation/<event_id>/`, keyed by AOI or address, with no time component.
  - [inferred] Later runs replay the earlier `latest` conflation until `--overwrite-cache`.
- [nmaipy] **Resume.** A chunk is skipped on resume once its metadata file exists, and that file is written as soon as any AOI in the chunk succeeds. AOIs that failed in a partly successful chunk are therefore not retried on resume; re-run them with a fresh output directory or after clearing `chunks/`.

## How this differs from Feature API damage classification

- The **Feature API** (`building_lifecycle` features, `damage` field) rates one survey's imagery.
  - Its classes are `Undamaged/Affected/Minor/Major/Destroyed`, with 2- and 3-tier groupings.
  - nmaipy's `damage_class` is the class with the highest confidence.
  - Its `ratios` are per-indicator `ratioAbove50PctConf` values (`damage_ratio_<indicator>`).
- The **Damage Conflation API** gives one rating per building per event, chosen by the API, with a pre-event baseline. Its lowest tier is `NoDamage`, not `Undamaged`.
- **`classRatios` (conflation) and `ratios` (Feature API) are different things** despite the similar names.
