# Feature API: queries, gridding, classes, includes and nmaipy outputs

Evidence tags:
- **[API]**: shown by bundled payloads in nmaipy's `tests/data/` or by code that handles the response.
- **[nmaipy]**: nmaipy's own behaviour. The function names let you grep the installed source.
- **[inferred]**: not shown directly.

Geometry, area and parcelMode semantics are in `feature-api-geometry-and-areas.md`.

## Contents
1. Survey and date selection
2. Gridding
3. Packs, classes, system versions
4. Includes (scores)
5. Rollup semantics: counts, primary features, null vs 0 vs N
6. Errors, retries, resume and cache
7. Feature identity

## 1. Survey and date selection

- [nmaipy] **How a request is built.** Each request is a POST whose body holds only the AOI geometry. Every other option is a URL parameter: `since`, `until`, `order`, `packs`, `classes`, `include`, `systemVersionPrefix`/`systemVersion`, `parcelMode`, `3dCoverage`, `rapid`, `excludeTilesWithOcclusion`, `alpha`/`beta`/`prerelease` (`FeatureApi._create_post_request`).
- [nmaipy] **Per-AOI dates.** `since`/`until` are passed through as date strings. Per-row `since`/`until` columns in the AOI file override the bulk values where the row has a string value; empty cells fall back to the bulk value. The exporter rejects non-string (datetime) date columns.
- [nmaipy] **Pinning a survey.** A `survey_resource_id` (per row) uses the `/surveyresources/{id}/features.json` endpoint and silently ignores `since`/`until`.
  - It is the AI resource id, not the Coverage survey id.
  - [API] Resource ids change when a survey's AI is re-run. To pin a survey reproducibly, set `since` = `until` = its date.
- [nmaipy] **`order`** (`earliest`/`latest`) is sent only when set.
- [nmaipy] **System version prefix.** The `FeatureApi` class defaults to `system_version_prefix="gen6-"`, but the exporter CLI's default sends no prefix. So library callers and CLI users can get different system versions by default.
- [nmaipy] **`prefer3d`** asks for 3D coverage first and re-runs AOIs that returned 404 in 2D. A gridded AOI with any 404 cell is redone entirely in 2D. AOIs pinned to a survey resource are not retried.
- [API] **Response metadata.** Each response carries `surveyDate`, `surveyId`, `resourceId`, `systemVersion`, `perspective`, `postcat`, `3dDate` (empty for 2D), `link` and `versions`. Each feature also carries its own `surveyDate` and `meshDate`.
- [nmaipy] **Where those dates land.**
  - Rollup rows get AOI-level `survey_date`, `survey_id`, `survey_resource_id`, `system_version`, `perspective`, `postcat`, `mesh_date` and `link`.
  - Feature and per-class rows keep each feature's own `survey_date`/`mesh_date`.
  - If an input column has the same name, the metadata column becomes `nmaipy_<name>`.

## 2. Gridding

- [nmaipy] **When an AOI is gridded** (`FeatureApi._attempt_gridding`):
  - proactively, when it exceeds 1 sq km in the country's equal-area CRS (`MAX_AOI_AREA_SQM_BEFORE_GRIDDING`);
  - reactively, on HTTP 413/504, on a 400 `AOI_EXCEEDS_MAX_SIZE`, on a 502 with an empty body once retries are used up, on a read timeout, or on an unparseable 200 body.

  Address-only queries can't be gridded and fail instead.
- [nmaipy] **Cell size** is 0.005° by default (about 550 m; `--aoi-grid-cell-size`). Each cell inherits the AOI's dates and survey pin.
- [nmaipy] **Gridding turns parcelMode off for every cell.** See the geometry reference for the consequences: neighbours' features are kept, outlines are unclipped, and `belongs_to_parcel` is null.
- [nmaipy] **Acceptance.**
  - With the default `aoi_grid_min_pct=100`, every cell must answer.
  - Lower values allow up to floor(n × (100 − pct)/100) failed cells, but at least one must answer.
  - If answered cells come from more than one survey *date*, the AOI fails ("Multiple dates…", status −1) unless `aoi_grid_inexact` is set. With `aoi_grid_inexact`, the rollup's `survey_date`/`link` come from one cell.
- [nmaipy] **Merging cells** (`geometry_utils.combine_features_from_grid`): features are merged per `feature_id`. Non-area columns come from the piece with the largest clipped area. Area handling is described in the geometry reference.
- [nmaipy] **Gridding outputs.**
  - Metadata gains `gridded`, `grid_cells`, `grid_cells_answered` and `answered_area_fraction`.
  - The rollup's `answered_area_fraction` is 1.0 for ungridded AOIs and null for failed ones.
  - Each failed cell of an accepted grid becomes an error row with `failure_type=grid`. So an AOI can have data *and* rows in the errors file.

## 3. Packs, classes, system versions

- [nmaipy] `packs` and `classes` are mutually exclusive, and the CLI defaults to the `building` pack. Unknown packs or classes raise an error.
- [nmaipy] **Deprecated classes** ("Building (Deprecated)", "Flat (Deprecated)") are dropped after fetching. Because a Roof's API `parent_id` pointed at "Building (Deprecated)", nmaipy nulls it.
  - [API] Every Roof has a deprecated-Building twin with identical area.
- [API] `/classes.json` gives each class an `availability` list of `{systemVersion, perspective, status}`.
- [nmaipy] **Unreturnable classes are nulled.**
  - A class is "returnable" at an AOI's exact `system_version` if a matching entry has status `prod`, or `alpha`/`beta` when those flags are on.
  - For each AOI, the rollup columns of unreturnable classes are set to **null**, not 0, because absence means "can't be detected at this version", not "none present".
  - The class list used is saved as `final/classes_availability.json`.
- Discrete vs connected classes: see the geometry reference.

## 4. Includes (scores)

- [nmaipy] Includes are requested as one comma-joined `include=` parameter.
  - [API] Model versions come back at response level, e.g. `versions.roofSpotlightIndex.modelVersion`.
  - [nmaipy] nmaipy copies the model version onto each feature carrying that include.
- [nmaipy] **Column mapping** (`feature_attributes._flatten_include_params`):
  - `roofSpotlightIndex` → `roof_spotlight_index`, `roof_spotlight_index_confidence`.
  - `hurricaneScore` / `windScore` / `hailScore` / `wildfireScore` → `<peril>_vulnerability_score`, `_probability`, `_rate_factor`, `<peril>_fema_version`. Wind/hail add `_risk_score`, `_risk_rate_factor` and `_fema_annual_frequency`.
  - `windHailRiskScore` → `wind_hail_risk_score`, `_rate_factor`.
  - Each score's `modelInputFeatures` → `<peril>_vulnerability_model_input_<snake_case>`. These inputs are the score model's own per-parcel values. They are distinct from roof component columns of similar names (`tree_overhang_ratio`), which come from the base AI model.
  - Every include also gets `<key>_model_version` and `<key>_skipped`.
- [nmaipy] **`defensibleSpace`**:
  - Per roof: `defensible_space_zone_<zoneId>_…` columns (zone area, defensible area, risk-object area, coverage ratio, per risk-object area/ratio).
  - Per parcel: the response-level aggregate becomes `aggregate_defensible_space_zone_*`. According to the `FeatureApi` docstring, gridded AOIs run without parcelMode, so their aggregate stays null.
- [nmaipy] **`roofConditionConfidenceStats`** adds confidence histograms inside roof components. They become `<component>_confidence_stats_<binType>_bin_<i>`.
- [API] An include can come back as `{"skipped": true}`. nmaipy records `*_skipped=Y` and leaves the score columns null.
- [nmaipy] **Resolving the primary roof's scores.**
  - The rollup's `primary_roof_spotlight_index` and peril scores come from the primary roof.
  - If it lacks them, nmaipy follows Roof →(IoU ≥ 0.005)→ Building →(parent)→ Building lifecycle. Some scores live on the building lifecycle and need the `building_structures` pack.
  - Parcel-level `*_min`/`*_max`/`*_area_weighted_mean` (weighted by clipped area) are computed across all roofs.

## 5. Rollup semantics (`parcels.parcel_rollup`)

- [nmaipy] **Per-class columns.** The prefix is the class description, lower-cased with spaces replaced by `_`. Areas are in sqft for the US and sqm elsewhere.
  - `<class>_present` (Y/N).
  - `<class>_count`: every feature row of that class. There is **no size filter**, so garages, sheds and carports count as roofs. On gridded AOIs, neighbours' roofs are included too.
  - `<class>_total_area_*`, `<class>_total_clipped_area_*`, `<class>_total_unclipped_area_*`.
  - `<class>_confidence` = 1 − Π(1 − cᵢ) over the class's features: the probability that at least one detection is real. It can exceed every individual confidence.
  - `<class>_multiparcel_feature_count` for building-style classes.
- [nmaipy] **Primary feature** (Roof, Building, Building lifecycle, Building under construction, Swimming pool):
  - `largest_intersection` (the default) picks the largest `clipped_area`, with ties broken by `unclipped_area`.
  - `nearest` needs `lat`/`lon` input columns. Only features with clipped area > 30 sqm are eligible. It prefers a feature containing the point, then the nearest within 1 m, favouring confidence ≥ 0.9. It can return none.
  - `optimal` tries `nearest`, then falls back to `largest_intersection`.
  - Rows with null lat/lon fall back to largest.
- [nmaipy] **Primary columns:** `primary_<class>_area_*`, `_clipped_area_*`, `_unclipped_area_*`, `_confidence`, `_fidelity`, `_feature_id` and `_multiparcel_feature`, plus flattened attributes (roof components, `dominant_roof_material_*`, `dominant_roof_types_*`).
- [nmaipy] **Null vs 0 vs N:**
  - **Queried, nothing found:** `_present=N`, `_count=0`, totals 0.0, `_confidence` null, `primary_*_area` 0.0.
  - **`nearest` found no eligible feature:** `_present=Y`, `_count>0`, but the primary columns hold their empty defaults.
  - **API call failed for the AOI, or class unreturnable at its system version:** every column for the class is **null**, including `_present` and `_count`. `feature_api_success=N` marks the failure case.
  - **Attribute columns** exist only where emitted, so they are null when there is no primary feature.

## 6. Errors, retries, resume and cache

- [nmaipy] **Retries.** 429, 500, 502 and 503 are retried, with exponential backoff, jitter and `Retry-After` honoured. 504 is retried only inside grid cells; otherwise it triggers gridding. 404 ("no coverage") is never retried.
- [nmaipy] **Error rows** carry `aoi_id`, `status_code` (−1 when there is no HTTP status: timeouts, multi-date grid rejection, gridding exceptions), `message`, `request` and `failure_type` (`standard`/`grid`/`unexpected`). They are written to the export's errors file.
- [nmaipy] **Resume.**
  - A chunk's rollup file is its completion marker. It is written only when at least one AOI answered or every failure was a 404.
  - Completed chunks are never re-queried, so an AOI that failed inside a completed chunk stays null on resume [inferred].
  - The exporter refuses to resume into a `chunks/` directory written under a different config. Only operational settings such as processes, threads, retries and cache options may change.
- [nmaipy] **Cache.** Only successful responses are cached, keyed by request URL and body. A re-run that reuses the cache replays the originally fetched survey even for "latest", until `--overwrite-cache` or a fresh cache directory is used.

## 7. Feature identity

- `feature_id` is stable for the same physical feature within one survey, whichever AOI or query returned it. That is what lets grid cells be merged.
- It is **not** stable across survey dates, so never join on it across dates.
- In parcelMode, a feature shared by two AOIs appears in both with the same `feature_id` but per-parcel clipped values. Row identity is `(aoi_id, feature_id)`.
