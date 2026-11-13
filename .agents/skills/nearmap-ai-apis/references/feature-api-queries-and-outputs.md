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
8. Confidence scores and cutoffs
9. Comparing surveys over time

## 1. Survey and date selection

- [nmaipy] **How a request is built.** Each request is a POST whose body holds only the AOI geometry. Every other option is a URL parameter: `since`, `until`, `order`, `packs`, `classes`, `include`, `systemVersionPrefix`/`systemVersion`, `parcelMode`, `3dCoverage`, `rapid`, `excludeTilesWithOcclusion`, `alpha`/`beta`/`prerelease` (`FeatureApi._create_post_request`).
- [nmaipy] **Per-AOI dates.** `since`/`until` are sent as `YYYY-MM-DD` strings. Per-row `since`/`until` columns in the AOI file override the bulk `--since`/`--until` where the row has a value; empty or blank cells use the bulk value. On reading, date-typed columns (as GeoJSON, GeoPackage and typed parquet produce) become `YYYY-MM-DD` strings. Before any request, nmaipy rejects any bulk or per-row value that is not a `YYYY-MM-DD` calendar date, and any AOI whose effective `since` is after its `until` (`aoi_io.validate_date_cutoffs`). This applies in the exporters and in `FeatureApi.get_features_gdf_bulk` / `RoofAgeApi.get_roof_age_bulk`.
- [nmaipy] **Pinning a survey.** A `survey_resource_id` (per row) uses the `/surveyresources/{id}/features.json` endpoint and silently ignores `since`/`until`.
  - It is the AI resource id, not the Coverage survey id.
  - [API] Resource ids change when a survey's AI is re-run. To pin a survey reproducibly, set `since` = `until` = its date.
- [nmaipy] **`order`** (`earliest`/`latest`) is sent only when set.
- [nmaipy] **System version prefix.** The exporter (CLI and `NearmapAIExporter`) sends no prefix by default, so the API chooses the system version. Only code that constructs `FeatureApi` directly gets its default `system_version_prefix="gen6-"`.
- [nmaipy] **`prefer3d`** asks for 3D coverage first and re-runs AOIs that returned 404 in 2D. A gridded AOI with any 404 cell is redone entirely in 2D. AOIs pinned to a survey resource are not retried.
- [API] **Response metadata.** Each response carries `surveyDate`, `surveyId`, `resourceId`, `systemVersion`, `perspective`, `postcat`, `3dDate` (empty for 2D), `link` and `versions`. Each feature also carries its own `surveyDate` and `meshDate`.
- [nmaipy] **Where those dates land.**
  - Rollup rows get AOI-level `survey_date`, `survey_id`, `survey_resource_id`, `system_version`, `perspective`, `postcat`, `mesh_date` and `link`.
  - Feature and per-class rows keep each feature's own `survey_date`/`mesh_date`.
  - If an input column has the same name, the metadata column becomes `nmaipy_<name>`.

## 2. Gridding

- [nmaipy] **When an AOI is gridded** (`FeatureApi._attempt_gridding`):
  - proactively, when it exceeds 1 sq km in the country's equal-area CRS (`MAX_AOI_AREA_SQM_BEFORE_GRIDDING`);
  - reactively, on HTTP 413/504, on a 400 `AOI_EXCEEDS_MAX_SIZE`, on a 502 with an empty body once retries are used up, on a read timeout (after one retry), or on an unparseable 200 body;
  - reactively, when the API still skips a realtime score after one retry and the AOI spans more than one small cell (`retry_on_realtime_skip`, on by default; `--no-retry-on-realtime-skip` turns it off). See section 4.

  Address-only queries can't be gridded and fail instead. Why cells run without parcelMode, and what that changes in the output, is in the geometry reference, section 6.
- [nmaipy] **Cell size** is 0.005° by default (about 550 m; `--aoi-grid-cell-size`). Each cell inherits the AOI's dates and survey pin.
- [nmaipy] **Gridding turns parcelMode off for every cell,** deliberately (geometry reference, section 6): neighbours' features crossing the outer boundary are kept, outlines are unclipped, includes are whole-roof, and `belongs_to_parcel` is null.
- [nmaipy] **Acceptance.**
  - With the default `aoi_grid_min_pct=100`, every cell must answer.
  - Lower values allow up to floor(n × (100 − pct)/100) failed cells, but at least one must answer.
  - If answered cells come from more than one survey *date*, the AOI fails ("Multiple dates…", status −1) unless `aoi_grid_inexact` is set. With `aoi_grid_inexact`, the rollup's `survey_date`/`link` come from one cell.
- [nmaipy] **Merging cells** (`geometry_utils.combine_features_from_grid`): features are merged per `feature_id`. Non-area columns come from the piece with the largest clipped area. Area handling is described in the geometry reference.
- [nmaipy] **Gridding outputs.**
  - The rollup gains `grid_cells`, `grid_cells_answered`, `answered_area_fraction` and `answered_area_<unit>` (sqft for the US, sqm elsewhere). `answered_area_fraction` is 1.0 for ungridded AOIs and null for failed ones.
  - Each failed cell of an accepted grid becomes an error row with `failure_type=grid`, so an AOI can have data *and* rows in the errors file. A grid rejected as a whole also gets a `failure_type=grid` row; check `feature_api_success` to tell them apart.

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
  - `hurricaneScore` / `windScore` / `hailScore` / `wildfireScore` → `<peril>_vulnerability_score`, `_probability`, `_rate_factor`, `<peril>_fema_version`. Wind and hail add `_risk_score` and `_risk_rate_factor`; wind, hail and wildfire add `_fema_annual_frequency`.
  - `windHailRiskScore` → `wind_hail_risk_score`, `_rate_factor`.
  - Each score's `modelInputFeatures` → `<peril>_vulnerability_model_input_<snake_case>`, or `wind_hail_risk_model_input_<snake_case>` for `windHailRiskScore`. These inputs are the score model's own per-parcel values. They are distinct from roof component columns of similar names (`tree_overhang_ratio`), which come from the base AI model.
  - Each include also gets `<key>_skipped` (True/False), where `<key>` is the snake-cased include name (`hurricane_score_skipped`), and `<key>_model_version` when the response reports a model version for it. `roofConditionConfidenceStats` gets neither.
- [nmaipy] **`defensibleSpace`**:
  - Per roof: `defensible_space_zone_<zoneId>_…` columns (zone area, defensible area, risk-object area, coverage ratio, per risk-object area/ratio).
  - Per parcel: the response-level aggregate becomes `aggregate_defensible_space_zone_*`. According to the `FeatureApi` docstring, gridded AOIs run without parcelMode, so their aggregate stays null.
- [nmaipy] **`roofConditionConfidenceStats`** adds confidence histograms inside roof components. They become `<component>_confidence_stats_<binType>_bin_<i>`.
- [API] An include can come back as `{"skipped": true}`: the API skips scores for very large parcels (the scorer gives up on very many roofs in one request) or under load.
  - [nmaipy] nmaipy first requests the AOI once more, unchanged (parcelMode included) and past its cache, after `REALTIME_SKIP_RETRY_DELAY_SECONDS` (10 s). That clears a skip caused by load.
  - [nmaipy] If the score is still skipped and the AOI spans more than one small cell (`REALTIME_SKIP_GRID_SIZE_DEGREES`, about 0.04 sq km, roughly 20 roofs even in dense urban areas), nmaipy grids it at that cell size (section 2), pinned to the response's survey, so each cell is a request the scorer can handle. The scores then come from cells queried without parcelMode, with what that means for the rest of the AOI's results (geometry reference, section 6).
  - [nmaipy] `--no-retry-on-realtime-skip` (`retry_on_realtime_skip=False`) turns off both steps. The 5.3.0 names `--no-regrid-on-skip` and `regrid_on_skip` are deprecated aliases.
  - Otherwise, or if the grid fails, the AOI keeps its response: `*_skipped` is True and the score columns are null. That is also what a grid cell that itself skips reports.
- [nmaipy] **Resolving the primary roof's scores.**
  - The rollup's `primary_roof_spotlight_index` and peril scores come from the primary roof.
  - If it lacks them, nmaipy follows Roof →(IoU ≥ 0.005)→ Building →(parent)→ Building lifecycle. Some scores live on the building lifecycle and need the `building_structures` pack.
  - Parcel-level `*_min`/`*_max`/`*_area_weighted_mean` (weighted by clipped area) are computed across all roofs.

## 5. Rollup semantics (`parcels.parcel_rollup`)

The Feature API's former rollup endpoints are deprecated; leave them aside. nmaipy builds its rollup from the features response (`parcels.parcel_rollup`), and everything below describes that.

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
- [nmaipy] **Primary columns:** `primary_<class>_area_*`, `_clipped_area_*`, `_unclipped_area_*`, `_confidence` and `_feature_id`, plus `_fidelity` and `_multiparcel_feature` for building-style classes (not Swimming pool), plus flattened attributes (roof components, `dominant_roof_material_*`, `dominant_roof_types_*`).
- [nmaipy] **Null vs 0 vs N:**
  - **Queried, nothing found:** `_present=N`, `_count=0`, totals 0.0, `_confidence` null, `primary_*_area` 0.0.
  - **`nearest` found no eligible feature:** `_present=Y`, `_count>0`, but the primary columns hold their empty defaults.
  - **API call failed for the AOI (including a 404, no coverage), or class unreturnable at its system version:** every column for the class is **null**, including `_present` and `_count`. `feature_api_success=N` marks the failure case.
  - **Attribute columns** exist only where emitted, so they are null when there is no primary feature.
  - **Clipped roof component whose area cannot be computed (an unrepairable invalid geometry):** its `_present`, area and `_ratio` are null with no `_confidence`, and the `dominant_roof_material_*` and `dominant_roof_types_*` columns are null too, not `unknown`.

## 6. Errors, retries, resume and cache

- [nmaipy] **Retries.** 429, 500, 502 and 503 are retried, with exponential backoff, jitter and `Retry-After` honoured. 504 is retried only inside grid cells; otherwise it triggers gridding. 404 ("no coverage") is never retried.
- [nmaipy] **Error rows** carry `aoi_id`, `status_code`, `message`, `text`, `request` and `failure_type` (`standard`/`grid`/`unexpected`). They are written to the export's errors file.
  - A read timeout is retried once, then recorded as **504** (which triggers gridding).
  - `status_code` is **−1** for a connect timeout, a retry error with no response, and a grid rejected for mixing survey dates.
  - It is **null** when gridding itself raised an exception (`message` starts "Gridding failed").
- [nmaipy] **Resume.**
  - A chunk's rollup file is its completion marker. It is written only when at least one AOI answered or every failure was a 404, and every other chunk file was written.
  - Completed chunks are never re-queried, so an AOI that failed inside a completed chunk stays null on resume [inferred].
  - The exporter refuses to resume into a `chunks/` directory written under a different config. Only operational settings such as processes, threads, retries and cache options may change.
- [nmaipy] **Cache.** Only successful responses are cached, keyed by request URL and body. A re-run that reuses the cache replays the originally fetched survey even for "latest", until `--overwrite-cache` or a fresh cache directory is used.

## 7. Feature identity

- `feature_id` is stable for the same physical feature within one survey, whichever AOI or query returned it. That is what lets grid cells be merged.
- It is **not** stable across survey dates, so never join on it across dates.
- In parcelMode, a feature shared by two AOIs appears in both with the same `feature_id` but per-parcel clipped values. Row identity is `(aoi_id, feature_id)`.

## 8. Confidence scores and cutoffs

- **Scales differ by API.**
  - [API] A Feature API feature's `confidence` is 0–1, per detection.
  - [API] A Roof Age instance's `trustScore` is 0–100 (in the bundled samples).
  - [API] A Damage Conflation rating's `confidence` is 0–1: the probability of the chosen rating, in the bundled sample.
- **A per-class minimum confidence applies.**
  - [API] Nearmap applies a minimum confidence per class (and, for some classes, a minimum fidelity). The values are not published; don't guess them.
  - [live] parcelMode doesn't change which features come back. For the same AOI, parcelMode on and off returned exactly the same features (43,885 of 120 classes, on 485 random lots across 12 US and AU metros). Very few features below a moderate confidence come back in either mode; parcelMode's per-class minimums show only in `belongsToParcel`.
  - [live] For some classes, parcelMode flags a feature below the class's minimum `belongsToParcel=false` even when it lies wholly inside the parcel, so nmaipy leaves it out of a parcelMode export. An export without parcelMode, or a gridded AOI, keeps it, so the two can differ in low-confidence counts and areas. For many classes, roofs among them, no such cut showed.
- **[nmaipy] An export's class-level confidence is a formula, not a detection's confidence.**
  - `<class>_confidence` = 1 − Π(1 − cᵢ) over the AOI's features of that class: the chance that at least one detection is real. It runs higher than any single detection's confidence, so a cutoff tuned on feature confidence doesn't carry over to it.
  - `primary_<class>_confidence` is the primary feature's own confidence.
- **No recommended cutoff.** The public documentation gives no recommended confidence threshold. Set one per class against the user's own outcomes (a labelled sample of their properties), and use the public Gen 6 system report's property-level figures for the accuracy to expect. Don't invent a "typical" cutoff.

## 9. Comparing surveys over time

Questions like "when did this roof change from shingle to slate?" mean comparing results from several surveys of the same place.

- **List the surveys first.** Use the Coverage API (see the public docs), then query each survey with `since` = `until` = its date.
- **Don't join on `feature_id`.** It changes between surveys (section 7). Match features across dates by location, e.g. by overlap.
- **Compare one system version.** Pin it, e.g. `systemVersionPrefix=gen6-` (nmaipy: `--system-version-prefix gen6-`; the exporter sends no prefix by default). Leave out any survey with no result at that version, rather than comparing across system versions, which compares different models.
- **Roof material** comes from the roof's material components: the API's component `ratio`s, or nmaipy's `<material>_present`, `<material>_ratio` and `dominant_roof_material_*`. A material change between two surveys is stronger evidence when the Roof Age API also shows a roof change between those captures (evidence types 7 or 8, roof-age reference).
