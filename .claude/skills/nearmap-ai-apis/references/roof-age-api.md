# Roof Age API (US only)

Evidence tags:
- **[API]**: shown by bundled responses in nmaipy's `tests/data/` (`test_roof_age_fl_response.json`, `test_roof_age_nj_response.json`) or by the code that handles them.
- **[nmaipy]**: nmaipy's behaviour (`roof_age_api.py`, `roof_age_exporter.py`, `parcels.link_roof_instances_to_roofs`, `feature_attributes.flatten_roof_instance_attributes`).
- **[inferred]**: not shown directly.

## Requests

- [nmaipy] **Endpoint and body.** Requests are POSTs to `…/ai/roofage/v1/resources/{resource_id}.json`. The body holds either `aoi` (GeoJSON) or `address` (street, city, state, zip, country), plus optional `cursor`, `limit`, `sinceAsOfDate` and `untilAsOfDate`.
- [nmaipy] **Pagination** follows `nextCursor` until it is absent, merging pages. Top-level metadata comes from page 1.
- [nmaipy] **Datasets.**
  - The dataset is `latest` (the default) or a named version (`A.0`, `A.1`, …) or UUID.
  - Date cutoffs are rejected client-side for A.0, the version `latest` currently resolves to according to nmaipy's constants.
- [nmaipy] **Unified export** (`exporter.py --roof-age`):
  - The same `until`/`since` go to both the Feature API and the Roof Age API.
  - Roof Age queries are made by AOI geometry.
  - `--roof-age` with a non-US country is disabled with a warning.

## Response

- [API] **Top level.** The response is a FeatureCollection with `resourceId`, `modelVersion`, `presentationVersion` and `nextCursor`.
- [API] **`kind`.** Each polygon is a **roof instance** with `kind` = `roof` or `parcel`.
  - `roof` instances are tied to imagery of a roof.
  - In the samples, `parcel` instances have no capture dates or map link.
  - nmaipy's column docs describe `parcel` as a property-level estimate given when there is no roof instance or imagery. Treat that as nmaipy's description, not an API guarantee.
- [API] **Per-instance properties:**
  - `installationDate` (can be null) and `asOfDate`;
  - `trustScore` (0–100 in samples);
  - `evidenceType` and `evidenceTypeDescription`;
  - `beforeInstallationCaptureDate`/`afterInstallationCaptureDate`, `minCaptureDate`/`maxCaptureDate` and `numberOfCaptures`;
  - `area` (sqm), `hilbertId` and `mapBrowserUrl`;
  - `relevantPermits` and `relevantPermitsDetails` (list of `{permitId, date, score}`);
  - `assessorData` and `assessorDataDetails`;
  - `timeline`.
- [API] **Evidence types** (nmaipy's legend, from Nearmap's public help):

  | Code | Meaning |
  |---|---|
  | 7 / 8 | A roof change was detected; the installation date is the midpoint between the two captures |
  | 5 / 6 | Multiple images, no change seen; the age is ML-predicted |
  | 3 / 4 | Single image |
  | 2 | Permit or assessor data only |
  | 0 | No evidence |

  Codes 4, 6 and 8 also have permit or assessor corroboration.
- [API] Some `roof` instances have `installationDate: null` and `trustScore: 0` even with evidence types 5 or 7. The samples don't explain why.

## What nmaipy does with it

- [nmaipy] **Normalisation.**
  - Every instance gets the synthetic class "Roof Instance".
  - `area` becomes `area_sqm`, and `hilbertId` becomes `feature_id`.
  - `resourceId`/`modelVersion` are copied to each row.
  - Permit and assessor details are stored as JSON strings.
  - **`timeline` is dropped.**
- [nmaipy] **Column prefix.** Only whitelisted fields get the `roof_age_` prefix in the unified export. Permit and assessor flags become Y/N.
- [nmaipy] **Roof age in years** = (`as_of_date` − `installation_date`) / 365.25, rounded to 1 decimal place. It is **null** when either date is missing, which is not an error.

### Linking instances to Feature API roofs (`parcels.link_roof_instances_to_roofs`)

Matching happens within each AOI only, by IoU (intersection area / union area).

- **Instance → roof:** an instance's `parent_id` is the roof with the highest IoU, if IoU ≥ 0.005.
- **Roof → instances:**
  - `child_roof_instances` lists **every** overlapping instance (IoU > 0), sorted `roof`-kind first, then by IoU.
  - `primary_child_roof_age_feature_id` is the first entry, if its IoU ≥ 0.005.
  - This is why one roof can show several instances: all overlaps are listed, but only one is primary.
- [inferred] **Consequences of matching each way independently:**
  - An instance's parent can be roof A while roof B lists it as primary.
  - Several instances can attach to one roof.
  - A multiparcel roof's Feature API geometry is the parcel clip while instances are not clipped, so IoU for shared structures is lower.
- [nmaipy] **Per-class files:**
  - `roof.csv` carries the link columns and `primary_child_roof_age_*`.
  - `building.csv` follows building → primary child roof → that roof's instance.
  - `roof_instance.csv` includes `parcel`-kind rows.

### Rollup

- [nmaipy] **Counts.** `roof_instance_present`, `_count` and `_total_area` count **`roof`-kind instances only**.
- [nmaipy] **Primary instance.**
  - It is the instance linked to the primary Feature API roof.
  - If there's no link, nmaipy selects among `roof`-kind instances, or among all instances if there are none of that kind.
  - Its columns are `primary_roof_instance_roof_age_*` and `primary_roof_instance_feature_id` (the hilbertId).
  - [inferred] So `roof_instance_count = 0` can appear alongside a filled `primary_roof_instance_*` that came from a `parcel` estimate.
- [nmaipy] **Failures.** If the Roof Age call failed for an AOI, its roof-instance columns are null and `roof_age_api_success=N`. A successful call with no results gives zero counts.

## Errors, cache, resume

- [nmaipy] **Errors.** Per-AOI failures (status code and message) go to `roof_age_errors.csv`.
- [nmaipy] **Resume.**
  - A chunk is final if any AOI answered, or every failure was a 404 (no coverage) or 403 (no access).
  - Completed chunks are not re-queried on resume.
- [nmaipy] **Cache.** The cache is keyed by resource, date cutoffs and AOI/address.
  - [inferred] Results cached under `latest` keep being served after `latest` moves, until `--overwrite-cache`.
- [nmaipy] **Standalone exporter** (`python -m nmaipy.roof_age_exporter`): it writes `roofs.parquet` (all columns) and `roofs.csv` (a curated subset without permit and assessor details). It does no Feature API linking.
