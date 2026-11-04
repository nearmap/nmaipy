---
name: nearmap-ai-apis
description: Answer technical questions about how Nearmap's AI Feature API, Roof Age API and Damage Conflation API behave, and how nmaipy's exports reflect that behaviour. Covers parcelMode and clipping, multiparcel and overhanging roofs, area vs clipped vs unclipped area, discrete vs connected classes (roofs vs vegetation), roof component ratios vs score includes (RSI, hurricane, defensible space), child features, gridding, survey and date selection, rollup counts, primary features and nulls, roof age instances and evidence types, and damage ratings and event scoping. Use it whenever someone asks why a value, count, area, ratio, date, null or missing feature looks the way it does in Nearmap AI data or an nmaipy export, or how a parameter changes the results, even if they never name the API. Examples include "why is my roof bigger than the lot?", "why does this roof have two ages?", "why is tree overhang 100% on half the roof?", "what does belongs_to_parcel mean?" and "why is this column null?".
---

# Nearmap AI APIs: technical Q&A

This skill answers questions from customers and solution engineers about the subtle behaviour of three Nearmap AI APIs:
- the **AI Feature API**: vector features such as roofs, buildings, vegetation and surfaces, with attributes and score includes;
- the **Roof Age API**: roof installation-date estimates;
- the **Damage Conflation API**: per-building post-catastrophe damage ratings.

It also covers **nmaipy**, the open-source exporter that turns these APIs' responses into rollups and per-class files. Most confusion comes from mixing two layers:
- what the API returns;
- what nmaipy computes from it.

A good answer keeps the two apart.

## How to answer

1. **Work out which layer the question is about.**
   - It is about the *raw API response* if the asker mentions JSON fields such as `clippedAreaSqm`, `belongsToParcel`, `attributes` or `hurricaneScore`.
   - It is about *nmaipy's output* if they mention snake_case columns such as `roof_total_clipped_area_sqft`, `primary_roof_tree_overhang_ratio` or `feature_api_success`.
   - It may be about both. Say which layer each part of your answer describes.
2. **Read the matching reference file** from the map below before answering. The details are subtle: for example, `areaSqm` means different things for roofs and for vegetation. Answering from general intuition gets them wrong.
3. **Say how well established each claim is.** The references tag claims as verified against the live API, read from nmaipy code, or inferred. Carry that through.
   - If the references don't cover something, say so. Don't fill the gap with plausible API behaviour; offer to check it (see `references/verifying-with-live-data.md`).
4. **Confirm nmaipy-behaviour claims against the installed source when you can.**
   - The references describe nmaipy 5.3.x. Function names are given, so you can grep for them.
   - Find the source with `python -c "import nmaipy, os; print(os.path.dirname(nmaipy.__file__))"`, or use a repo checkout.
   - If the code disagrees with the reference, trust the code and mention the difference.
5. **Lead with the direct answer**, then the mechanism, then what to do about it: which column to use instead, which flag to set, or how to check. Use exact column and field names, because the asker will look them up.
6. **Stick to documented behaviour.** Don't speculate about Nearmap's internal models or pipelines beyond what the references state.

## Reference map

| The question is about… | Read |
|---|---|
| parcelMode, `clippedGeometry`, `belongsToParcel`, multiparcel or overhanging features, `area` vs `clipped_area` vs `unclipped_area`, discrete vs connected classes, roof component ratios vs score includes, child features outside the parcel, how areas merge for gridded AOIs | `references/feature-api-geometry-and-areas.md` |
| survey/date selection (`since`, `until`, `order`, prefer3d, system versions), when and how AOIs are gridded, packs/classes/deprecated classes, includes → columns, rollup counts, primary feature selection, null vs 0 vs N, errors, retries, resume and cache, `feature_id` identity | `references/feature-api-queries-and-outputs.md` |
| roof age, roof instances, `kind` = roof/parcel, trust score, evidence types, installation dates, linking instances to roofs | `references/roof-age-api.md` |
| post-catastrophe damage ratings, event ids, pre-event vs event, `--parcel-mode` on the damage exporter, damage rollups, how this differs from Feature API damage classification | `references/damage-conflation-api.md` |
| checking a claim yourself against the live API (needs an API key) | `references/verifying-with-live-data.md` |

## Frequent questions

The details are in the reference files named in brackets.

- **"Roof area is bigger than the lot", or "roof total area exceeds the parcel".**
  - Discrete classes (roofs, buildings) report their *whole* area in `area_*`, even when part of the object lies outside the parcel.
  - Use `*_clipped_area_*` for the in-parcel part.
  - [geometry-and-areas]
- **"Vegetation area never exceeds the parcel, but roof area can."**
  - Connected classes (vegetation, surfaces) are clipped to the query AOI by the API, so for them `area` equals `clipped_area`.
  - `unclipped_area` is the whole connected patch, which can be enormous.
  - [geometry-and-areas]
- **"Tree overhang ratio on a shared (multiparcel) roof."**
  - The API's component `ratio` is always relative to the whole roof.
  - nmaipy recalculates it on the parcel's clipped roof from the tree overhang polygons. That matches the API's own per-parcel `hurricaneScore.modelInputFeatures.treeOverhangRatio`.
  - [geometry-and-areas]
- **"A roof that slightly overhangs the boundary isn't clipped."**
  - That is by design: parcelMode gives clipped geometry only to features with a substantial part in another parcel.
  - Small overhangs come back whole and are analysed whole.
  - [geometry-and-areas]
- **"Two rows have the same `feature_id`."**
  - The same physical feature can appear once per parcel it touches, with per-parcel values.
  - A row's identity is `(aoi_id, feature_id)`.
  - `feature_id`s are only stable within one survey.
  - [geometry-and-areas, queries-and-outputs]
- **"Roof count is 2 for one house."**
  - `roof_count` counts every roof feature (garages, sheds), with no size filter.
  - Gridded AOIs also include neighbours' roofs.
  - [queries-and-outputs]
- **"Why is this column null rather than N or 0?"**
  - Null means no data: the API call failed for that AOI, or the class can't be returned at that AOI's system version.
  - N or 0 means it was queried and nothing was found.
  - [queries-and-outputs]
- **"Which date is this result from?"**
  - The rollup `survey_date` is the AOI's response metadata.
  - Each feature row carries its own `survey_date`/`mesh_date`.
  - [queries-and-outputs]
- **"This roof has several roof age instances."**
  - `child_roof_instances` lists every overlapping instance of both kinds.
  - Only the best-matching one becomes the primary.
  - [roof-age]
- **"A building is missing from my damage rollup."**
  - The damage exporter's `--parcel-mode` may have dropped it.
  - Its AOI may have failed.
  - It may have no event rating, which leaves it in `n_buildings` but in no rating count.
  - [damage-conflation]
