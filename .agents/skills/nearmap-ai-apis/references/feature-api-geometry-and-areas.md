# Feature API: geometry, areas, parcelMode and child features

Evidence tags:
- **[live]**: verified against the live Feature API in November 2026, on ~100 New Jersey parcels bundled with nmaipy (`tests/data/test_parcels_2.csv`) plus synthetic halves of some of them, queried with parcelMode on and off. `tests/test_parcel_mode_semantics.py` pins the shared-roof facts and what parcelMode does and doesn't change, on one split parcel; it is a live test, so it runs only with an API key.
- **[nmaipy]**: what nmaipy's code does.
- **[inferred]**: reasoned from the above, not observed directly.

## Contents
1. The three area fields, and discrete vs connected classes
2. What parcelMode changes, and what it doesn't
3. Roof attributes vs score includes
4. Child features (roof components, tree overhang, …)
5. How nmaipy recomputes roof component columns
6. Gridded AOIs
7. Reading the rollup and per-class files
8. Worked example: one roof shared by two parcels

## 1. Area fields; discrete vs connected classes

Every feature has `areaSqm`/`areaSqft`, `clippedAreaSqm`/`clippedAreaSqft` and `unclippedAreaSqm`/`unclippedAreaSqft`. In nmaipy these become the columns `area_*`, `clipped_area_*` and `unclipped_area_*`.

What `area` means depends on the kind of class [live]. The rule is exact, with no rounding differences: all 2,884 discrete and 4,964 connected features tested follow it.

| | Discrete classes | Connected classes |
|---|---|---|
| Examples (as tested) | Roof, Building, roof components (materials, shapes, Tree Overhang), Driveway | Vegetation (low, medium/high, woody), Lawn Grass, Asphalt, Concrete, Road (Driveable Surface), Natural (soft) |
| Nature | Whole objects that can extend past the AOI | No whole-object extent |
| `geometry` | Whole outline, even where it lies outside the AOI | Already clipped to the query AOI |
| `area` | = `unclipped_area` (whole object) | = `clipped_area` (inside the AOI) |
| `clipped_area` | Area inside the AOI | Area inside the AOI |
| `unclipped_area` | Whole object | The whole connected patch, which can be far larger than the AOI (hundreds of thousands of sqm for a woodland) |

This holds with parcelMode on and with it off. `clippedAreaSqm` is computed in both modes; it is not a parcelMode feature [live].

`clippedAreaSqm` is computed separately from `areaSqm`, and for a discrete object wholly inside the AOI it can exceed `areaSqm` by a few tenths of a sqm. This happens on ordinary queries, not just grid cells [live]:
- on 56 single-parcel queries of the bundled parcels with parcelMode on, 33 of 214 roofs, buildings and pools (15%) had `clippedAreaSqm > areaSqm`, by 0.1–0.5 sqm, up to about 2% on small structures;
- on gridded AU and US AOIs, by 0.1–0.4 sqm, e.g. a roof at 28.1 sqm with a clipped area of 28.3.

Treat differences that small as measurement noise, not as the object extending further or as a gridding artefact.

Most classes are connected; only bounded objects (roof, building, pool, roof components, …) are discrete. [nmaipy] nmaipy uses no list of either kind, because the API's classes change; code that needs the kind reads it from these areas. (`nmaipy.constants.CONNECTED_CLASS_IDS` is deprecated, incomplete and unused.)

## 2. What parcelMode changes

nmaipy sends `parcelMode=true` by default. The exporter's `--no-parcel-mode` turns it off, and gridded AOIs always run with it off (section 6).

parcelMode adds or removes no features: for the same AOI, both modes return exactly the same set (43,885 features of 120 classes, on 485 random lots across 12 US and AU metros) [live]. It adds two things to them [live]:

- **`belongsToParcel`**: whether the feature meaningfully belongs within the parcel, as assessed by the API's geospatial algorithms.
  - A feature judged not to belong comes back `false`. Typically this is a neighbour's roof that appears to cross the boundary because of parcel/imagery offset or parallax.
  - It is not a simple share-of-area threshold: in the worked example (section 8), a roof with 49% of its area inside the parcel belongs.
  - Nearmap's public parcelMode guide (`https://developer.nearmap.com/docs/parcel-mode-user-guide.md`) outlines it: non-building objects belong when roughly half or more of their area is inside; building-like features belong when the parcel and the building–parcel intersection are both building-like; a child follows its parent.
  - For some classes, a feature below the class's minimum confidence comes back `false` even when it lies wholly inside the parcel [live]. The API still returns it; parcelMode just doesn't attribute it. For many classes, roofs among them, no such cut showed. The minimums are not published.
  - Connected features are nearly all `true`; the few Lawn Grass patches that came back `false` were low-confidence ones.
  - A few classes come back with no `belongsToParcel` at all, e.g. Pole [live].
  - [nmaipy] `FeatureApi.payload_gdf` drops `false` features. So every row answered with parcelMode on has `belongs_to_parcel` = true, except rows of classes the API doesn't flag (e.g. Pole), which are kept with it null. The column is null for gridded AOIs' rows (section 6) and when parcelMode is off. A parcelMode export therefore leaves out low-confidence features of the classes with a minimum, which an export without parcelMode, or a gridded AOI, keeps.
- **`clippedGeometry`**: only for discrete features the API judges to be **multiparcel**, genuinely shared with another parcel. It is the feature cut at the parcel boundary.
  - [nmaipy] nmaipy swaps it into `geometry` and sets `multiparcel_feature = True`.
  - Connected features never get it, because their geometry is already clipped.

So a belonging discrete feature that crosses the boundary comes back in one of two ways [live]:
- **Multiparcel**: it gets `clippedGeometry`. In testing the outside part was 24–128 sqm.
- **Slight overhang**: it is returned *whole*, with no `clippedGeometry`, and analysed whole. In testing the outside part was 0.1–13 sqm. Its `clippedAreaSqm` is still the in-parcel area.

The API's geospatial assessment decides which. The sizes above are what testing happened to show, not a rule.

Where the public parcelMode guide differs from what responses showed [live, November 2026]:
- The guide says connected features are populated with a clipped geometry. Live, they carry no `clippedGeometry`; their `geometry` is already clipped.
- The guide says connected features belong to every parcel they touch. Live, a few Lawn Grass patches came back `belongsToParcel=false`.
- The guide describes features below a per-class minimum confidence being skipped by parcel processing, without a `belongsToParcel` flag. Live, they are still returned, flagged `false`, and parcelMode on and off return the same features. The features that came back without a flag were of particular classes (e.g. Pole), at any confidence.

parcelMode does **not** change `areaSqm`, `unclippedAreaSqm`, `geometry` or `attributes` [live]. Across both modes and every parcel returning the same feature, the only fields that differ are `clippedArea*`, `clippedGeometry`, `belongsToParcel` and the score includes. For the same AOI, the clipped areas agree between modes to within the 0.1 rounding step.

## 3. Roof attributes vs score includes

These are two different kinds of per-roof data, with different clipping behaviour [live].

- **`attributes` are never recomputed for a clip.**
  - What they cover: roof material, roof shape and roof overhang components (each with `areaSqm`, `ratio`, `confidence`), pitch, and building attributes.
  - They are identical with parcelMode on or off, and in every parcel that returns the feature.
  - A component's `ratio` is component area divided by the **whole** roof's area.
- **Score includes are per-parcel for multiparcel features.** For a multiparcel roof these differ between the parcels sharing it, and differ from the parcelMode-off response:
  - `roofSpotlightIndex` (value and confidence);
  - `hurricaneScore` (and the other peril scores);
  - `defensibleSpace`.

  For slight-overhang and fully inside roofs they are whole-roof values, identical to parcelMode off.

A component's whole-roof `ratio` sits on the roof feature at `attributes[]` (the entry whose `description` is e.g. "Roof material", "Roof types" or "Roof overhang attributes") → `components[]` (e.g. "Tree Overhang") → `ratio` [live].

**Per-lot component ratio from basic fields.** Σ `clippedAreaSqm` of the component's child features (`parentId` = the roof's `id`, e.g. `description` "Tree Overhang") ÷ the roof's `clippedAreaSqm` [live]. On 21 bundled NJ lots, each split through its largest roof (November 2026, `building` and `roof_char` packs), it matched nmaipy's `tree_overhang_ratio` and `hurricaneScore.modelInputFeatures.treeOverhangRatio` within 0.002 on all 27 halves with a multiparcel roof. With parcelMode off it gave the same per-lot figures within 0.001, while the hurricane figure was whole-roof. Use this calculation to explain per-lot values on a multiparcel roof: it needs no score includes. It hasn't been checked on slight-overhang roofs, which the API analyses whole.

For example, `hurricaneScore.modelInputFeatures.treeOverhangRatio` is the clipped part's tree overhang ratio for a multiparcel roof, and the whole roof's ratio otherwise.

nmaipy flattens includes into columns such as `roof_spotlight_index`, `hurricane_vulnerability_score` and `hurricane_vulnerability_model_input_tree_overhang_ratio`, passing the values through unchanged [nmaipy].

## 4. Child features

Roof components (material, shape, Tree Overhang, Leaf-off Tree Overhang, Dormer Window, …) are separate features whose `parentId` is the roof [live].

- **A returned roof brings all its children, whole.** Every child comes back with its full `geometry`, even one lying entirely outside the AOI.
  - This happens with parcelMode on *and* off.
  - Every parcel that returns the roof gets the same child set.
- **parcelMode only flags them.**
  - Children of a multiparcel roof:
    - wholly outside the parcel: `belongsToParcel=false`, `clippedAreaSqm=0`, so nmaipy drops them;
    - crossing the boundary: `clippedGeometry`;
    - inside: belong.
  - Children of a slight-overhang roof all belong and stay whole, even any that lie outside.
  - Children of a non-belonging roof are all `false`.

## 5. How nmaipy recomputes roof component columns

[nmaipy] `feature_attributes.flatten_roof_attributes` writes, for every roof component:
- `<component>_present`
- `<component>_area_sqft`/`_sqm`
- `<component>_ratio`
- `<component>_confidence`

The names come from the API description, lower-cased with spaces replaced by `_`: `tree_overhang_ratio`, `metal_ratio`, `hip_present`, … In the rollup they are prefixed `primary_roof_`.

- **Roof not clipped** (`clipped_area >= 0.99 × unclipped_area`): the API component values are passed through.
- **Roof clipped**: `feature_attributes.calculate_child_feature_attributes` recomputes them.
  - It uses that AOI's child features, matched by class id.
  - Areas are measured in the country's equal-area CRS.
  - `ratio = Σ area(child ∩ roof geometry) / area(roof geometry)`.
  - `confidence` = the highest confidence among the intersecting children.
  - What "roof geometry" is depends on the roof:
    - **Multiparcel roof**: the stored geometry is the clip, so the result is the **per-parcel** value. It equals the API's own per-parcel `hurricaneScore.modelInputFeatures.treeOverhangRatio` to 3 decimal places on every multiparcel roof tested [live].
    - **Slight-overhang roof**: the stored geometry is the whole outline, so the result reproduces the **whole-roof** value, matching the API's component ratio to about 0.002 [live].
    - **Roof on a gridded AOI**: nmaipy first clips the outline to the AOI (section 6), so the result is the **in-AOI** value, even for a slight overhang.
  - Children dropped for `belongsToParcel=false` lie outside the clipped roof, so dropping them doesn't change the result.

So `tree_overhang_ratio` in nmaipy output is the share of *this parcel's part* of the roof that is under tree overhang. The raw API component `ratio` is the share of the whole roof.

Only the `components` lists are recomputed. Includes and pitch pass through unchanged. Within the components, `<component>_dominant` and `<component>_confidence_stats_*` also stay whole-roof, while `dominant_roof_material_*` and `dominant_roof_types_*` are rebuilt from the recomputed values. On a clipped roof the two can therefore disagree, e.g. `metal_dominant` vs `dominant_roof_material_description`.

## 6. Gridded AOIs

[nmaipy] nmaipy splits an AOI into a grid of cells (0.005° by default, about 550 m), queries each cell and merges the results, when:
- the AOI is over about 1 sq km;
- the API rejects it as too large, or times out (see the queries-and-outputs reference, section 2);
- the API still skips a realtime score (`{"skipped": true}`) after one retry, typically for a parcel with very many roofs, which the scorer gives up on in one request. The AOI is then gridded into small cells (`REALTIME_SKIP_GRID_SIZE_DEGREES`, about 0.04 sq km), if it spans more than one, so each cell is a request the scorer can handle. The cells are pinned to the first response's survey, and if the grid fails the AOI keeps that response. `--no-retry-on-realtime-skip` turns off both the retry and this.

**Grid cells are queried with parcelMode off, by design.** Gridding is meant for very large areas. There, a discrete object crossing the AOI's outer boundary matters little, whereas parcelMode would treat every cell as a parcel and cut features at the grid's internal lines. As a result, for a gridded AOI:

- **Nothing is filtered by `belongsToParcel`.** Features crossing the AOI's outer boundary are kept, neighbours' roofs included, and so are low-confidence features a parcelMode query would flag `false` (section 2). They count in `<class>_count` and `<class>_total_area_*` (whole object), but only their in-AOI part counts in `<class>_total_clipped_area_*`.
- **`belongs_to_parcel` is null on every row** (the API sends no such field without parcelMode), and **`multiparcel_feature` is always N** (no feature gets `clippedGeometry`). Discrete features keep their whole outlines.
- **Includes describe whole roofs.** RSI, the peril scores and per-roof defensible space are the parcelMode-off, whole-roof values. The parcel-level `aggregate_defensible_space_*` columns are null, except on an AOI gridded after an include skip, which keeps the aggregate from its first, parcelMode response.
- **Roof components are recomputed on the in-AOI part.** nmaipy clips each roof's outline to the AOI before recomputing its components: in the rollup's `primary_roof_*` columns (`parcels.parcel_rollup`, `clip_roof_to_parcel`) and in the per-class files (`exporter._clip_roof_geoms_to_aoi`). So `tree_overhang_ratio` describes the roof's in-AOI part while the includes describe the whole roof, and on a gridded AOI the two no longer match as they do for a parcelMode roof (section 5). The Roof Age instance link compares the same clipped outline (`exporter._link_roof_instances`).
- **The primary feature is still the largest by `clipped_area`,** so a neighbour's roof wins only if most of the in-AOI roof area is its.

`geometry_utils.combine_features_from_grid` merges the cells per `feature_id`:
- `clipped_area` is summed.
- A feature whose `area` equals its `clipped_area` in every cell is connected: each cell reported only its piece, so its `area` becomes the summed `clipped_area`.
- Every other feature is discrete: each cell repeated the whole-object area, so `area` is kept once.
- `unclipped_area` plays no part in that decision (for a connected feature it is the whole patch), and is taken from one cell, never summed. So `<class>_total_unclipped_area_*` counts each object once per AOI.
- Other columns come from the piece with the largest clipped area.
- Every cell must report either its clipped piece or the whole object: `area` equal to `clipped_area`, or to `unclipped_area`. If a cell reports neither, the merge raises and that AOI fails as a grid error.

On 8 bundled parcels split into ~10 m cells with seven packs, all 249 features (191 spanning 2–21 cells) matched the same parcels queried whole, per feature and in every per-class total [live].

## 7. Reading the rollup and per-class files

- `<class>_total_area_*` sums `area`: whole objects for discrete classes, in-AOI area for connected classes.
- `<class>_total_clipped_area_*` is the in-parcel total for every class. Use it for "how much X is on this lot". It covers features attributed to the parcel. A feature judged not to belong, typically a neighbour's roof that appears to overhang because of parallax or parcel/imagery offset, is left out entirely, by design (gridded AOIs keep such features, section 6).
- `<class>_total_unclipped_area_*` is whole objects, or whole patches for connected classes. It is rarely what someone wants for connected classes.
- `multiparcel_feature` / `<class>_multiparcel_feature_count`: whether the API returned the feature clipped to this parcel because it is shared with another. A slight-overhang roof is `N` here, even though it crosses the boundary.
- `belongs_to_parcel` is true on every row answered with parcelMode on, because non-belonging features were dropped, except classes the API doesn't flag (e.g. Pole), which are null. It is null for gridded AOIs. It is not a "fully inside" flag.
- The same `feature_id` can appear in several AOIs' rows with different clipped values. Row identity is `(aoi_id, feature_id)`. `feature_id` is stable within a survey but not across survey dates.

## 8. Worked example: one roof shared by two parcels

This is a real roof of about 187 sqm, split down the middle into two synthetic parcels, W and E [live]:

| | W | E |
|---|---|---|
| `clippedAreaSqm` | 92.1 | 94.2 |
| `areaSqm` / `unclippedAreaSqm` | 186.9 | 186.9 |
| API Tree Overhang component `ratio` | 0.612 | 0.612 (whole roof, identical) |
| API `hurricaneScore…treeOverhangRatio` | 0.287 | 0.929 (per parcel) |
| nmaipy `tree_overhang_ratio` | 0.287 | 0.929 |
| `roofSpotlightIndex.confidence` | 0.61 | 0.12 (per parcel) |

Most of the tree cover sits over the eastern half. A tree overhang polygon lying wholly in the western half came back in E's response too, with its full geometry, `belongsToParcel=false` and `clippedAreaSqm=0`.
