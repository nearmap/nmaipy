# Feature API: geometry, areas, parcelMode and child features

Evidence tags:
- **[live]**: verified against the live Feature API in November 2026, on ~100 New Jersey parcels bundled with nmaipy (`tests/data/test_parcels_2.csv`) plus synthetic halves of some of them, queried with parcelMode on and off. `tests/test_parcel_mode_semantics.py` keeps the key facts pinned.
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
| Examples (as tested) | Roof, Building, roof components (materials, shapes, Tree Overhang), Driveway | Vegetation (low, medium/high, woody), Lawn Grass, Asphalt, Concrete, Road (Driveable Surface), Natural (soft); nmaipy lists them in `CONNECTED_CLASS_IDS` |
| Nature | Whole objects that can extend past the AOI | No whole-object extent |
| `geometry` | Whole outline, even where it lies outside the AOI | Already clipped to the query AOI |
| `area` | = `unclipped_area` (whole object) | = `clipped_area` (inside the AOI) |
| `clipped_area` | Area inside the AOI | Area inside the AOI |
| `unclipped_area` | Whole object | The whole connected patch, which can be far larger than the AOI (hundreds of thousands of sqm for a woodland) |

This holds with parcelMode on and with it off. `clippedAreaSqm` is computed in both modes; it is not a parcelMode feature [live].

## 2. What parcelMode changes

nmaipy sends `parcelMode=true` by default. The exporter's `--no-parcel-mode` turns it off, and gridded AOIs always run with it off (section 6).

parcelMode adds two things [live]:

- **`belongsToParcel`**: whether the API attributes the feature to this parcel.
  - Roofs with about half or less of their area inside came back `false`. These are neighbours' roofs.
  - Connected features are nearly all `true`; a few Lawn Grass patches came back `false` (under 1% of connected area).
  - [nmaipy] `FeatureApi.payload_gdf` drops `false` features. So every row answered with parcelMode on has `belongs_to_parcel` = true. The column is null for gridded AOIs' rows (section 6) and when parcelMode is off.
- **`clippedGeometry`**: only for **multiparcel** discrete features, those with a substantial part in another parcel. It is the feature cut at the parcel boundary.
  - [nmaipy] nmaipy swaps it into `geometry` and sets `multiparcel_feature = True`.
  - Connected features never get it, because their geometry is already clipped.

So a belonging discrete feature that crosses the boundary comes back in one of two ways [live]:
- **Multiparcel**: it gets `clippedGeometry`. In testing the outside part was 24–128 sqm.
- **Slight overhang**: it is returned *whole*, with no `clippedGeometry`, and analysed whole. In testing the outside part was 0.1–13 sqm. Its `clippedAreaSqm` is still the in-parcel area.

The exact threshold the API uses is not documented here.

parcelMode does **not** change `areaSqm`, `unclippedAreaSqm`, `geometry` or `attributes` [live]. Across both modes and every parcel returning the same feature, the only fields that differ are `clippedArea*`, `clippedGeometry`, `belongsToParcel` and the score includes.

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
- **Roof clipped**: `parcels.calculate_child_feature_attributes` recomputes them.
  - It uses that AOI's child features, matched by class id.
  - Areas are measured in the country's equal-area CRS.
  - `ratio = Σ area(child ∩ roof geometry) / area(roof geometry)`.
  - `confidence` = the highest confidence among the intersecting children.
  - What "roof geometry" is depends on the roof:
    - **Multiparcel roof**: the stored geometry is the clip, so the result is the **per-parcel** value. It equals the API's own per-parcel `hurricaneScore.modelInputFeatures.treeOverhangRatio` to 3 decimal places on every multiparcel roof tested [live].
    - **Slight-overhang roof**: the stored geometry is the whole outline, so the result reproduces the **whole-roof** value, matching the API's component ratio to about 0.002 [live].
  - Children dropped for `belongsToParcel=false` lie outside the clipped roof, so dropping them doesn't change the result.

So `tree_overhang_ratio` in nmaipy output is the share of *this parcel's part* of the roof that is under tree overhang. The raw API component `ratio` is the share of the whole roof.

Only the `components` lists are recomputed. Includes and pitch pass through unchanged.

## 6. Gridded AOIs

[nmaipy] AOIs over about 1 sq km, or that hit size errors, are split into a grid of cells (0.005° by default) and the results merged (see the queries-and-outputs reference). **Grid sub-queries run with parcelMode off.** As a result:

- There's no `belongsToParcel` filtering, so neighbours' features that intersect the AOI are kept.
- No feature gets `clippedGeometry`. Discrete features keep their whole outlines while `clipped_area` is summed per cell. For clipped roofs, nmaipy clips the stored outline to the AOI before recomputing components (`_clip_roof_geoms_to_aoi`), so component areas aren't inflated to whole-roof size.
- `geometry_utils.combine_features_from_grid` merges cells per `feature_id`:
  - `clipped_area` is summed.
  - `area` is kept once for discrete features, because every cell repeats the whole-object area.
  - `area` is summed for connected features, because each cell reports only its piece.
  - The merge identifies discrete features by `area == unclipped_area` in every cell.

  Releases before this rule (nmaipy ≤ 5.3.0) summed `area` for every class, multiplying a discrete feature's `area` by the number of cells it touched.

## 7. Reading the rollup and per-class files

- `<class>_total_area_*` sums `area`: whole objects for discrete classes, in-AOI area for connected classes.
- `<class>_total_clipped_area_*` is the in-parcel total for every class. Use it for "how much X is on this lot".
- `<class>_total_unclipped_area_*` is whole objects, or whole patches for connected classes. It is rarely what someone wants for connected classes.
- `multiparcel_feature` / `<class>_multiparcel_feature_count`: whether the API returned the feature clipped to this parcel because it is shared with another. A slight-overhang roof is `N` here, even though it crosses the boundary.
- `belongs_to_parcel` is true on every row answered with parcelMode on, because non-belonging features were dropped, and null for gridded AOIs. It is not a "fully inside" flag.
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
