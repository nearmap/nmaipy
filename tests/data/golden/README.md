# Golden exports

`tests/test_golden_exports.py` runs `NearmapAIExporter` end to end with no network. Every HTTP request the export makes, in the workers as well as the main process, is answered by an in-process fake of the Feature API and the Roof Age API (`tests/golden/fake_api.py`), serving a synthetic world (`tests/golden/world.py`). The test compares every file in `final/` by value against `expected/` (`latency_stats.csv` and `export_config.json` excepted), and the API cache file layout against `cache_paths.txt`. A request the fake has no route for, a worker without the fake, or a socket connection attempt fails the test.

## Cases

| Case | World | Exporter settings |
| --- | --- | --- |
| `us` | Five parcels (one empty, two sharing a roof, one with a neighbour's roof crossing in), an AOI the fake answers with 403 and one outside every survey (404); roof instances of both kinds, one with no installation date | `country="us"`, packs `building`, `building_char`, `pool`, `roof_char`, `roof_cond`, `solar`, `surfaces`, `vegetation`, `roof_age=True`, `until` set |
| `au` | Four parcels (one empty, two sharing a roof), the three invalid rows of `tests/data/test_parcels.csv`, and one AOI larger than the gridding threshold with features crossing grid lines | `country="au"`, packs `building`, `roof_char`, `vegetation`, `since`/`until` set |

The world is synthetic: rectangles laid out in metres from arbitrary origins, so no AOI is a real parcel, and every value in `expected/` comes from the fake. Class ids are nmaipy's, class descriptions come from `nmaipy/class_descriptions.json`, and attribute block and component names come from `tests/golden/vocabulary.json`, which holds names only and is written from the live API.

Each case folder holds `aois.csv`, `expected/` and `cache_paths.txt`, all written by `regenerate`. In `expected/`, each table keyed by `aoi_id` is transposed: line 1 is `column` then the row keys (`aoi_id`, plus `feature_id` when present), and each later line is one column then its values, sorted by column name. A `.parquet` whose `.csv` twin exists keeps only the columns the twin lacks. Parquet dtypes and GeoParquet metadata are in `schemas.json`.

## Re-baseline or refresh

- **Re-baseline** (an intended change to the exporter's output): run `python -m tests.golden.regenerate` (or name a case). It is offline. Review the diff of `expected/` before committing.
- **Refresh** (the live API changed): run `python -m tests.golden.drift` with a real `API_KEY`. It requests the same packs and options as each case, with a fixed `until`, on public sample AOIs already in the repository, and lists every difference from the fake. Then run `python -m tests.golden.drift --write-vocabulary` if names changed, update `world.py` and `fake_api.py` to match, and re-baseline. `test_drift` (marked `live_api`) runs the same check.

## Field allowlist

`FIELD_PATHS` in `fake_api.py` lists every JSON key path the fake may emit, per endpoint. `test_fake_fields_are_allowlisted` requires the fake to emit exactly that set during the golden runs. `IGNORED_LIVE_FIELDS` is the reviewed list of live paths the fake leaves out on purpose; the drift check fails on any live path in neither list, on any allowlisted path the live API no longer sends, and on any JSON type the fake emits that the live API does not. `test_golden_data_is_small_text` keeps the committed golden data to UTF-8 text under 1 MiB and free of credentials and fields from the ignored list.

## Comparing two exports

`python -m tests.golden.compare LEFT RIGHT` prints every difference between two exports, each an export folder (its `final/` when present) or a canonical `expected/` folder, with the same value comparison the tests use, and exits 1 when there are any. Use it to compare the output of old and new code on the same input.
