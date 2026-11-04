# Golden exports

`tests/test_golden_exports.py` runs `NearmapAIExporter` end to end with no network: every HTTP request the export makes is answered from the recordings here, in the workers as well as the main process. It then compares every file in `final/` by value against `expected/` (`latency_stats.csv` and `export_config.json` excepted) and the API cache file layout against `cache_paths.txt`. A request with no recording, a recording the export never asks for, or a socket connection attempt fails the test.

## Cases

| Case | AOIs | Exporter settings |
| --- | --- | --- |
| `us_parcels` | First 10 rows of `tests/data/test_parcels_2.csv`, plus a synthetic square in the ocean | `country="us"`, all eight public packs, `roof_age=True` |
| `au_parcels` | `tests/data/test_parcels.csv` | `country="au"`, packs `building`, `roof_char` |
| `au_gridded` | The large Adelaide AOI from `tests/conftest.py` (gridded) | `country="au"`, pack `building`, survey date `2025-03-06` |

Each case folder holds `aois.csv`, `exchanges/` (one JSON file per request/response pair), `expected/` (the canonical export: tables sorted by row key and column name, the README's generated-date line removed) and `cache_paths.txt`.

## Recording policy

- Only AOIs already in this repository.
- Only the public packs `building`, `building_char`, `roof_char`, `roof_cond`, `surfaces`, `vegetation`, `pool`, `solar`; no alpha, beta or prerelease flags.
- The `packs.json` and `classes.json` catalogue responses are cut to the requested packs and their classes.
- No credentials: the api key query parameter is dropped and no request headers are stored.

`test_recordings_are_public_safe` enforces all of this, plus that every class id in a response is in the bundled `nmaipy/class_descriptions.json`.

## Re-baseline or re-record

- **Re-baseline** (an intended change to the exporter's output, same API responses): comment out the skip line on `test_regenerate_golden_expected` and run it. It is offline; it rewrites `expected/` and `cache_paths.txt` from the committed recordings. Review the diff, then restore the skip line.
- **Re-record** (a new case, or the API's responses changed): comment out the skip line on `test_record_golden_exchanges` and run it with a real `API_KEY`. It runs each case live, rewrites `exchanges/`, re-baselines from the new recordings, and fails unless replaying the recordings reproduces the live export. If the live class catalogue differs from `nmaipy/class_descriptions.json` it fails; refresh that file in a separate change first. Restore the skip line before committing.

Never commit an uncommented skip line.

## Comparing two exports

`python -m tests.golden.compare LEFT RIGHT` prints every difference between two export folders (their `final/` when present) with the same value comparison the tests use, and exits 1 when there are any. Use it to compare the output of old and new code on the same input.
