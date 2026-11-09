---
name: nearmap-ai-apis
description: Answer technical questions about how Nearmap's AI Feature API, Roof Age API and Damage Conflation API behave, and how nmaipy's exports reflect that behaviour. Covers parcelMode and clipping, multiparcel and overhanging roofs, area vs clipped vs unclipped area, discrete vs connected classes (roofs vs vegetation), roof component ratios vs score includes (RSI, hurricane, defensible space), child features, gridding, survey and date selection, rollup counts, primary features and nulls, roof age instances and evidence types, and damage ratings and event scoping. Use it whenever someone asks why a value, count, area, ratio, date, null or missing feature looks the way it does in Nearmap AI data or an nmaipy export, or how a parameter changes the results, even if they never name the API. Examples include "why is my roof bigger than the lot?", "why does this roof have two ages?", "why is tree overhang 100% on half the roof?", "what does belongs_to_parcel mean?" and "why is this column null?".
---

# Nearmap AI APIs: technical Q&A

This skill answers questions from customers and solution engineers about the subtle behaviour of three Nearmap AI APIs:
- the **AI Feature API**: vector features such as roofs, buildings, vegetation and surfaces, with attributes and score includes;
- the **Roof Age API**: roof installation-date estimates;
- the **Damage Conflation API**: per-building post-catastrophe damage ratings.

Most users call these APIs directly; few use **nmaipy**, the open-source exporter that turns the responses into rollups and per-class files. Treat nmaipy as a reference for how to interpret API responses, not as the user's product. The exception is a user who received a data export produced with nmaipy: their columns, rollups and primary features are nmaipy's.

A user's API key sets what they can access: which feature and attribute classes, packs and score includes, and whether post-catastrophe content, Roof Age and Damage Conflation work. So if something documented here doesn't seem to work for a user, it is usually down to the permissions on their API key. The same applies to a live check you run: a class or include that doesn't come back may simply not be enabled on the key in use.

## How to answer

1. **Pin down what the question is about before investigating.** You need to know:
   - which API it concerns (AI Feature, Roof Age or Damage Conflation);
   - where the value came from: returned by the API, computed in an nmaipy export, or derived by the user from values we delivered (e.g. a ratio of two areas they calculated themselves).

   Most questions turn out to be about the raw API, but don't assume it. Don't infer the API or the source from how names are written (camelCase vs snake_case, upper-cased or renamed columns); downstream products rename fields freely. If the question doesn't settle either point, give the part of the answer that holds under every reading, then ask only what decides the rest: list the readings you see and what you need to know to choose. Don't answer one reading with a caveat. If every reading leads to the same answer, just answer.

   When the question reports a value that looks wrong or surprising, validate it before explaining it: ask for a few real examples. Ask first for the exact API request they sent, e.g. a `curl` command with the API key removed, because replaying it is the most reliable way to reproduce what they saw. Otherwise ask for the geometry they queried, the feature ids and the values they see; for an nmaipy export, ask instead for the `aoi_id`s of the rows and the export's files. Then reproduce them against the live API (see `references/verifying-with-live-data.md`). Explain what you see in those examples, not what usually happens. While you wait for examples, still give the user what they can do now, if it holds in every case.

   Questions about what the API can do or which classes exist ("can we get…", "does the API tell us…") rarely depend on where a value came from, and a question about a kind of object (pools, decks, roofs) means the AI Feature API. Answer those directly, but still ask if something you need is unclear.
2. **Read the matching reference file** from the map below before answering. The details are subtle: for example, `areaSqm` means different things for roofs and for vegetation. Answering from general intuition gets them wrong.
3. **Say how well established each claim is.** The references tag claims as verified against the live API, read from nmaipy code, or inferred. Carry that through.
   - If the references don't cover something, say so. Don't fill the gap with plausible API behaviour; offer to check it (see `references/verifying-with-live-data.md`).
   - If you notice something else while answering, such as a possible bug, mention it separately at the end as unchecked. Keep it out of the answer and out of the text written for the user. Suggest a next step only once you've reproduced it: for a fault in nmaipy, an issue at github.com/nearmap/nmaipy that reproduces it with public or made-up parcels and contains no API key, addresses, customer data or company names; for a fault in the API's own responses, Nearmap support.
   - Before saying something isn't documented, search Nearmap's public documentation: developer.nearmap.com (through `llms.txt`, step 4) and help.nearmap.com, Nearmap's knowledge hub. If neither covers it, say "I don't know" or that it isn't publicly documented, and suggest the user ask Nearmap support. Assume the asker is usually an external customer, so don't name internal teams. Keep inferred claims out of the text written for the user.
4. **Check Nearmap's official API documentation for what the API promises,** if you can fetch URLs.
   - `https://developer.nearmap.com/llms.txt` indexes Markdown versions of the docs, e.g. `docs/ai-api.md`, `docs/parcel-mode-user-guide.md`, `docs/roof-age-api.md`, `docs/damage-conflation-api.md`, and a `reference/*.md` page per endpoint.
   - Use it for parameters, fields, endpoints and anything the references don't cover, and cite the page.
   - Where it differs from a reference file's `[live]` finding, give both: the docs state the intended contract, the `[live]` tag what responses actually showed.
   - For what a class means (definition, what it includes and excludes, which packs carry it, release status), use Nearmap's public AI layer pages, indexed at `https://help.nearmap.com/kb/articles/160-about-ai-layers`. A class appearing in the public docs doesn't mean a user can get it: check its release status and which packs carry it.
   - For model accuracy, quote only the public Gen 6 system report (`https://help.nearmap.com/api/v2/media/download-by-url?url=https%3A%2F%2Fus.v-cdn.net%2F6037589%2Fuploads%2F533RTKFGWY84%2Fgen6-systemreport.pdf`). For whether a feature is present on a property, use its property-level metrics. Its pixel-level and hard-example figures measure something different and aren't comparable.
   - Web-fetch tools often return the help-centre pages truncated (their content is rendered by script) and can't parse the report PDF. If that happens, download the file directly (e.g. with `curl`) and extract the text (e.g. `pdftotext -layout` for the PDF; the report's tables lose some alignment, so check figures against the row and column headers).
5. **Confirm nmaipy-behaviour claims against the installed source when you can.**
   - The references describe the nmaipy in the same repository checkout (its version is in `nmaipy/__version__.py`), and only that. Function names are given, so you can grep for them.
   - Find the source with `python -c "import nmaipy, os; print(os.path.dirname(nmaipy.__file__))"`, or use a repo checkout.
   - If the code disagrees with the reference, trust the code and mention the difference.
   - An export records the nmaipy version that made it in `final/export_config.json` (`_metadata.nmaipy_version`). If it is older than the checkout's, don't reason about how that version behaved: re-query a sample of the rows in question with the current nmaipy (see `references/verifying-with-live-data.md`) and explain the current result.
6. **Open with an "Applies to" line** naming the API or APIs the answer covers (AI Feature, Roof Age, Damage Conflation) and whether it also covers nmaipy exports, with the nmaipy version from `nmaipy/__version__.py`. For example: "Applies to: AI Feature API. Also nmaipy 5.5.1 exports." or "Applies to: Roof Age API only, not nmaipy." Repeat the API scope at the top of the What to tell the user text, mentioning nmaipy there only if the user received an nmaipy export.
   **Lead with the direct answer**, then the mechanism, then what to do about it: which field to use instead, which parameter to set, or how to check. Use exact field and column names, because the asker will look them up.
   - **Show spatial behaviour in a picture.** Area, clipping and parcel questions are about geometry, and are much easier to reason about drawn than described. When something is counterintuitive, find one real example (step 1) and draw it: the query AOI, the feature's outline and its in-AOI part, labelled with the values in question. Use what your environment can show: an interactive map page, an image (`references/verifying-with-live-data.md` has a plotting snippet), or a GeoJSON file the user can open at geojson.io or in QGIS.
     - Use a real example: the user's own, or real adjoining parcels. If you can't find one, use the worked example in `references/feature-api-geometry-and-areas.md` (section 8) or make one up, and say in your first sentence that it is made up. Don't describe a made-up example as a real property, and don't present its numbers as real findings in the text written for the user.
     - Make sure the user sees the picture without having to ask. A file path, or an image link to a local file, often shows nothing. Open the image for them, or put it somewhere your environment displays, such as a page or an inline image. Then say in one line what they're looking at. Assume the asker may not know how to open a file from the command line.
     - Save scripts, images and GeoJSON in the user's working folder, not a temporary system directory, so they can find them again. Use a folder that is already excluded from version control if there is one (in an nmaipy checkout, `data/`, e.g. `data/nearmap-ai-apis/<topic>/`). Otherwise create a clearly named folder and tell the user it exists.
   - Always end with a **What to tell the user** section: plain-language text the asker can paste straight into an email or Slack message. Write it for the user, not the asker. Leave out internal references (source files, function names, reference-file tags, test details) and mention nmaipy only if the user received an nmaipy export.
     - Name every value by where the user finds it: its path in the response (e.g. the roof feature's `attributes[]` entry "Roof overhang attributes" → `components[]` "Tree Overhang" → `ratio`), or the column name for an nmaipy export. Never a loose description such as "the whole-roof attributes".
     - Only describe values the user can see or calculate themselves. Show a calculation from the fields they have rather than saying a value was "worked out".
     - Don't use the peril scores (e.g. `hurricaneScore`) or `defensibleSpace` to demonstrate behaviour unless the question is about them. Many users don't request them.
7. **Stick to documented behaviour.** Don't speculate about Nearmap's internal models or pipelines beyond what the references and the official documentation state.
8. **Leave commercial questions to the account manager.** For credits, pricing, licensing or where a user may display the data, say what the official docs state and refer the rest to their account manager. Quote the docs rather than interpreting contract or legal terms, and don't point users to deprecated endpoints.

## Reference map

| The question is about… | Read |
|---|---|
| parcelMode, `clippedGeometry`, `belongsToParcel`, multiparcel or overhanging features, `area` vs `clipped_area` vs `unclipped_area`, discrete vs connected classes, roof component ratios vs score includes, child features outside the parcel, how areas merge for gridded AOIs | `references/feature-api-geometry-and-areas.md` |
| survey/date selection (`since`, `until`, `order`, prefer3d, system versions), when and how AOIs are gridded, packs/classes/deprecated classes, includes → columns, rollup counts, primary feature selection, null vs 0 vs N, errors, retries, resume and cache, `feature_id` identity, confidence scores and cutoffs, comparing surveys over time (e.g. roof material changes) | `references/feature-api-queries-and-outputs.md` |
| roof age, roof instances, `kind` = roof/parcel, trust score, evidence types, installation dates, linking instances to roofs | `references/roof-age-api.md` |
| post-catastrophe damage ratings, event ids, pre-event vs event, `--parcel-mode` on the damage exporter, damage rollups, how this differs from Feature API damage classification | `references/damage-conflation-api.md` |
| checking a claim yourself against the live API (needs an API key) | `references/verifying-with-live-data.md` |
| the API's official, documented contract: parameters, fields, endpoints, other Nearmap APIs | `https://developer.nearmap.com/llms.txt` and the page it links |
| what a class or layer means, which packs carry it, its release status | `https://help.nearmap.com/kb/articles/160-about-ai-layers` and the layer's own page |
| model accuracy (precision, recall, area error) | the public Gen 6 system report (link in step 4) |

## Frequent questions

These come after step 1, not instead of it: if the answer depends on which field, mode or source the user means, ask first. The details are in the reference files named in brackets.

- **"Roof area is bigger than the lot", or "roof total area exceeds the parcel".**
  - Ask first: which number (one roof's `areaSqm`/`area_*`, a total across roofs such as `roof_total_area_*`, the primary roof's area, or the polygon measured in their own GIS); whether parcelMode was on, off, or the AOI was gridded; and where the lot area came from.
  - One roof's `areaSqm` is the *whole* roof in either mode, even when part of it lies outside the parcel. `clippedAreaSqm` (`*_clipped_area_*`) is the in-parcel part.
  - A total across roofs without parcelMode, or for a gridded AOI, can include neighbours' roofs that cross the boundary. With parcelMode, those come back `belongsToParcel=false` (nmaipy drops them).
  - In an nmaipy export, a multiparcel roof's geometry is the clipped piece while `area_*` stays whole, so an area measured from the polygon won't match `area_*`.
  - [geometry-and-areas]
- **"Vegetation area never exceeds the parcel, but roof area can."**
  - Connected classes (vegetation, surfaces) are clipped to the query AOI by the API, so for them `area` equals `clipped_area`.
  - `unclipped_area` is the whole connected patch, which can be enormous.
  - [geometry-and-areas]
- **"Tree overhang ratio on a shared (multiparcel) roof."**
  - The API's component `ratio` (roof `attributes[]` "Roof overhang attributes" → `components[]` "Tree Overhang") is always relative to the whole roof.
  - The per-lot ratio is Σ `clippedAreaSqm` of the Tree Overhang features whose `parentId` is the roof's `id`, divided by the roof's `clippedAreaSqm`. It works with parcelMode on or off. Show this calculation.
  - nmaipy's `tree_overhang_ratio` gives the same figure for a multiparcel roof (but the whole-roof ratio for a slight-overhang roof).
  - [geometry-and-areas]
- **"A roof that slightly overhangs the boundary isn't clipped."**
  - That is by design: the API's geospatial assessment decides whether a crossing feature is genuinely shared (multiparcel, clipped) or just overhangs.
  - Small overhangs come back whole and are analysed whole.
  - [geometry-and-areas]
- **"Two rows have the same `feature_id`."**
  - The same physical feature can appear once per parcel it belongs to, with per-parcel values.
  - A row's identity is `(aoi_id, feature_id)`.
  - `feature_id`s are only stable within one survey.
  - [geometry-and-areas, queries-and-outputs]
- **"Roof count is 2 for one house."**
  - `roof_count` counts every roof feature (garages, sheds), with no size filter.
  - Without parcelMode (`--no-parcel-mode`, or a gridded AOI) it also includes neighbours' roofs that cross the boundary.
  - [queries-and-outputs]
- **"Why is this column null rather than N or 0?"**
  - A whole class's columns, `_present` and `_count` included, are null when the API call failed for that AOI (`feature_api_success=N`) or the class can't be returned at that AOI's system version.
  - Queried and nothing found gives `_present=N` and zero counts and totals, but `<class>_confidence` is null.
  - Primary-feature attribute columns are null when there is no primary feature.
  - Score columns are null when the API skipped that include (`*_skipped`).
  - `belongs_to_parcel` is null on rows answered without parcelMode, gridded AOIs included, and for classes the API doesn't flag (e.g. Pole).
  - [queries-and-outputs, geometry-and-areas]
- **"Clipped area is bigger than the area."**
  - For an object wholly inside the AOI, `clippedAreaSqm` can exceed `areaSqm` by a few tenths of a sqm (up to about 2% on small structures), on ordinary and gridded queries alike.
  - It is measurement noise: the two are computed separately. It doesn't mean the object extends further.
  - Show one example: the outline sits wholly inside the parcel, yet the numbers differ.
  - [geometry-and-areas]
- **"What confidence cutoff should we use?"**
  - Nearmap publishes no recommended cutoff; set one per class against the user's own outcomes.
  - Scales differ: feature `confidence` 0–1, Roof Age `trustScore` 0–100. An nmaipy export's `<class>_confidence` combines all of a class's detections, so it runs higher than any one of them.
  - [queries-and-outputs]
- **"Which date is this result from?"**
  - The rollup `survey_date` is the AOI's response metadata.
  - Each feature row carries its own `survey_date`/`mesh_date`.
  - [queries-and-outputs]
- **"This roof has several roof age instances."**
  - `child_roof_instances` lists every overlapping instance of both kinds.
  - Only one becomes the primary: roof-level instances first, then the best overlap.
  - A parcel-level estimate becomes the primary when no roof-level instance overlaps the roof.
  - [roof-age]
- **"A building is missing from my damage rollup."**
  - The damage exporter's `--parcel-mode` may have dropped it.
  - Its AOI may have failed.
  - It may have no event rating, which leaves it in `n_buildings` but in no rating count.
  - [damage-conflation]
