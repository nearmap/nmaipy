"""Column metadata loader and lookup for nmaipy outputs.

Single source of truth for column descriptions used by both the README generator
and the data dictionary generator. Reads from ``nmaipy/data/column_metadata.json``,
which the PM is expected to edit when descriptions need refining.

Usage::

    from nmaipy.column_metadata import lookup_column

    meta = lookup_column("primary_roof_staining_area_sqft", area_unit="sqft")
    print(meta.description)
    # -> "Total area of staining detections on the primary roof of the parcel."
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import resources
from typing import Optional

# Sentinel string for fields that the PM must fill in. Surfaces visibly in
# generated dictionaries so missing descriptions are easy to spot.
UNKNOWN_SENTINEL = "?"

# Long-form unit names paired with their short column-suffix form.
_AREA_UNIT_LONG_NAMES = {"sqft": "square feet", "sqm": "square metres"}

# Scope-prefix → human-readable phrase used to disambiguate parcel-wide vs
# primary-feature-scope columns. Parcel-scope (no prefix) maps to a default
# phrase produced by ``_format_scope_phrase``.
_SCOPE_PHRASES: dict[str, str] = {
    "primary_roof_": "on the primary roof of the parcel",
    "primary_building_": "on the primary building of the parcel",
    "primary_roof_instance_": "on the primary roof age instance of the parcel",
    "primary_building_lifecycle_": "on the primary building lifecycle of the parcel",
    "primary_swimming_pool_": "on the primary swimming pool of the parcel",
    "primary_solar_panel_": "on the primary solar panel of the parcel",
    "primary_child_roof_": "on the primary child roof of the parent feature",
    "primary_child_roof_age_": "on the primary linked roof age instance of the parent feature",
}
_DEFAULT_SCOPE_PHRASE = "across the entire parcel"

# When scope-stripping resolves a bare entry against a scoped column, the scope
# prefix carries class information. Two flavours of label:
#   - ``simple``: bare class name (used inside e.g. "Total {class_label} area"
#     so the rendering reads "Total roof area").
#   - ``primary``: full "primary <class> on the parcel" phrase (used inside
#     e.g. "Feature ID of the {class_label}" so the rendering reads
#     "Feature ID of the primary roof on the parcel"). The template chooses
#     which flavour by where it places ``{class_label}``.
# Today both fall back to the simple form unless the description text reads
# more naturally with the primary phrasing — that's controlled by the seeded
# JSON's wording, not by code.
_SCOPE_CLASS_LABELS_SIMPLE: dict[str, str] = {
    "primary_roof_": "roof",
    "primary_building_": "building",
    "primary_roof_instance_": "roof age instance",
    "primary_building_lifecycle_": "building lifecycle",
    "primary_swimming_pool_": "swimming pool",
    "primary_solar_panel_": "solar panel",
    "primary_child_roof_": "roof",
    "primary_child_roof_age_": "roof age instance",
}
_SCOPE_CLASS_LABELS_PRIMARY: dict[str, str] = {
    "primary_roof_": "primary roof on the parcel",
    "primary_building_": "primary building on the parcel",
    "primary_roof_instance_": "primary roof age instance on the parcel",
    "primary_building_lifecycle_": "primary building lifecycle on the parcel",
    "primary_swimming_pool_": "primary swimming pool on the parcel",
    "primary_solar_panel_": "primary solar panel on the parcel",
    "primary_child_roof_": "primary child roof of the parent feature",
    "primary_child_roof_age_": "primary linked roof age instance of the parent feature",
}
# Backwards-compatibility alias kept for any callers using the old name.
_SCOPE_CLASS_LABELS = _SCOPE_CLASS_LABELS_SIMPLE


@dataclass(frozen=True)
class ColumnMeta:
    """In-memory representation of one column's metadata.

    Field order matches the data-dictionary CSV column order requested in
    the INDS-2080 ticket: column name → description → allowed values →
    dtype → source → min → max → precision. Plus internal-only helpers.
    """

    dtype: str = ""
    description: str = ""
    allowed_values: str = ""
    source: str = ""
    min: str = ""
    max: str = ""
    unit: str = ""
    precision: str = ""
    notes: str = ""
    group: str = ""

    def is_unknown(self) -> bool:
        """True when description and source are both the unknown sentinel."""
        return self.description == UNKNOWN_SENTINEL and self.source == UNKNOWN_SENTINEL


@dataclass(frozen=True)
class _CompiledPattern:
    """Internal: a compiled pattern row from the JSON config."""

    regex: re.Pattern
    template: ColumnMeta


def _format_scope_phrase(scope_prefix: Optional[str]) -> str:
    """Return the human-readable phrase for a scope prefix.

    Empty/None → the parcel-wide phrase ("across the entire parcel").
    """
    if not scope_prefix:
        return _DEFAULT_SCOPE_PHRASE
    return _SCOPE_PHRASES.get(scope_prefix, _DEFAULT_SCOPE_PHRASE)


def _substitute(
    value: str,
    *,
    area_unit: str = "",
    scope: str = "",
    cls: str = "",
    class_label: str = "",
    class_label_primary: str = "",
    scope_phrase_override: str = "",
    extra_groups: Optional[dict[str, str]] = None,
) -> str:
    """Substitute templated placeholders.

    Recognised placeholders:
      ``{unit}``          – ``"sqft"`` / ``"sqm"``.
      ``{unit_name}``     – ``"square feet"`` / ``"square metres"``.
      ``{scope_phrase}``  – e.g. ``"on the primary roof of the parcel"``.
      ``{class}``         – pattern-captured detection class (humanised).
      ``{class_label}``   – simple class name from filename or scope (e.g. ``"roof"``).
      ``{class_label_primary}`` – primary-feature scope phrasing (e.g. ``"primary roof on the parcel"``).
        Falls back to ``{class_label}`` when no scope is set.
    """
    if not value:
        return value
    unit_long = _AREA_UNIT_LONG_NAMES.get(area_unit, area_unit)
    scope_phrase = scope_phrase_override or _format_scope_phrase(scope)
    # Render {class} as a human-readable phrase. Three stages:
    #  1. Strip the rollup-internal ``_total`` suffix (sometimes paired with
    #     ``_clipped`` / ``_unclipped``) so e.g. ``roof_total_area_sqft`` reads
    #     "Total area of roof detected …" rather than "roof total".
    #  2. Map ``_clipped`` / ``_unclipped`` suffixes to a parenthetical so the
    #     distinction survives without doubling up the word "area".
    #  3. Replace remaining underscores with spaces.
    cls_phrase = cls
    suffix_phrase = ""
    if cls_phrase.endswith("_total_clipped"):
        cls_phrase = cls_phrase[: -len("_total_clipped")]
        suffix_phrase = " (clipped to parcel boundary)"
    elif cls_phrase.endswith("_total_unclipped"):
        cls_phrase = cls_phrase[: -len("_total_unclipped")]
        suffix_phrase = " (before parcel clipping)"
    elif cls_phrase.endswith("_clipped"):
        cls_phrase = cls_phrase[: -len("_clipped")]
        suffix_phrase = " (clipped to parcel boundary)"
    elif cls_phrase.endswith("_unclipped"):
        cls_phrase = cls_phrase[: -len("_unclipped")]
        suffix_phrase = " (before parcel clipping)"
    elif cls_phrase.endswith("_total"):
        cls_phrase = cls_phrase[: -len("_total")]
    cls_phrase = cls_phrase.replace("_", " ") + suffix_phrase
    # {class_label_primary} falls back to plain {class_label} when no scope-aware
    # primary phrase is supplied.
    primary_label = class_label_primary or class_label
    result = (
        value.replace("{unit}", area_unit)
        .replace("{unit_name}", unit_long)
        .replace("{scope_phrase}", scope_phrase)
        .replace("{class}", cls_phrase)
        .replace("{class_label_primary}", primary_label)
        .replace("{class_label}", class_label)
    )
    # Substitute any other named groups (e.g. {peril}, {feature}) the pattern
    # exposed. Underscores → spaces so multi-word groups read naturally.
    # Each group also exposes a Title-case variant ({Peril}, {Feature}, ...) for
    # sentence-initial use.
    for key, val in (extra_groups or {}).items():
        if val:
            humanised = val.replace("_", " ")
            result = result.replace(f"{{{key}}}", humanised)
            result = result.replace(f"{{{key.capitalize()}}}", humanised.capitalize())
    return result


def _meta_from_json(entry: dict) -> ColumnMeta:
    """Construct ColumnMeta from a JSON entry. Empty strings are the default."""
    return ColumnMeta(
        dtype=entry.get("dtype", ""),
        description=entry.get("description", ""),
        allowed_values=entry.get("allowed_values", ""),
        source=entry.get("source", ""),
        min=entry.get("min", ""),
        max=entry.get("max", ""),
        unit=entry.get("unit", ""),
        precision=entry.get("precision", ""),
        notes=entry.get("notes", ""),
        group=entry.get("group", ""),
    )


def _validate(payload: dict) -> None:
    """Basic shape validation. Raises ValueError with a friendly message on bad input."""
    if not isinstance(payload, dict):
        raise ValueError("column_metadata.json: top-level value must be an object/dict.")
    for required in ("exact_matches", "patterns"):
        if required not in payload:
            raise ValueError(f"column_metadata.json: missing required top-level key '{required}'.")
    if not isinstance(payload["exact_matches"], dict):
        raise ValueError("column_metadata.json: 'exact_matches' must be an object (column name → metadata).")
    if not isinstance(payload["patterns"], list):
        raise ValueError("column_metadata.json: 'patterns' must be a list of pattern objects.")
    for i, pattern in enumerate(payload["patterns"]):
        if not isinstance(pattern, dict):
            raise ValueError(f"column_metadata.json: patterns[{i}] must be an object.")
        if "regex" not in pattern:
            raise ValueError(f"column_metadata.json: patterns[{i}] is missing 'regex'.")
        try:
            re.compile(pattern["regex"])
        except re.error as exc:
            raise ValueError(f"column_metadata.json: patterns[{i}] regex is invalid: {exc}.") from exc


@lru_cache(maxsize=1)
def load_metadata(
    json_path: Optional[str] = None,
) -> tuple[dict[str, ColumnMeta], tuple[_CompiledPattern, ...], dict[str, str], dict[str, str]]:
    """Load and validate the column metadata JSON.

    Returns a 4-tuple of:
      - exact_matches: dict mapping column name → ColumnMeta.
      - patterns: tuple of compiled patterns, ordered by config order.
      - evidence_type_legend: dict mapping integer code (as string) → description.
      - defensible_space_zone_distances: dict mapping zoneId (as string) → distance band.

    Cached: subsequent calls with the same ``json_path`` reuse the parsed result.
    Pass an explicit path to load a different config (used by tests).
    """
    if json_path is None:
        # Use importlib.resources so this works in installed packages too.
        with resources.files("nmaipy.data").joinpath("column_metadata.json").open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        with open(json_path, encoding="utf-8") as fh:
            payload = json.load(fh)

    _validate(payload)

    exact: dict[str, ColumnMeta] = {
        name: _meta_from_json(entry) for name, entry in payload["exact_matches"].items()
    }
    patterns = tuple(
        _CompiledPattern(regex=re.compile(p["regex"]), template=_meta_from_json(p))
        for p in payload["patterns"]
    )
    legend = {
        k: v
        for k, v in payload.get("evidence_type_legend", {}).items()
        if not k.startswith("_")
    }
    zone_distances = {
        k: v
        for k, v in payload.get("defensible_space_zone_distances", {}).items()
        if not k.startswith("_")
    }
    return exact, patterns, legend, zone_distances


def reload_metadata() -> None:
    """Clear the loader cache. Useful for tests that edit the JSON in place."""
    load_metadata.cache_clear()


def lookup_column(name: str, area_unit: str = "", class_label: str = "feature") -> ColumnMeta:
    """Resolve a column name to its ColumnMeta.

    Args:
        name: Column name to look up.
        area_unit: ``"sqft"`` or ``"sqm"`` for area-unit substitution.
        class_label: Human-readable class label for the file the column lives in
            (e.g. ``"building"`` for ``building_features.parquet``). Used to
            substitute ``{class_label}`` in bare entries like ``area_sqft``.
            Defaults to ``"feature"`` for mixed-class or rollup files.

    Lookup precedence:
      1. Exact match in ``exact_matches`` (templates substituted with ``area_unit``).
      2. Scope-stripped exact match: strip a known scope prefix from the front,
         re-look up the remainder, and append the scope phrase to the description.
         This catches columns like ``primary_roof_instance_roof_age_installation_date``
         that share semantics with the bare ``roof_age_installation_date`` entry but
         appear with a scope prefix in the rollup.
      3. First matching regex in ``patterns`` (templates substituted with the
         pattern's named groups: ``scope``, ``class``, ``unit``).
      4. Sentinel ColumnMeta with ``description = source = "?"``.
    """
    exact, patterns, _, zone_distances = load_metadata()

    # 1. Exact match. Try the literal name first, then a unit-templated form
    #    (e.g. "tile_area_sqft" → check "tile_area_{unit}").
    if name in exact:
        return _instantiate(exact[name], area_unit=area_unit, class_label=class_label)
    if area_unit:
        templated = name.replace(f"_{area_unit}", "_{unit}")
        if templated != name and templated in exact:
            return _instantiate(exact[templated], area_unit=area_unit, class_label=class_label)

    # 2. Scope-stripped exact match.
    for scope_prefix in _SCOPE_PHRASES:
        if not name.startswith(scope_prefix):
            continue
        remainder = name[len(scope_prefix):]
        scoped_meta = _scope_stripped_lookup(remainder, exact, area_unit, scope_prefix, class_label)
        if scoped_meta is not None:
            return scoped_meta

    # 3. Pattern match.
    for pattern in patterns:
        match = pattern.regex.match(name)
        if match:
            groups = match.groupdict()
            scope = groups.get("scope", "") or ""
            cls = groups.get("class", "") or ""
            # Forward any other named groups (e.g. peril, feature, unit, zone_id)
            # so the template can reference them as {peril}, {feature}, {zone_id}, ...
            extra_groups = {k: (v or "") for k, v in groups.items() if k not in ("scope", "class")}
            # If the pattern captured a zone_id, look up its distance band and
            # expose it as {zone_band}. Falls back to the captured id if unknown.
            zone_id = extra_groups.get("zone_id", "")
            if zone_id:
                extra_groups["zone_band"] = zone_distances.get(zone_id, zone_id)
            # If the pattern captured an optional ``primary_`` prefix, expose
            # {primary_phrasing} so a single description can read sensibly in
            # both the rollup (primary-roof scope, with BL fallback) and the
            # per-row roof.csv contexts.
            if "primary_prefix" in extra_groups:
                if extra_groups["primary_prefix"]:
                    extra_groups["primary_phrasing"] = (
                        "the primary roof on the parcel (with Building Lifecycle fallback when structural damage is present)"
                    )
                else:
                    extra_groups["primary_phrasing"] = f"this {class_label}"
            # Defensible-space scope phrasing. The same patterns serve three
            # contexts: rollup (``primary_``/``aggregate_`` prefixes) and the
            # per-roof ``roof.csv`` (no prefix — the row already identifies the
            # roof). Surface a single ``{ds_scope_phrasing}`` token so one
            # description can read sensibly in all three.
            if "roof_scope" in extra_groups:
                rs = extra_groups["roof_scope"]
                if rs == "primary_":
                    extra_groups["ds_scope_phrasing"] = (
                        "around the primary roof on the parcel"
                    )
                elif rs == "aggregate_":
                    extra_groups["ds_scope_phrasing"] = (
                        "aggregated across all roofs on the parcel"
                    )
                else:
                    extra_groups["ds_scope_phrasing"] = f"around this {class_label}"
            # Dominant-subject phrasing for ``<class>_dominant`` Y/N columns,
            # which read as "...whether {dominant_subject} is predominantly
            # {class}." Resolves against the captured ``scope`` group:
            # primary_roof_/etc. → the primary <thing>; no scope → reflects the
            # filename (per-class file = "this <thing>"; rollup unscoped = "the parcel").
            scope_for_subject = groups.get("scope") or ""
            if scope_for_subject in _SCOPE_CLASS_LABELS_SIMPLE:
                extra_groups["dominant_subject"] = (
                    f"the primary {_SCOPE_CLASS_LABELS_SIMPLE[scope_for_subject]}"
                )
            elif class_label and class_label != "feature" and class_label != "parcel":
                extra_groups["dominant_subject"] = f"this {class_label}"
            else:
                extra_groups["dominant_subject"] = "the parcel"
            # Swimming-pool condition-ratio phrasing. Pattern captures
            # ``condition`` ∈ {unmaintained, in_lanai, covered, empty,
            # above_ground} and an optional ``primary_swimming_pool_`` scope.
            # Surfaces ``{pool_subject}`` (parcel total vs primary pool) and
            # ``{condition_phrase}`` (humanised, including the irregular
            # "inside a lanai" / "above-ground" cases).
            if "condition" in extra_groups:
                cond = extra_groups["condition"]
                _COND_PHRASES = {
                    "unmaintained": "unmaintained",
                    "in_lanai": "inside a lanai",
                    "covered": "covered",
                    "empty": "empty",
                    "above_ground": "above-ground",
                }
                extra_groups["condition_phrase"] = _COND_PHRASES.get(cond, cond.replace("_", " "))
                extra_groups["pool_subject"] = (
                    "the primary swimming pool on the parcel"
                    if scope_for_subject == "primary_swimming_pool_"
                    else "the parcel's total swimming pool area"
                )
            # Per-class file override: when a column has no scope prefix
            # (e.g. ``tile_ratio`` on roof.csv) and the file is a per-class
            # output, the parcel-default scope phrase reads wrong — each row
            # is one feature, not a parcel aggregate. Override to "on this
            # {class_label}".
            scope_phrase_override = ""
            if not scope and class_label and class_label not in ("feature", "parcel"):
                scope_phrase_override = f"on this {class_label}"
            return _instantiate(
                pattern.template,
                area_unit=area_unit,
                scope=scope,
                cls=cls,
                class_label=class_label,
                scope_phrase_override=scope_phrase_override,
                extra_groups=extra_groups,
            )

    # 4. Sentinel.
    return ColumnMeta(description=UNKNOWN_SENTINEL, source=UNKNOWN_SENTINEL)


def _scope_stripped_lookup(
    remainder: str,
    exact: dict[str, ColumnMeta],
    area_unit: str,
    scope_prefix: str,
    class_label: str = "feature",
) -> Optional[ColumnMeta]:
    """If ``remainder`` is in ``exact_matches``, return its meta with scope phrase appended.

    Also tries ``roof_age_<remainder>`` for child/instance scope prefixes, since the
    bare metadata is keyed under the ``roof_age_`` prefix (e.g.
    ``primary_child_roof_age_installation_date`` strips to ``installation_date`` but
    the seeded entry is ``roof_age_installation_date``).
    """
    candidates = [remainder]
    if area_unit:
        templated = remainder.replace(f"_{area_unit}", "_{unit}")
        if templated != remainder:
            candidates.append(templated)
    if "roof_age" in scope_prefix:
        candidates.append(f"roof_age_{remainder}")
        if area_unit:
            ra_templated = f"roof_age_{remainder}".replace(f"_{area_unit}", "_{unit}")
            if ra_templated != f"roof_age_{remainder}":
                candidates.append(ra_templated)

    base: Optional[ColumnMeta] = None
    for candidate in candidates:
        if candidate in exact:
            base = exact[candidate]
            break
    if base is None:
        return None
    # The scope prefix carries class information; let it override the
    # file-level class_label so e.g. primary_roof_area_sqft on a rollup
    # renders as "Total roof area" rather than "Total feature area".
    effective_simple = _SCOPE_CLASS_LABELS_SIMPLE.get(scope_prefix, class_label)
    effective_primary = _SCOPE_CLASS_LABELS_PRIMARY.get(scope_prefix, effective_simple)
    rendered = _instantiate(
        base,
        area_unit=area_unit,
        class_label=effective_simple,
        class_label_primary=effective_primary,
    )
    scope_phrase = _format_scope_phrase(scope_prefix)
    # Append the scope phrase to the description so callers can tell parcel-scope
    # apart from primary-feature-scope at a glance. Skip it when the template
    # already references {scope_phrase}, {class_label}, or {class_label_primary}
    # — those placeholders already encode the scope-aware information.
    template_uses_scope = (
        "{scope_phrase}" in base.description
        or "{class_label}" in base.description
        or "{class_label_primary}" in base.description
    )
    if not template_uses_scope and scope_phrase not in rendered.description:
        new_desc = rendered.description.rstrip(".")
        if new_desc:
            new_desc = f"{new_desc} ({scope_phrase})."
        else:
            new_desc = f"{scope_phrase}."
        rendered = replace(rendered, description=new_desc)
    return rendered


def _instantiate(
    template: ColumnMeta,
    *,
    area_unit: str = "",
    scope: str = "",
    cls: str = "",
    class_label: str = "",
    class_label_primary: str = "",
    scope_phrase_override: str = "",
    extra_groups: Optional[dict[str, str]] = None,
) -> ColumnMeta:
    """Render templates in a ColumnMeta against the given context."""
    kw = dict(
        area_unit=area_unit,
        scope=scope,
        cls=cls,
        class_label=class_label,
        class_label_primary=class_label_primary,
        scope_phrase_override=scope_phrase_override,
        extra_groups=extra_groups or {},
    )
    return replace(
        template,
        dtype=_substitute(template.dtype, **kw),
        description=_substitute(template.description, **kw),
        allowed_values=_substitute(template.allowed_values, **kw),
        source=_substitute(template.source, **kw),
        min=_substitute(template.min, **kw),
        max=_substitute(template.max, **kw),
        unit=_substitute(template.unit, **kw),
        precision=_substitute(template.precision, **kw),
    )


def evidence_type_legend() -> dict[str, str]:
    """Return a copy of the roof age evidence_type integer-code → description mapping."""
    _, _, legend, _ = load_metadata()
    return dict(legend)


# ---------------------------------------------------------------------------
# Backwards-compatible README dict re-exports
# ---------------------------------------------------------------------------
# The existing readme_generator.py expects module-level dicts named
# COMMON_COLUMNS, ADDRESS_QUERY_COLUMNS, RSI_COLUMNS, ROOF_AGE_COLUMNS,
# DOMINANT_ROOF_MATERIAL_COLUMNS, DOMINANT_ROOF_TYPES_COLUMNS, and
# DEFENSIBLE_SPACE_ZONE_COLUMNS. We rebuild them from the JSON by group.
# ---------------------------------------------------------------------------


def _by_group(group_name: str) -> dict[str, ColumnMeta]:
    """Return all exact-match entries with the given ``group``, in JSON order."""
    exact, _, _, _ = load_metadata()
    return {name: meta for name, meta in exact.items() if meta.group == group_name}


def _group_dict() -> dict[str, dict[str, ColumnMeta]]:
    """Pre-computed group → entries, evaluated lazily and cached via load_metadata."""
    return {
        "COMMON_COLUMNS": _by_group("common"),
        "ADDRESS_QUERY_COLUMNS": _by_group("address_query"),
        "RSI_COLUMNS": _by_group("rsi"),
        "ROOF_AGE_COLUMNS": _by_group("roof_age"),
        "DOMINANT_ROOF_MATERIAL_COLUMNS": _by_group("dominant_roof_material"),
        "DOMINANT_ROOF_TYPES_COLUMNS": _by_group("dominant_roof_types"),
        "DEFENSIBLE_SPACE_ZONE_COLUMNS": _by_group("defensible_space"),
    }


def __getattr__(name: str):
    """Lazily expose group dicts as module attributes for README generator imports."""
    groups = _group_dict()
    if name in groups:
        return groups[name]
    raise AttributeError(f"module 'nmaipy.column_metadata' has no attribute {name!r}")
