"""Interpretation of the Feature API's class schema (/classes.json)."""


def is_class_returnable_at_version(
    availability,
    system_version: str | None,
    alpha: bool = False,
    beta: bool = False,
) -> bool:
    """Decide whether a class can return features at a specific system version.

    Interprets the `availability` list from /classes.json — each entry is a
    {systemVersion, perspective, status} dict where status is one of
    "prod", "alpha", "beta". A class is returnable at `system_version` if any
    matching entry's status is "prod", or "alpha" when `alpha=True`, or
    "beta" when `beta=True`.

    Fails open (returns True) when `system_version` is missing or the
    availability shape is unrecognized — preferring phantom columns over
    silent drops.
    """
    if not isinstance(system_version, str) or not system_version or not isinstance(availability, list):
        return True
    for entry in availability:
        if not isinstance(entry, dict):
            continue
        if entry.get("systemVersion") != system_version:
            continue
        status = entry.get("status")
        if status == "prod":
            return True
        if status == "alpha" and alpha:
            return True
        if status == "beta" and beta:
            return True
    return False
