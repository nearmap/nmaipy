"""Name conversion for output column names."""

import re

_SEPARATOR = re.compile(r"[-.\s]")
_UPPER = re.compile(r"[A-Z]")


def snake_case(name: object) -> str:
    """Convert ``name`` (as ``str(name)``) to snake case.

    Replace each ``-``, ``.`` and whitespace character with ``_``, lowercase the first character, and put ``_``
    before each later ASCII capital, lowercased. This is the rule of ``stringcase.snakecase``, so the column names
    derived with it do not change. It keeps that rule's quirks: ``Has3D`` gives ``has3_d`` and ``foo_Bar`` gives
    ``foo__bar``.
    """
    text = _SEPARATOR.sub("_", str(name))
    return text[:1].lower() + _UPPER.sub(lambda m: "_" + m.group(0).lower(), text[1:])
