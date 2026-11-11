"""Re-baseline golden cases: run each offline on the fake API and rewrite its `aois.csv`, `expected/` and
`cache_paths.txt`.

Usage: python -m tests.golden.regenerate [case ...]
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from .compare import canonicalise_export
from .replay import CASES, GOLDEN_DIR, run_case


def main() -> None:
    """Regenerate the named cases (all when none are named)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="*", metavar="case", help=f"one of {sorted(CASES)} (default: all)")
    cases = parser.parse_args().cases or sorted(CASES)
    if unknown := set(cases) - set(CASES):
        parser.error(f"unknown cases {sorted(unknown)}")
    for case in cases:
        regenerate(case)


def regenerate(case: str) -> None:
    """Rewrite one case's golden data from a fresh offline run."""
    case_dir = GOLDEN_DIR / case
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        final = run_case(case, out_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_dir / "aois.csv", case_dir / "aois.csv")
        canonicalise_export(final, case_dir / "expected")
        (case_dir / "cache_paths.txt").write_text("".join(f"{p}\n" for p in cache_paths(out_dir)), encoding="utf-8")
    print(f"regenerated {case_dir}")


def cache_paths(out_dir: Path) -> list[str]:
    """The API cache files a run wrote, relative to its cache root."""
    root = out_dir / "cache" / "cache"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


if __name__ == "__main__":
    main()
