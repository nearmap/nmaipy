"""
Regression test: importing nmaipy.exporter installs a module-level filter that
suppresses the geographic-CRS centroid UserWarning.

The exporter computes centroids in EPSG:4326 to feed mapbrowser URL pins —
geographic CRS is intentional there and the resulting warning is cosmetic.
A prior `with warnings.catch_warnings(): filterwarnings(...)` block around the
centroid call appeared to suppress it but leaked under load, because
catch_warnings is documented as not thread-safe and the exporter's
ThreadPoolExecutor workers raced on warnings.filters. The fix promoted the
filter to module level.

This must be tested in a subprocess: pytest's own warning-capture plugin wraps
each test in catch_warnings() and the session in its own filter state, so a
module-level filterwarnings() call during import is invisible to assertions
running inside the test. A subprocess runs without pytest's harness and
matches the production CLI environment.
"""

import subprocess
import sys
import textwrap


def test_centroid_warning_suppressed_in_production_environment():
    # The script triggers the exact warning that production emits and exits 0
    # iff no warning was raised. ``warnings.simplefilter("error")`` converts any
    # un-suppressed warning into an exception, which makes silent leaks fatal.
    script = textwrap.dedent(
        """
        import warnings
        warnings.simplefilter("error")  # any un-ignored warning becomes an error
        import nmaipy.exporter  # installs the module-level ignore filter
        import geopandas as gpd
        from shapely.geometry import Polygon

        gs = gpd.GeoSeries([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
        _ = gs.centroid  # would raise UserWarning if not suppressed
        """
    ).strip()

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        "Subprocess raised — module-level centroid suppression failed.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
