from setuptools import find_packages, setup
import os

name = "nmaipy"
pysrc_dir = "."
packages = [p for p in find_packages(pysrc_dir) if not p.startswith("tests")]
package_dir = {"": pysrc_dir}

# Read version from __version__.py
version_file = os.path.join(os.path.dirname(__file__), "nmaipy", "__version__.py")
with open(version_file) as f:
    exec(f.read())

with open("LICENSE") as f:
    _license = f.read()

dev_packages = [
    "black",
    "isort",
    "pytest",
]

notebooks_packages = [
    "ipykernel",
    "matplotlib",
]

required_packages = [
    "geopandas",
    "numpy",
    "pandas",
    "psutil",
    "pyarrow",
    "pyproj",
    "python-dotenv",
    "requests",
    "rtree",
    "shapely",
    "stringcase",
    "tqdm",
]

setup(
    name=name,
    version=__version__,
    description="Nearmap AI Python Library for extracting AI features from aerial imagery",
    url=f"https://github.com/nearmap/{name}",
    author="Nearmap AI Systems",
    author_email="ai.systems@nearmap.com",
    license=_license,
    packages=packages,
    package_dir=package_dir,
    python_requires=">=3.11",
    install_requires=required_packages,
    extras_require={
        "dev": dev_packages,
        "notebooks": notebooks_packages,
    },
    zip_safe=False,
)
