from setuptools import find_packages, setup

name = "nmaipy"
pysrc_dir = "."
packages = [p for p in find_packages(pysrc_dir) if not p.startswith("tests")]
package_dir = {"": pysrc_dir}

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
    version="0.0.0",
    description="Nearmap AI examples and utilities",
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
