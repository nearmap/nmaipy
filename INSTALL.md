# Installation Guide for nmaipy

## Quick Start (Recommended)

### For Data Scientists using Conda

```bash
# Create a new environment with all dependencies
conda env create -f environment-minimal.yaml
conda activate nmaipy
```

### For Python users with pip

```bash
pip install -e .
```

## Detailed Installation Options

### 1. Conda Installation (Recommended for Data Scientists)

#### Option A: Minimal Installation
Only core dependencies needed to run nmaipy:

```bash
conda env create -f environment-minimal.yaml
conda activate nmaipy
```

#### Option B: Full Installation
Includes development tools and notebook support:

```bash
conda env create -f environment.yaml
conda activate nmaipy
```

#### Option C: Install into Existing Conda Environment

```bash
# Activate your environment
conda activate your-env-name

# Install dependencies
conda install -c conda-forge geopandas pandas numpy pyarrow psutil pyproj python-dotenv requests rtree shapely stringcase tqdm fsspec s3fs

# Install nmaipy
pip install -e .
```

### 2. Pip Installation

#### Basic Installation
```bash
pip install -e .
```

#### With Notebook Support
```bash
pip install -e ".[notebooks]"
```

#### For Development
```bash
pip install -e ".[dev]"
```

#### All Extras
```bash
pip install -e ".[dev,notebooks]"
```

### 3. Production Installation

For production deployments without development dependencies:

```bash
pip install .
```

### 4. Building a Conda Package (Advanced)

For system administrators who want to create a redistributable conda package:

```bash
# Build the package
conda build conda.recipe

# Install locally
conda install --use-local nmaipy
```

## Verifying Installation

After installation, verify nmaipy is working:

```python
python -c "from nmaipy.exporter import NearmapAIExporter; print('nmaipy installed successfully')"
```

## Troubleshooting

### Missing Dependencies

If you get import errors, ensure all dependencies are installed:

```bash
# For conda
conda install -c conda-forge geopandas pandas numpy pyarrow psutil pyproj python-dotenv requests rtree shapely stringcase tqdm fsspec s3fs

# For pip
pip install geopandas pandas numpy pyarrow psutil pyproj python-dotenv requests rtree shapely stringcase tqdm fsspec s3fs
```

### GEOS/GDAL Issues

If you encounter GEOS or GDAL errors, conda installation is recommended as it handles these C dependencies automatically:

```bash
conda install -c conda-forge geopandas
```

### API Key Setup

Remember to set your Nearmap API key:

```bash
export API_KEY=your_api_key_here
```

Or create a `.env` file in your project directory:

```
API_KEY=your_api_key_here
```

### S3 Output Setup (Optional)

To write export output directly to Amazon S3, configure AWS credentials:

```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Option 2: AWS credentials file (via aws configure or manual creation)
# ~/.aws/credentials
```

The `s3fs` package (installed as a core dependency) handles S3 authentication automatically when credentials are available.

## Requirements

- Python 3.12 or higher
- Nearmap API key
- 4GB+ RAM for large area extractions
- Internet connection for API access

## Platform Support

nmaipy is tested on:
- Linux (Ubuntu 22.04+)
- macOS 12+
- Windows 10/11 (via WSL2 or native Python)

## Getting Help

- Check the [README](README.md) for usage examples
- See [examples.py](examples.py) for common use cases
- Open an issue on [GitHub](https://github.com/nearmap/nmaipy) for bugs or questions