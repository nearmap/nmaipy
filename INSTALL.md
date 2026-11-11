# Installation Guide for nmaipy

## Installing nmaipy

nmaipy is published on PyPI and conda-forge. Install it into your own environment with either:

```bash
pip install nmaipy
```

```bash
conda install -c conda-forge nmaipy
```

conda-forge is recommended when you also work with other geospatial packages, as it ships GEOS, GDAL and PROJ as managed binary dependencies.

## Verifying Installation

After installation, verify nmaipy is working:

```bash
python -c "from nmaipy.exporter import NearmapAIExporter; print('nmaipy installed successfully')"
```

## Development Setup

Contributors use [pixi](https://pixi.sh) (0.83 or newer), which builds a locked conda-forge environment from `pyproject.toml` and `pixi.lock`:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
git clone https://github.com/nearmap/nmaipy.git
cd nmaipy
pixi install
pixi run test
```

See [CLAUDE.md](CLAUDE.md) for the available environments and tasks.

## Configuration

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
