"""
Test address-based exports (address mode without geometry).

This test verifies that address-only parcel files work correctly with nmaipy's
address-based Feature API endpoint.
"""
import os
import pytest
import pandas as pd
from nmaipy.exporter import AOIExporter


@pytest.fixture
def api_key():
    """Get API key from environment"""
    key = os.getenv('API_KEY')
    if not key:
        pytest.skip("API_KEY environment variable not set")
    return key


@pytest.fixture
def single_address_csv(tmp_path):
    """
    Create a CSV file with a single address (no geometry column).

    This represents a GNAF address file with all required address fields:
    - streetAddress: Full street address
    - city: City/locality name
    - state: State code
    - zip: Postcode
    """
    # Real address from GNAF: 95 WOODLANDS ROAD, GATTON QLD 4343
    data = {
        'ADDRESS_DETAIL_PID': ['GAQLD157935435'],
        'ADDRESS_LABEL': ['95 WOODLANDS RD, GATTON QLD 4343'],
        'STREET_NAME': ['WOODLANDS'],
        'STREET_TYPE': ['ROAD'],
        'LOCALITY_NAME': ['GATTON'],
        'STATE': ['QLD'],
        'POSTCODE': ['4343'],
        'LONGITUDE': [152.2829366],
        'LATITUDE': [-27.56838582],
        # nmaipy-required address fields
        'streetAddress': ['95 WOODLANDS ROAD'],
        'city': ['GATTON'],
        'state': ['QLD'],
        'zip': ['4343']
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "single_address.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_address_mode_detection(single_address_csv):
    """
    Test that nmaipy correctly detects address mode when no geometry column exists.
    """
    df = pd.read_csv(single_address_csv)

    # Verify CSV has no geometry column
    assert 'geometry' not in df.columns, "Test CSV should not have geometry column"

    # Verify required address fields exist
    required_fields = ['streetAddress', 'city', 'state', 'zip']
    for field in required_fields:
        assert field in df.columns, f"Missing required address field: {field}"
        assert df[field].notna().all(), f"Address field {field} has null values"


def test_single_address_export(api_key, single_address_csv, tmp_path):
    """
    Test that a single address can be exported using nmaipy's address mode.

    This test verifies:
    1. nmaipy detects address mode (no geometry column)
    2. nmaipy makes successful API request with address parameters as query parameters
    3. Building features are returned for the address

    Expected behavior:
    - API request should include address parameters (streetAddress, city, state, zip) as query parameters
    - API should return building features for the specified address
    - Features file should be created with building data
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create exporter for single address
    exporter = AOIExporter(
        aoi_file=single_address_csv,
        output_dir=str(output_dir),
        packs=["building"],
        since="2024-11-06",
        until="2025-11-06",
        save_features=True,
        no_cache=True,
        processes=1,
        threads=2,
        chunk_size=10,
        parcel_mode=True,  # Use parcel mode for AOI-filtered queries
        country='au',  # Australian address
        api_key=api_key
    )

    # Run export
    exporter.run()

    # Check for errors
    error_files = list((output_dir / "chunks").glob("feature_api_errors_*.parquet"))
    if error_files:
        df_errors = pd.read_parquet(error_files[0])
        if len(df_errors) > 0:
            error = df_errors.iloc[0]
            pytest.fail(
                f"Export failed with error:\n"
                f"  Status: {error['status_code']}\n"
                f"  Message: {error['message']}\n"
                f"  Request: {error['request']}\n"
                f"\n"
                f"Expected: Address parameters should be in query string\n"
                f"Actual: Address parameters missing from request"
            )

    # Check that features were exported
    features_file = output_dir / "final" / "single_address_features.parquet"
    assert features_file.exists(), (
        f"Features file not created: {features_file}\n"
        f"This likely means the API request failed.\n"
        f"Check that address parameters are included in the query string."
    )

    # Load and verify features
    df_features = pd.read_parquet(features_file)
    assert len(df_features) > 0, "No features found for address"

    # Verify building features exist
    assert 'class_id' in df_features.columns or 'description' in df_features.columns, \
        "No class information in features"

    # Verify we got building features
    assert len(df_features) > 0, f"No features found (expected building features)"


def test_url_encoding_in_address_fields():
    """
    Test that address fields are properly URL-encoded when creating POST requests.

    Verifies that special characters in address fields (spaces, apostrophes, etc.)
    are correctly encoded in the query string.
    """
    from nmaipy.feature_api import FeatureApi
    from shapely.geometry import Point
    import os

    # Get API key
    api_key = os.getenv('API_KEY')
    if not api_key:
        pytest.skip("API_KEY environment variable not set")

    # Create API instance
    api = FeatureApi(api_key=api_key)

    # Test address with special characters
    address_fields = {
        'streetAddress': "123 O'Reilly Street",  # apostrophe
        'city': "San José",  # accented character
        'state': "CA",
        'zip': "94102"
    }

    # Create a simple geometry (not actually used for address mode)
    geometry = Point(0, 0).buffer(0.001)

    # Create POST request parameters
    url, body, exact = api._create_post_request(
        base_url=api.FEATURES_URL,
        geometry=geometry,
        address_fields=address_fields,
        region='us'
    )

    # Verify URL encoding
    assert "O%27Reilly" in url, "Apostrophe should be URL-encoded as %27"
    assert "Jos%C3%A9" in url or "Jos%E9" in url, "Accented character should be URL-encoded (UTF-8 or Latin-1)"
    assert "123%20O" in url, "Space should be URL-encoded as %20"
    assert "&country=US" in url, "Country code should be uppercase"

    # Verify all address fields are present
    assert "streetAddress=" in url
    assert "city=" in url
    assert "state=" in url
    assert "zip=" in url


def test_invalid_region_validation():
    """
    Test that invalid region codes are rejected with helpful error messages.
    """
    from nmaipy.feature_api import FeatureApi
    from shapely.geometry import Point
    import os

    # Get API key
    api_key = os.getenv('API_KEY')
    if not api_key:
        pytest.skip("API_KEY environment variable not set")

    # Create API instance
    api = FeatureApi(api_key=api_key)

    # Test address fields
    address_fields = {
        'streetAddress': "123 Main St",
        'city': "Test City",
        'state': "XX",
        'zip': "12345"
    }

    # Create a simple geometry
    geometry = Point(0, 0).buffer(0.001)

    # Test with invalid region code
    with pytest.raises(ValueError) as exc_info:
        api._create_post_request(
            base_url=api.FEATURES_URL,
            geometry=geometry,
            address_fields=address_fields,
            region='invalid'
        )

    # Verify error message is helpful
    error_msg = str(exc_info.value)
    assert "invalid" in error_msg.lower()
    assert "au" in error_msg
    assert "ca" in error_msg
    assert "nz" in error_msg
    assert "us" in error_msg


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
