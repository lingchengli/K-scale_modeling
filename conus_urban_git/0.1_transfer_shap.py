import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import numpy as np
import xarray as xr

## Read the feather file
dir_in = '/compyfs/lili400/project/ICoM/conus_urban/'
file_in = 'huc12.feather'
file_out = 'huc12.shp'

## Read and print information
gdf = gpd.read_feather(dir_in + file_in)

## Print basic information
print("\nDataset Info:")
print(gdf.info())

print("\nFirst 5 rows:")
print(gdf.head())

print("\nColumn names:")
print(gdf.columns.tolist())

print("\nCRS Information:")
print(gdf.crs)

print("\nSpatial bounds:")
print(gdf.total_bounds)

# Define exact boundaries
start_lon, end_lon = -125, -66
start_lat, end_lat = 25, 50
resolution = 1/120

# Create coordinate arrays (centers of grid cells)
lons = np.arange(start_lon + resolution/2, end_lon, resolution)
lats = np.arange(start_lat + resolution/2, end_lat, resolution)

# Calculate grid dimensions
width = len(lons)
height = len(lats)

# Create the transform
transform = from_bounds(start_lon, start_lat, end_lon, end_lat, width, height)

# Reproject to WGS84
gdf_wgs84 = gdf.to_crs('EPSG:4326')

# to shapefile
gdf_wgs84.to_file(dir_in + file_out)

# Create numeric IDs for HUC12s
gdf_wgs84['huc12_id'] = range(1, len(gdf_wgs84) + 1)
shapes = ((geom, value) for geom, value in zip(gdf_wgs84.geometry, gdf_wgs84['huc12_id']))

print("Starting rasterization...")
print(f"Grid dimensions: {width} x {height}")

# Rasterize
mask = rasterize(
    shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.int32
)
mask = np.flipud(mask)  # Flip the mask array vertically, so that the first row corresponds to the northernmost latitude

# Create xarray Dataset
ds = xr.Dataset(
    data_vars={
        'huc12_mask': (['lat', 'lon'], mask)
    },
    coords={
        'lat': lats,
        'lon': lons
    },
    attrs={
        'description': 'HUC12 watershed ID mask',
        'resolution': f'{resolution} degrees',
        'projection': 'WGS84'
    }
)

# Add variable attributes
ds.huc12_mask.attrs = {
    'long_name': 'HUC12 Watershed ID',
    'units': 'watershed_id',
    '_FillValue': 0
}

# Save as NetCDF
output_nc = dir_in + 'huc12_mask.nc'
ds.to_netcdf(output_nc)

# Create comprehensive lookup table
columns_to_include = [
    'HUC12', 'huc12_id', 'NAME', 'STATES', 
    'AREAACRES', 'AREASQKM', 'HUTYPE', 'HUMOD', 'TOHUC',
    'outlet_comid', 'outlet_lon', 'outlet_lat',
    'natural', 'developed', 'impervious', 'urban',
    'water_sqkm', 'inflow_historical', 'snow_fraction'
]

# Create comprehensive lookup table
lookup_df = gdf_wgs84[columns_to_include].copy()

# Save the lookup table
output_lookup = dir_in + 'huc12_lookup.csv'
lookup_df.to_csv(output_lookup, index=False)

# Print statistics
print("\nMask Statistics:")
print(f"Mask shape: {mask.shape}")
print(f"Non-zero pixels: {np.sum(mask != 0)}")
print(f"Unique HUC12 IDs in mask: {len(np.unique(mask[mask != 0]))}")
print(f"Total HUC12s in original data: {len(gdf)}")
print(f"\nSaved NetCDF to: {output_nc}")
print(f"Saved lookup to: {output_lookup}")

# Verify grid centers
print("\nGrid Verification:")
print(f"First grid center longitude: {lons[0]}")
print(f"Last grid center longitude: {lons[-1]}")
print(f"First grid center latitude: {lats[0]}")
print(f"Last grid center latitude: {lats[-1]}")

# Add verification code
print("\nLookup Table Statistics:")
print(f"Number of columns in lookup table: {len(lookup_df.columns)}")
print("\nLookup Table Columns:")
print(lookup_df.columns.tolist())
print("\nSample from lookup table:")
print(lookup_df.head())

# Verify the orientation
print("\nLatitude Coordinates Check:")
print(f"Northernmost latitude: {lats[0]}")
print(f"Southernmost latitude: {lats[-1]}")

# Quick check of the mask orientation
unique_ids_top = np.unique(mask[0:10, :][mask[0:10, :] != 0])
unique_ids_bottom = np.unique(mask[-10:, :][mask[-10:, :] != 0])
print("\nMask Orientation Check:")
print(f"Number of unique HUC12s in top 10 rows: {len(unique_ids_top)}")
print(f"Number of unique HUC12s in bottom 10 rows: {len(unique_ids_bottom)}")

# Optional: Read back and verify netCDF
ds_check = xr.open_dataset(output_nc)
print("\nNetCDF Check:")
print(ds_check)