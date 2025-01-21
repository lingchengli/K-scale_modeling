import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import glob
import logging
warnings.filterwarnings('ignore', category=RuntimeWarning)

def update_lookup_with_area_stats(lookup_df, mask_ds, urban_mask, grid_areas):
    """
    Update lookup table with pixel count and area statistics for urban and non-urban areas,
    only for HUCs present in the mask dataset
    """
    # Get unique HUCs from mask dataset
    huc_ids = np.unique(mask_ds.huc12_mask.values[mask_ds.huc12_mask.values > 0])
    print(f"\nProcessing area statistics for {len(huc_ids)} HUCs found in mask dataset")
    
    # Create new columns for area statistics
    stats_columns = [
        'pixel_count_total',
        'pixel_count_urban',
        'pixel_count_nonurban',
        'area_total_km2',
        'area_urban_km2',
        'area_nonurban_km2'
    ]
    
    # Initialize a new DataFrame with only the HUCs present in the mask
    filtered_lookup_df = lookup_df[lookup_df['huc12_id'].isin(huc_ids)].copy()
    
    # Initialize new columns with zeros
    for col in stats_columns:
        filtered_lookup_df[col] = 0.0
    
    # Calculate statistics for each HUC
    for idx, row in filtered_lookup_df.iterrows():
        huc_id = row['huc12_id']
        print(f"Processing HUC ID: {huc_id}, Name: {row['HUC12']}")
        
        huc_mask = (mask_ds.huc12_mask == huc_id)
        
        # Calculate urban and non-urban masks for this HUC
        urban_area_mask = huc_mask & urban_mask
        nonurban_area_mask = huc_mask & ~urban_mask
        
        # Get pixel counts
        total_pixels = huc_mask.sum().values
        urban_pixels = urban_area_mask.sum().values
        nonurban_pixels = nonurban_area_mask.sum().values
        
        # Calculate areas (convert from m² to km²)
        total_area = grid_areas.where(huc_mask).sum(['lat', 'lon']).values / 1e6
        urban_area = grid_areas.where(urban_area_mask).sum(['lat', 'lon']).values / 1e6
        nonurban_area = grid_areas.where(nonurban_area_mask).sum(['lat', 'lon']).values / 1e6
        
        # Update lookup table
        filtered_lookup_df.loc[idx, 'pixel_count_total'] = total_pixels
        filtered_lookup_df.loc[idx, 'pixel_count_urban'] = urban_pixels
        filtered_lookup_df.loc[idx, 'pixel_count_nonurban'] = nonurban_pixels
        filtered_lookup_df.loc[idx, 'area_total_km2'] = total_area
        filtered_lookup_df.loc[idx, 'area_urban_km2'] = urban_area
        filtered_lookup_df.loc[idx, 'area_nonurban_km2'] = nonurban_area
        
        # Print statistics for verification
        print(f"  Pixel counts - Total: {total_pixels}, Urban: {urban_pixels}, Non-urban: {nonurban_pixels}")
        print(f"  Areas (km²) - Total: {total_area:.2f}, Urban: {urban_area:.2f}, Non-urban: {nonurban_area:.2f}")
    
    return filtered_lookup_df

def calculate_grid_cell_area(lat, lon, dlat=1/120, dlon=1/120):
    """Calculate grid cell area in m² for each lat/lon point"""
    R_EARTH = 6371000  # Earth's radius in meters
    
    # Create meshgrid for lat/lon
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    
    # Convert degrees to radians
    lat_rad = np.deg2rad(lat_mesh)
    
    # Calculate cell boundaries
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)
    
    # Calculate area using latitude bounds
    area = (R_EARTH**2 * 
            dlon_rad * 
            (np.sin(lat_rad + dlat_rad/2) - np.sin(lat_rad - dlat_rad/2)))
    
    # Convert to xarray DataArray with proper coordinates
    area_da = xr.DataArray(
        area,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon}
    )
    
    return area_da

def convert_model_year_to_real_date(year_str):
    """Convert model year (0001-0021) to real year (2001-2021)"""
    try:
        model_year = int(year_str.lstrip('0'))
        if not (1 <= model_year <= 21):
            raise ValueError(f"Model year {model_year} out of expected range (1-21)")
        real_year = 2000 + model_year
        return real_year
    except ValueError as e:
        print(f"Error converting year {year_str}: {str(e)}")
        raise

def get_overlapping_integer_bounds(mask_ds, elm_ds):
    """Get integer bounds from overlapping region of two datasets"""
    mask_lat_min = mask_ds.lat.min().item()
    mask_lat_max = mask_ds.lat.max().item()
    mask_lon_min = mask_ds.lon.min().item()
    mask_lon_max = mask_ds.lon.max().item()
    
    elm_lat_min = elm_ds.lsmlat.min().item()
    elm_lat_max = elm_ds.lsmlat.max().item()
    elm_lon_min = elm_ds.lsmlon.min().item()
    elm_lon_max = elm_ds.lsmlon.max().item()
    
    print("\nOriginal bounds:")
    print(f"Mask dataset - Lat: {mask_lat_min} to {mask_lat_max}, Lon: {mask_lon_min} to {mask_lon_max}")
    print(f"ELM dataset - Lat: {elm_lat_min} to {elm_lat_max}, Lon: {elm_lon_min} to {elm_lon_max}")
    
    overlap_lat_min = max(mask_lat_min, elm_lat_min)
    overlap_lat_max = min(mask_lat_max, elm_lat_max)
    overlap_lon_min = max(mask_lon_min, elm_lon_min)
    overlap_lon_max = min(mask_lon_max, elm_lon_max)
    
    bounds = {
        'lat_min': np.floor(overlap_lat_min),
        'lat_max': np.ceil(overlap_lat_max),
        'lon_min': np.floor(overlap_lon_min),
        'lon_max': np.ceil(overlap_lon_max)
    }
    
    print("\nOverlapping integer bounds:")
    print(f"Latitude: {bounds['lat_min']} to {bounds['lat_max']}")
    print(f"Longitude: {bounds['lon_min']} to {bounds['lon_max']}")
    
    return bounds

def convert_lon_360_to_180(ds):
    """Convert longitude from 0-360 to -180-180 format"""
    ds = ds.copy()
    if 'lon' in ds.coords:
        lon_name = 'lon'
    else:
        lon_name = 'lsmlon'
    
    lon = ds[lon_name].values
    lon = np.where(lon > 180, lon - 360, lon)
    ds = ds.assign_coords({lon_name: lon})
    
    return ds.sortby(lon_name)

def calculate_streamflow_and_runoff(data, mask, area):
    """Calculate both streamflow (m³/s) and area-averaged runoff (mm/day)"""
    # Initialize empty datasets with zeros and time dimension from input data
    streamflow_result = xr.Dataset({
        var: xr.DataArray(
            np.zeros(len(data.time)),
            dims=['time'],
            coords={'time': data.time}
        ) for var in data.data_vars
    })

    runoff_result = xr.Dataset({
        var: xr.DataArray(
            np.zeros(len(data.time)),
            dims=['time'],
            coords={'time': data.time}
        ) for var in data.data_vars
    })

    # Check if mask has any True values
    if mask.sum() == 0:
        return streamflow_result, runoff_result

    # Calculate total masked area for area-averaging
    total_masked_area = area.where(mask).sum(['lat', 'lon'])

    # Process each variable
    for var in data.data_vars:
        # Replace NaN with 0 in data
        filled_data = data[var].fillna(0)
        
        # Calculate streamflow (m³/s):
        # 1. Convert from mm/s to m/s (divide by 1000)
        # 2. Multiply by grid cell area (m²) to get m³/s
        # 3. Apply mask and sum over spatial dimensions
        streamflow = ((filled_data / 1000) * area).where(mask)
        total_streamflow = streamflow.sum(['lat', 'lon'])
        
        # Calculate area-averaged runoff:
        # 1. Multiply by area for weighted sum
        # 2. Divide by total area to get area-weighted average
        # 3. Convert from mm/s to mm/day
        weighted_runoff = (filled_data * area).where(mask)
        area_avg_runoff = (weighted_runoff.sum(['lat', 'lon']) / total_masked_area) * 86400
        
        streamflow_result[var] = total_streamflow
        runoff_result[var] = area_avg_runoff
        
        print(f"\nStatistics for {var}:")
        print(f"Streamflow - Min: {total_streamflow.min().values:.6f} m³/s, Max: {total_streamflow.max().values:.6f} m³/s, Mean: {total_streamflow.mean().values:.6f} m³/s")
        print(f"Area-averaged runoff - Min: {area_avg_runoff.min().values:.6f} mm/day, Max: {area_avg_runoff.max().values:.6f} mm/day, Mean: {area_avg_runoff.mean().values:.6f} mm/day")
    
    return streamflow_result, runoff_result

def calculate_runoff_statistics(elm_ds, mask_ds, urban_mask, lookup_df, real_year, output_dir, grid_areas):
    """Calculate runoff statistics for all HUCs efficiently using vectorized operations"""
    
    # Calculate runoff components
    Qsurf = (elm_ds.QH2OSFC + elm_ds.QOVER + elm_ds.QRGWL + elm_ds.QSNWCPICE)
    Qsub = (elm_ds.QDRAI + elm_ds.QDRAI_PERCH)
    print('min:', Qsurf.min().values, 'max:', Qsurf.max().values)
    print('min:', Qsub.min().values, 'max:', Qsub.max().values)

    # Combine into dataset
    runoff_ds = xr.Dataset({
        'Qsurf': Qsurf,
        'Qsub': Qsub
    })
    
    # Create dates array
    dates = pd.date_range(start=f"{real_year}-01-01", periods=len(elm_ds.time), freq='D')
    
    # Get unique HUCs
    huc_ids = np.unique(mask_ds.huc12_mask.values[mask_ds.huc12_mask.values > 0])
    print(f"Processing {len(huc_ids)} unique HUCs...")
    
    for huc_id in huc_ids:
        # Get HUC info
        huc_info = lookup_df[lookup_df['huc12_id'] == huc_id].iloc[0]
        print(f"\nProcessing HUC ID: {huc_id}, Name: {huc_info['HUC12']}")
        
        # Create mask for this HUC
        huc_mask = (mask_ds.huc12_mask == huc_id)
        # Create urban and non-urban masks for this HUC
        urban_area_mask = huc_mask & urban_mask
        nonurban_area_mask = huc_mask & ~urban_mask
        print(f"HUC mask sum: {huc_mask.sum().values}")
        print(f"Urban mask sum: {urban_area_mask.sum().values}")
        print(f"Non-urban mask sum: {nonurban_area_mask.sum().values}")

        # Initialize empty datasets
        urban_streamflow = xr.Dataset({
            'Qsurf': xr.DataArray(np.zeros(len(elm_ds.time)), dims=['time'], coords={'time': elm_ds.time}),
            'Qsub': xr.DataArray(np.zeros(len(elm_ds.time)), dims=['time'], coords={'time': elm_ds.time})
        })
        urban_runoff = urban_streamflow.copy()
        nonurban_streamflow = urban_streamflow.copy()
        nonurban_runoff = urban_streamflow.copy()
        
        # Calculate streamflow and runoff only if masks have valid grid cells
        if urban_area_mask.sum() > 0:
            print("Calculating urban streamflow and runoff...")
            urban_streamflow, urban_runoff = calculate_streamflow_and_runoff(
                runoff_ds,
                urban_area_mask,
                grid_areas
            )
        
        if nonurban_area_mask.sum() > 0:
            print("Calculating non-urban streamflow and runoff...")
            nonurban_streamflow, nonurban_runoff = calculate_streamflow_and_runoff(
                runoff_ds,
                nonurban_area_mask,
                grid_areas
            )
        
        # Create DataFrame for this HUC with all timesteps
        huc_df = pd.DataFrame({
            'date': dates,
            'HUC12': huc_info['HUC12'],
            'NAME': huc_info['NAME'],
            'Qsurf_urban_m3s': urban_streamflow.Qsurf.values,
            'Qsub_urban_m3s': urban_streamflow.Qsub.values,
            'Qsurf_nonurban_m3s': nonurban_streamflow.Qsurf.values,
            'Qsub_nonurban_m3s': nonurban_streamflow.Qsub.values,
            'Rsurf_urban_mmday': urban_runoff.Qsurf.values,
            'Rsub_urban_mmday': urban_runoff.Qsub.values,
            'Rsurf_nonurban_mmday': nonurban_runoff.Qsurf.values,
            'Rsub_nonurban_mmday': nonurban_runoff.Qsub.values
        })
        
        print(huc_df.head())
        
        # Save results
        output_file = output_dir / f"HUC12_{huc_info['HUC12']}_{real_year}.csv"
        huc_df.to_csv(output_file, index=False)
        print(f"Saved results for HUC {huc_info['HUC12']} to: {output_file}")
        # save as Parquet
        output_file = output_dir / f"HUC12_{huc_info['HUC12']}_{real_year}.parquet"
        huc_df.to_parquet(output_file)
        print(f"Saved results for HUC {huc_info['HUC12']} to: {output_file}")

def main():
    # Set up paths
    base_dir = Path('/compyfs/bish218/e3sm_scratch')
    run_dir = base_dir / 'conus_1k_small_lat1.f85ea21.fdrain_est.modified_infil_5000_wa.v2.fc_est.daily_runoff/run'
    mask_path = Path('/compyfs/lili400/project/ICoM/conus_urban/huc12_mask.nc')
    lookup_path = Path('/compyfs/lili400/project/ICoM/conus_urban/huc12_lookup.csv')
    surfdata_path = Path('/compyfs/bish218/conus1k/netcdf/surfdata_conus_1k_small_lat1_with_fdrain_and_fc_c240327.nc')
    output_dir = Path('/compyfs/lili400/project/ICoM/conus_urban/runoff_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Load static datasets
        print("\nLoading static datasets...")
        mask_ds = xr.open_dataset(mask_path)
        surf_ds = xr.open_dataset(surfdata_path)
        lookup_df = pd.read_csv(lookup_path)

        # Convert surface dataset coordinates
        surf_ds = convert_lon_360_to_180(surf_ds)
        
        # Get overlapping bounds
        bounds = get_overlapping_integer_bounds(mask_ds, surf_ds)

        # Process datasets with bounds
        mask_ds = mask_ds.sel(
            lat=slice(bounds['lat_min'], bounds['lat_max']),
            lon=slice(bounds['lon_min'], bounds['lon_max'])
        )
        
        surf_ds = surf_ds.drop_vars(['lat', 'lon']).sel(
            lsmlat=slice(bounds['lat_min'], bounds['lat_max']),
            lsmlon=slice(bounds['lon_min'], bounds['lon_max'])
        ).rename({'lsmlat': 'lat', 'lsmlon': 'lon'})
        
        # Calculate grid cell areas
        print("\nCalculating grid cell areas...")
        grid_areas = calculate_grid_cell_area(
            mask_ds.lat.values,
            mask_ds.lon.values
        )

        # Assign the same coordinates to surface dataset
        surf_ds = surf_ds.assign_coords(lat=mask_ds.lat, lon=mask_ds.lon)
        
        # Create urban mask
        print("\nCreating urban mask...")
        urban_pct = surf_ds.PCT_URBAN.sum(dim='density_class')
        urban_mask = urban_pct > 0.1

        # Get unique HUCs
        huc_ids = np.unique(mask_ds.huc12_mask.values[mask_ds.huc12_mask.values > 0])
        ## update lookup table with area stats
        if False:
            print("\nUpdating lookup table with area statistics...")
            updated_lookup_df = update_lookup_with_area_stats(lookup_df.copy(), mask_ds, urban_mask, grid_areas)
   
            output_lookup_path = lookup_path.parent / f"{lookup_path.stem}_with_area_stats.csv"
            updated_lookup_df.to_csv(output_lookup_path, index=False)
            print(f"Saved updated lookup table to: {output_lookup_path}")

            # Print summary statistics
            print("\nArea Statistics Summary:")
            print(f"Total number of HUCs: {len(updated_lookup_df)}")
            print("\nPixel count statistics:")
            print(updated_lookup_df[['pixel_count_total', 'pixel_count_urban', 'pixel_count_nonurban']].describe())
            print("\nArea statistics (km²):")
            print(updated_lookup_df[['area_total_km2', 'area_urban_km2', 'area_nonurban_km2']].describe())
            quit()

        # Process ELM files
        elm_files = sorted(glob.glob(str(run_dir / '*.h1.????-01-01-00000.nc')))
        if not elm_files:
            raise ValueError("No ELM input files found")

        for file_path in elm_files:
            print(f"\nProcessing file: {file_path}")
            
            # Extract year information
            filename = Path(file_path).name
            date_part = filename.split('.')[-2]
            model_year = date_part.split('-')[0]
            real_year = convert_model_year_to_real_date(model_year)

            # Load and preprocess ELM data
            elm_ds = xr.open_dataset(file_path)
            elm_ds = convert_lon_360_to_180(elm_ds)
            elm_ds = elm_ds.sel(
                lat=slice(bounds['lat_min'], bounds['lat_max']),
                lon=slice(bounds['lon_min'], bounds['lon_max'])
            )
            elm_ds = elm_ds.assign_coords(lat=mask_ds.lat, lon=mask_ds.lon)
            
            # Calculate statistics and save results
            calculate_runoff_statistics(elm_ds, mask_ds, urban_mask, lookup_df, real_year, output_dir, grid_areas)

            # Clean up
            elm_ds.close()
            quit()
        print("\nProcessing completed successfully!")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()