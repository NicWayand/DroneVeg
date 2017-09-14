import xarray as xr
from skimage.filters import threshold_otsu
import numpy as np
import os
import scipy
# Estimates Vegetation presence, Vegetation Height, and LAI from RGB and Digital Surface Model (DSM).

# Function to write out an ndarry as a GeoTIFF using the spatial references of a sample geotif file
def write_GeoTif_like(templet_tif_file, output_ndarry, output_tif_file):
    import rasterio
    orig = rasterio.open(templet_tif_file)
    with rasterio.open(output_tif_file, 'w', driver='GTiff', height=output_ndarry.shape[0],
                       width=output_ndarry.shape[1], count=1, dtype=output_ndarry.dtype,
                       crs=orig.crs, transform=orig.transform, nodata=-9999) as dst:
        dst.write(output_ndarry, 1)

# INPUT FILES
data_dir = r'F:\Work\e\Data\LIDAR\UAV\Fortress-20170718' # Directory were input and output data is/will-go
# DSM derived through photogrametry
DSM_Tif         = os.path.join(data_dir,'20170718-DSM-Clipped.tif')
# rgb imagery from drone
rgb_Tif         = os.path.join(data_dir,'20170718-mosaic-Clipped.tif')

# OUTPUT FILES
DEM_Tif         = os.path.join(data_dir,'DEM.tif')
VEG_Tif         = os.path.join(data_dir,'VegHeight.tif')
VEG_mask_Tif    = os.path.join(data_dir,'VegMask.tif')
LAI_Tif         = os.path.join(data_dir,'LAI.tif')

# Load stuff in
ds_dsm  = xr.open_rasterio(DSM_Tif, chunks={'x':1000, 'y':1000}).sel(band=1).drop('band')
ds_dsm  = ds_dsm.where(ds_dsm>0).astype(np.float32).load() # da_VARI = xr.open_dataarray(r'F:\Work\e\Data\LIDAR\UAV\Fortress-20170718\VARI.nc', chunks=10)
ds_rgb  = xr.open_rasterio(rgb_Tif, chunks={'x':1000, 'y':1000}) # drop band 4 which is empty
ds_rgb = ds_rgb.where(ds_rgb.sel(band=1)>0).astype(np.float32)
print("loaded")

# GRVI (Green Red Vegetation Index) -1 (not green) to 1 (green)
ds_RG = ( ds_rgb.sel(band=2) - ds_rgb.sel(band=1) ) / ( ds_rgb.sel(band=2) + ds_rgb.sel(band=1) )
print("Calculated GRVI")

# Threshold vegetation from non-vegetation
X = ds_RG.values.flatten()
X = X[np.isfinite(X)]
thresh = threshold_otsu(X)
X = None
print(thresh)
veg = (ds_RG > thresh)
print("Calculated VEG binary")

max_LAI = 3 # Max LAI for GRVI value of 1 (all green)
ds_LAI = ds_RG.where(veg)*max_LAI
print("Estimated LAI from GRVI")

# Minimum Filter approach
window_meters = 11 # m # should be odd, so center of filter is one pixel
pixel_meters = 0.05 # m/pixel
window_pixels = window_meters / pixel_meters
dsm_min = scipy.ndimage.minimum_filter(ds_dsm.values, size=window_pixels, mode='reflect')

print("filter applied.")
ds_DEM = xr.DataArray(dsm_min, coords=ds_dsm.coords, dims=ds_dsm.dims)

# Estimate Veg height as difference between DSM and DEM
# Set missing any edges where DSM or DEM are missing
# Set non-veg pixels to 0 vegetation height
ds_veg_H = (ds_dsm - ds_DEM).where((ds_dsm.notnull()) & (ds_DEM.notnull())).values.astype(np.float32) # .where(veg)
ds_veg_H[ds_veg_H<0] = 0 # Remove edge pixels that were less than 0
ds_veg_H[~veg.values] = 0 # Set non-vegetation to zero height

# Trim DEM to match vegheight extent
ds_DEM_out = ds_DEM.where((ds_dsm.notnull()) & (ds_DEM.notnull()))

# Trim LAI extent to match vegheight exent
ds_LAI_out = ds_LAI.where((ds_dsm.notnull()) & (ds_DEM.notnull())).values
ds_LAI_out[~veg.values] = 0 # Set non-vegetation to zero LAI

print("Writing out DEM Tif.")
write_GeoTif_like(DSM_Tif, ds_DEM_out.values, DEM_Tif)
print("Writing out VEG Height Tif.")
write_GeoTif_like(DSM_Tif, ds_veg_H, VEG_Tif)
print("Writing out VEG Mark Tif.")
write_GeoTif_like(DSM_Tif, veg.astype(int).values, VEG_mask_Tif)
print("Writing out LAI Tif.")
write_GeoTif_like(DSM_Tif, ds_LAI_out, LAI_Tif)
