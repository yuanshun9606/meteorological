import xarray as xr

ds= xr.open_dataset('landsea.nc')
print(ds.sel(time='1989-01-01T12:00:00'))