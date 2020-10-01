import numpy as np
import scipy
from matplotlib import pyplot as plt
from netCDF4 import Dataset, num2date
import datetime
# from mpl_toolkits.basemap import Basemap #不能和cartopy同时导入
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

f_test = xr.open_dataset('air.mon.mean.nc')  # 1948-2017 1月平均气温
# print(f_test) # nc数据信息
air = f_test['air']  # 抽取变量air
# print(air)
a = air.isel(level=1).mean(dim='time') #索引
# print(a)

#  basemap地图测试
# plt.figure(figsize=(8, 8))
#
# m = Basemap(projection='ortho', resolution=None,
#             lat_0=50, lon_0=-100)
# m.bluemarble(scale=0.5)

proj = ccrs.PlateCarree()  # 创建投影
fig = plt.figure(figsize=(20, 10),dpi=80)  # 创建页面
ax = fig.subplots(1, 1, subplot_kw={'projection': proj})  # 子图
# 设置地图属性
ax.add_feature(cfeat.BORDERS.with_scale('50m'), linewidth=0.8, zorder=1)
ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
ax.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=1)
ax.add_feature(cfeat.LAKES.with_scale('50m'), zorder=1)
# 设置网格点属性
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1.2, color='k', alpha=0.5, linestyle='--')
gl.toplabels_top = False  # 关闭顶端标签
gl.rightlabels_right = False  # 关闭右侧标签
gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度格式
gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度格式
# 设置colorbar
cbar_kwargs = {
    'orientation': 'horizontal',
    'label': '2m temperature (℃)',
    'shrink': 0.8,
    'ticks': np.arange(-30, 30 + 5, 5)
}
levels = np.arange(-30,30+1,1)
a.plot.contourf(ax=ax, levels=levels, cmap='Spectral_r',
    cbar_kwargs=cbar_kwargs, transform=ccrs.PlateCarree())

plt.show()
