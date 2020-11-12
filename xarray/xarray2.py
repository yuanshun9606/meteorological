import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt


def mask(ds, label='land'):
    landsea = xr.open_dataset('landsea.nc').sel(time='1989-01-01T12:00:00')['lsm'] #降维处理

    # --ds和地形数据分辨率不一致，需将地形数据插值
    landsea = landsea.interp(latitude=ds.lat.values, longitude=ds.lon.values)
    # --利用地形掩盖海陆数据
    ds.coords['mask'] = (('lat', 'lon'), landsea.values)

    if label == 'land':
        ds = ds.where(ds.mask < 1)
    elif label == 'ocean':
        ds = ds.where(ds.mask > 0)
    return ds


def create_map():
    # 创建画图空间
    proj = ccrs.PlateCarree()  # 创建坐标系
    fig = plt.figure()  # 创建页面
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})  # 创建子图
    # 设置地图属性
    ax.add_feature(cfeat.BORDERS.with_scale('50m'), linewidth=0.8)  # 加载分辨率为50的国界线
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6)  # 加载分辨率为50的海岸线
    # ax.add_feature(cfeat.RIVERS.with_scale('50m'))  # 加载分辨率为50的河流
    # ax.add_feature(cfeat.LAKES.with_scale('50m'))  # 加载分辨率为50的湖泊
    # 设置网格点属性
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1.2, color='k', alpha=0.5, linestyle='--')
    gl.toplabels_top = False  # 关闭顶端的经纬度标签
    gl.rightlabels_right = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    return fig, ax


if __name__ == '__main__':
    # 数据读取及时间平均处理
    fig, ax = create_map()
    ds = xr.open_dataset('../air.mon.mean.nc')
    lat = ds.lat
    lon = ds.lon
    time = ds.time
    temp = ds.air
    # 区域选择
    lon_range = lon[(lon > 70) & (lon < 140)]
    lat_range = lat[(lat > 0) & (lat < 60)]

    temp_region = temp.sel(lon=lon_range, lat=lat_range, time='2000-02-01', level='925.0')

    temp_mask = mask(temp_region, 'ocean')
    # --画图
    # 图例
    cbar_kwargs = {
        'label': '2m temperature (℃)',
        'ticks': np.arange(-30, 30 + 5, 5), }


    levels = np.arange(0, 50 + 1, 1)
    temp_mask.plot.contourf(ax=ax, levels=levels, cmap='Spectral_r', cbar_kwargs=cbar_kwargs,
                            transform=ccrs.PlateCarree())
    fig.show()
    # plt.savefig('xarray.png')
    print(ds.Coordinates)