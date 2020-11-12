import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

np.set_printoptions(suppress=True)  # 关闭科学计数法
matplotlib.rc("font", family='MicroSoft YaHei', weight='bold', size=12)  # 设置中文

lati = lambda i: (90 - 2.5 * i) * np.pi / 180


# 创建地图对象
def creatmap():
    proj = ccrs.PlateCarree()  # 创建投影
    fig = plt.figure(figsize=(15, 8), dpi=80)  # 创建页面
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})  # 子图
    # 设置地图属性
    # ax.add_feature(cfeat.BORDERS.with_scale('50m'), linewidth=0.8, zorder=1) #国界
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)  # 海岸线
    # 设置网格点属性
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1.2, color='k', alpha=0.5, linestyle='--')
    gl.toplabels_top = False  # 关闭顶端标签
    gl.rightlabels_right = False  # 关闭右侧标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度格式
    return fig, ax


#  批量读取数据
def loaddata(path, n):  # n是跳过的行数
    data = []
    files = os.listdir(path)
    files.sort(key=int)

    for file in files:
        level = path + '/' + file
        positions = os.listdir(level)
        for position in positions:
            if position[8:10] == '00':  # 仅读取指定时次数据
                f = level + '/' + position
                with open(f, 'r', errors='ignore') as f1:
                    lines = f1.readlines()[n:]  # 跳过前多少行
                    for line in lines:
                        linelist = [float(s) for s in line.split()]
                        data.extend(linelist)
    data = np.array(data)
    return data


# 将numpy转化成dataarray
def toxarray(data):
    lon = np.arange(0, 180 + 2.5, 2.5).tolist()
    lat = np.arange(90, -2.5, -2.5).tolist()
    time = pd.date_range("2020-06-24", periods=7)
    da = xr.DataArray(data, coords=[time, lat, lon], dims=['time', 'lat', 'lon'])
    return da


# 计算地转风Vg
def f_Vg(h):
    R = 6371000  # 地球半径
    d = 5 * np.pi * R / 180
    data_ug = np.empty((7, 37, 73))
    data_vg = np.empty((7, 37, 73))
    for k in range(7):  # 时次
        for i in range(0, 37):  # 去掉首尾 (共37行）
            for j in range(0, 73):
                if i == 0 or j == 0 or i == 36 or j == 72:  # 边界处等于 nan
                    data_ug[k, i, j] = np.nan
                    data_vg[k, i, j] = np.nan
                else:
                    data_ug[k, i, j] = -0.67197 / np.sin(lati(i)) * (h[k, i - 1, j] - h[k, i + 1, j]) / d * 1e5
                    data_vg[k, i, j] = 0.67197 / np.sin(lati(i)) * (
                            h[k, i, j + 1] - h[k, i, j - 1]) / d / np.cos(lati(i)) * 1e5

    return toxarray(data_ug), toxarray(data_vg)


data_air = loaddata('data/air', 4).reshape(11, 7, 37, 73)  # 11个层次，7个时次
data_height = loaddata('data/height', 4).reshape(11, 7, 37, 73)
data_uv = loaddata('data/uv', 3).reshape(11, 7, 37 * 2, 73)
data_rh = loaddata('data/rh', 4).reshape(7, 7, 37, 73)
data_u = data_uv[:, :, 0:37, :]  # u 分量
data_v = data_uv[:, :, 37:, :]  # v 分量

data_ug, data_vg = f_Vg(data_height[6, :, :, :])  # 取500hpa的高度场

fig, ax = creatmap()
lon, lat = data_ug.lon, data_ug.lat  # 设置范围
lon_range = lon[(lon > 50) & (lon < 160)]
lat_range = lat[(lat > 10) & (lat < 80)]
# levels = np.arange(-0.0009, 0.001, 0.0001)
cbar_kwargs = {
    'orientation': 'horizontal',  # 水平
    'shrink': 0.5,
    # 'label': '地转风v分量'   # 图例标签
}


# 绘制地转风分量图
# data_vg.sel(time='2020-06-28',lat=lat_range,lon=lon_range).plot.contourf(levels=np.arange(-3,3.5,0.5),ax=ax,cbar_kwargs=cbar_kwargs,transform=ccrs.PlateCarree())
#
# # data_vg.sel(time='2020-06-29', lat=lat_range, lon=lon_range).plot.contourf( ax=ax,levels=np.arange(-60,40,10),
# #                                                                             cbar_kwargs=cbar_kwargs, transform=ccrs.PlateCarree())
# plt.title('6月28日地转风v分量')
# plt.savefig('vg0628.png')


# 计算涡度
def f_wodu(u, v):
    R = 6371000  # 地球半径
    deltd = 2.5 * np.pi / 180
    wodu = np.empty((7, 37, 73))
    for k in range(7):
        for i in range(37):
            for j in range(73):
                if i == 0 or j == 0 or i == 36 or j == 72:  # 边界处等于 nan
                    wodu[k, i, j] = np.nan
                else:
                    wodu[k, i, j] = (((v[k, i, j + 1] - v[k, i, j - 1]) / np.cos(lati(i)) / deltd) -
                                     ((u[k, i - 1, j] - u[k, i + 1, j]) / deltd) + (
                                             2 * u[k, i, j] * np.tan(lati(i)))) / (2 * R)
    return wodu


#  涡度计算绘图
# x = f_wodu(data_u[8, :, :, :], data_v[8, :, :, :])
# x= toxarray(x)
# x.sel(time='2020-06-28', lat=lat_range, lon=lon_range).plot.contourf(
#           ax=ax,cbar_kwargs=cbar_kwargs,transform=ccrs.PlateCarree())
# plt.title('850hpa涡度28日')
# plt.savefig('850hpa涡度28日.png')
# 计算相对涡度平流
def f_wdpl(u, v):
    wodu = f_wodu(u, v)
    d = 2 * 2.5 * np.pi / 180 * 6371000
    wdpl = np.empty((7, 37, 73))
    for k in range(7):
        for i in range(37):
            for j in range(73):
                if i == 0 or j == 0 or i == 36 or j == 72:  # 边界处等于 nan
                    wdpl[k, i, j] = np.nan
                else:
                    wdpl[k, i, j] = -(
                            u[k, i, j] * (wodu[k, i, j + 1] - wodu[k, i, j - 1]) / d / np.cos(lati(i))) + \
                                    (v[k, i, j] * (wodu[k, i - 1, j] - wodu[k, i + 1, j]) / d)
    return toxarray(wdpl)


# 计算相对涡度平流
# wdpl = f_wdpl(data_u[6, :, :, :], data_v[6, :, :, :])
# wdpl.sel(time='2020-06-29', lat=lat_range, lon=lon_range).plot.contourf(ax=ax,cbar_kwargs=cbar_kwargs, transform=ccrs.PlateCarree())
# plt.title('500hPa涡度平流29日')
# plt.savefig('500hPa涡度平流29日.png')
# 计算散度
def f_D(u, v):
    R = 6371000  # 地球半径
    deltd = 2.5 * np.pi / 180
    D = np.empty((7, 37, 73))
    for k in range(7):
        for i in range(37):
            for j in range(73):
                if i == 0 or j == 0 or i == 36 or j == 72:  # 边界处等于 nan
                    D[k, i, j] = np.nan
                else:
                    D[k, i, j] = (((u[k, i, j + 1] - u[k, i, j - 1]) / np.cos(lati(i)) / deltd) +
                                  ((v[k, i - 1, j] - v[k, i + 1, j]) / deltd) - (
                                          2 * v[k, i, j] * np.tan(lati(i)))) / (2 * R)
    return toxarray(D)


#
# d = f_D(data_u[2, :, :, :], data_v[2, :, :, :])
# d.sel(time='2020-06-28', lat=lat_range, lon=lon_range).plot.contourf(ax=ax, cbar_kwargs=cbar_kwargs,
#                                                                      transform=ccrs.PlateCarree())
# plt.title('200hPa散度28日')
# plt.savefig('200hPa散度28日.png')

# 计算温度平流
def f_T(T, u, v):
    d = 2 * 2.5 * np.pi / 180 * 6371000
    vt = np.empty((7, 37, 73))
    for k in range(7):
        for i in range(37):
            for j in range(73):
                if i == 0 or j == 0 or i == 36 or j == 72:  # 边界处等于 nan
                    vt[k, i, j] = np.nan
                else:
                    vt[k, i, j] = -(u[k, i, j] * (T[k, i, j + 1] - T[k, i, j - 1]) / d / np.cos(lati(i)) +
                                    v[k, i, j] * (T[k, i - 1, j] - T[k, i + 1, j]) / d)

    return toxarray(vt)


# vt = f_T(data_air[8,:,:,:],data_u[8, :, :, :], data_v[8, :, :, :])
# vt.sel(time='2020-06-29', lat=lat_range, lon=lon_range).plot.contourf(ax=ax,cbar_kwargs=cbar_kwargs,
#                                                                      transform=ccrs.PlateCarree())
# plt.title('850hPa温度平流29日')
# plt.savefig('850hPa温度平流29日.png')
# 500hpa 温压场


# h = toxarray(data_height[8, :, :, :])
# h = h.sel(time='2020-06-29', lat=lat_range, lon=lon_range).plot.contour(levels=np.arange(120, 160, 4), ax=ax,
#                                                                         linewidths=2,
#                                                                         colors='k', transform=ccrs.PlateCarree())
# plt.clabel(h, inline=1, fontsize=10, fmt='%1.0f')
# t = toxarray(data_air[8, :, :, :])
# t = t.sel(time='2020-06-29', lat=lat_range, lon=lon_range).plot.contour(levels=np.arange(-60, 60, 4), ax=ax, colors='r',linestyles='--',
#                                                                         transform=ccrs.PlateCarree())
# plt.clabel(t, inline=1, fontsize=10, fmt='%1.0f', colors='red')
# plt.title('2020-06-29 8时  850hPa温压场配置', size=20)

plt.show()
