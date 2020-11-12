import xarray as xr
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

np.set_printoptions(suppress=True)  # 关闭科学计数法
matplotlib.rc("font", family='MicroSoft YaHei', weight='bold', size=12)  # 设置中文


def creatmap():
    proj = ccrs.PlateCarree()  # 创建投影
    fig = plt.figure(figsize=(15, 8), dpi=80)  # 创建页面
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})  # 子图
    # 设置地图属性
    ax.add_feature(cfeat.BORDERS.with_scale('50m'), linewidth=0.8, zorder=1)
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
    # 设置网格点属性
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1.0, color='k', alpha=0.5, linestyle='--')
    gl.toplabels_top = False  # 关闭顶端标签
    gl.rightlabels_right = False  # 关闭右侧标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度格式
    return fig, ax


# 读取数据
f = xr.open_dataset('sx1.nc')
pre = f['pre']

# 计算统计量
a = pre.isel(time=slice(20, 50)).mean(dim='time')  # 平均
b = (pre.sel(time='2010-07-16T12:00:00') - a) / a  # 距平百分率
c = pre.isel(time=slice(20, 50)).std(dim='time')  # 均方差

# 计算区域平均
d = pre.mean(axis=-1).values
# 计算权重数组
coslat = lambda i: np.cos(4 * np.arctan(np.pi/180) / 180 * i)
w = []
for i in range(6, 23):
    w.append(coslat((i * 2.5 + 0.5)))
# 创建蒙版数组 目的:忽略nan值得影响 average函数不会忽略nan
masked_data = np.ma.masked_array(d, np.isnan(d))
d = np.ma.average(masked_data, axis=-1, weights=w)
# 标准化

y = (d - np.mean(d[20:50])) / np.std(d[20:50])  # 区域平均标准化时间序列
# 绘制降水量气候态
fig1, ax = creatmap()
levels = np.arange(0, 1200 + 100, 100)
cbar_kwargs = {
    'orientation': 'horizontal',  # 水平
    'shrink': 0.8,
    'ticks': levels,
    'label': '降水气候态'
}
a.plot.contourf(levels=levels, ax=ax, cmap='Greys', cbar_kwargs=cbar_kwargs, transform=ccrs.PlateCarree())
plt.title('我国夏季总降水量气候态（1981-2010）的空间分布')
fig1.show()
plt.savefig('sx1_1.png')

# 绘制距平百分率
fig2, ax = creatmap()
levels = np.arange(-1.8, 1.8, 0.2)
cbar_kwargs = {
    'orientation': 'horizontal',  # 水平
    'shrink': 0.8,
    'ticks': levels,
    'label': '距平百分率'
}
b.plot.contourf(levels=levels, ax=ax, cmap='Greys', cbar_kwargs=cbar_kwargs, transform=ccrs.PlateCarree())
plt.title('2017年我国夏季总降水量相对于气候态的降水距平百分率空间分布')
fig2.show()
plt.savefig('sx1_2.png')
# 绘制均方差
fig3, ax = creatmap()
levels = np.arange(0, 300, 20)
cbar_kwargs = {
    'orientation': 'horizontal',  # 水平
    'shrink': 0.8,
    'ticks': levels,
    'label': '降水量均方差'
}

c.plot.contourf(levels=levels, ax=ax,cmap='Greys', cbar_kwargs=cbar_kwargs, transform=ccrs.PlateCarree())
fig3.show()
plt.savefig('sx1_3.png')
# 绘制区域平均序列
fig = plt.figure()
x = np.arange(1961, 2019)
_y = np.ones(58)
plt.plot(x, _y, c='r')
plt.plot(x, -_y, c='r')
plt.plot(x, y)
plt.grid(axis='y')
plt.title('我国区域平均的夏季总降水量的时间变化序列')
plt.show()
plt.savefig('sx1_4.png')
lao, han = np.where(y >= 1)[0], np.where(y <= -1)[0]
print('涝年：', lao + 1961, '\n旱年：', han + 1961)
