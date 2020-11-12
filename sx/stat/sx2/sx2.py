import xarray as xr
import pandas as pd
import numpy as np
import matplotlib
import pingouin as pg
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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


def toxarray(data):
    lon = np.arange(72.5, 135 + 2.5, 2.5).tolist()
    lat = np.arange(15.5, 55.5 + 2.5, 2.5).tolist()
    da = xr.DataArray(data, coords=[lat, lon], dims=['lat', 'lon'])
    return da


wpsh = np.loadtxt('WPSH.txt', dtype=float)
wpsh = wpsh[np.where(((wpsh[:, 0] >= 1961) & (wpsh[:, 0] <= 2018)) &
                     ((wpsh[:, 1] == 6) | (wpsh[:, 1] == 7) | (wpsh[:, 1] == 8)))]  # 取1961-2018夏季数值
w, p, s, h = np.mean(wpsh[:, 2].reshape(-1, 3), axis=1), np.mean(wpsh[:, 3].reshape(-1, 3), axis=1), \
             np.mean(wpsh[:, 4].reshape(-1, 3), axis=1), np.mean(wpsh[:, 5].reshape(-1, 3), axis=1)
# 取夏季平均   面积指数，强度指数，脊线位置，西伸脊点
x = np.vstack((w, p, s, h))

# print(np.corrcoef(x))  # 打印相关系数矩阵

# f = xr.open_dataset('../sx1/sx1.nc')
# pre = f['pre'].values
# print(pre.shape)
# ww, pp, ss, hh = np.empty((17, 26)), np.empty((17, 26)), np.empty((17, 26)), np.empty((17, 26))
# for i in range(17):
#     for j in range(26):
#         ww[i, j] = np.corrcoef(pre[:, i, j], w)[0, 1]
#         pp[i, j] = np.corrcoef(pre[:, i, j], p)[0, 1]
#         ss[i, j] = np.corrcoef(pre[:, i, j], s)[0, 1]
#         hh[i, j] = np.corrcoef(pre[:, i, j], h)[0, 1]
#
# ww,pp,ss,hh=toxarray(ww),toxarray(pp),toxarray(ss),toxarray(hh)

# fig1, ax = creatmap()
# ww.plot.contourf(levels=[-0.259,0,0.259],ax=ax,cmap='Spectral_r',  transform=ccrs.PlateCarree()) #cmap='Greys', cbar_kwargs=cbar_kwargs,
# plt.title('面积指数与中国夏季降水（1961-2018）的相关系数的空间分布')
# plt.savefig('D:\mywork\mathstduy\实习报告\面积指数.png')
#
# fig2, ax = creatmap()
# pp.plot.contourf(levels=[-0.259,0,0.259],ax=ax,cmap='Spectral_r',  transform=ccrs.PlateCarree()) #cmap='Greys', cbar_kwargs=cbar_kwargs,
# plt.title('强度指数与中国夏季降水（1961-2018）的相关系数的空间分布')
# plt.savefig('D:\mywork\mathstduy\实习报告\强度指数.png')
#
# fig3, ax = creatmap()
# ss.plot.contourf(levels=[-0.259,0,0.259],ax=ax,cmap='Spectral_r',  transform=ccrs.PlateCarree()) #cmap='Greys', cbar_kwargs=cbar_kwargs,
# plt.title('脊线位置与中国夏季降水（1961-2018）的相关系数的空间分布')
# plt.savefig('D:\mywork\mathstduy\实习报告\脊线位置.png')
#
# fig4, ax = creatmap()
# hh.plot.contourf(levels=[-0.259,0,0.259],ax=ax,cmap='Spectral_r',  transform=ccrs.PlateCarree()) #cmap='Greys', cbar_kwargs=cbar_kwargs,
# plt.title('西伸脊点与中国夏季降水（1961-2018）的相关系数的空间分布')
# plt.savefig('D:\mywork\mathstduy\实习报告\西伸脊点.png')

data = {
    'w': w,
    'p': p,
    's': s,
    'h': h,
}
df = pd.DataFrame(data, columns=['w', 'p', 's', 'h'])

c = df.corr()


# #  偏相关系数矩阵
# # pg.partial_corr(data=df, x='w', y='p', covar=['s','h'])
# print(df.pcorr())    #二阶


# 阴影是置信区间
# fig = plot_acf(w,lags=10,title='面积指数')
# plt.savefig('面积指数自相关系数.png')
# fig = plot_acf(p,lags=10,title='强度指数')
# plt.savefig('强度指数自相关系数.png')
# fig = plot_acf(s,lags=10,title='脊线位置')
# plt.savefig('脊线位置自相关系数.png')
# fig = plot_acf(h,lags=10,title='西伸脊点')
# plt.savefig('西伸脊点自相关系数.png')

# 落后交差相关系数
def r_xyj(x, y, j, n):
    s = 0
    _x = x.mean()
    _y = y.mean()
    if j > 0:
        for t in range(0, n - j):
            s += ((x[t] - _x) * (y[t + j] - _y))
    else:
        for t in range(-j, n):
            s += ((x[t] - _x) * (y[t + j] - _y))
    s = s / (n - j)
    return s / x.std() / y.std()


x = np.arange(-5, 6)
r_wpj, r_wsj, r_whj, r_psj, r_phj, r_shj = [], [], [], [], [], []
for i in range(-5, 6):
    r_wpj.append(r_xyj(w, p, i, 58))
    r_wsj.append(r_xyj(w, s, i, 58))
    r_whj.append(r_xyj(w, h, i, 58))
    r_psj.append(r_xyj(p, s, i, 58))
    r_phj.append(r_xyj(p, h, i, 58))
    r_shj.append(r_xyj(s, h, i, 58))

plt.figure(23)
plt.subplot(231), plt.stem(x, r_wpj), plt.title('w & p'), plt.axhline(y=0.259, ls=":", c="red")
plt.subplot(232), plt.stem(x, r_wsj), plt.title('w & s'), plt.axhline(y=-0.259, ls=":", c="red")
plt.subplot(233), plt.stem(x, r_whj), plt.title('w & h'), plt.axhline(y=-0.259, ls=":", c="red")
plt.subplot(234), plt.stem(x, r_psj), plt.title('p & s'), plt.axhline(y=-0.259, ls=":", c="red")
plt.subplot(235), plt.stem(x, r_phj), plt.title('p & h'), plt.axhline(y=-0.259, ls=":", c="red")
plt.subplot(236), plt.stem(x, r_shj), plt.title('s & h'), plt.axhline(y=0.259, ls=":", c="red")
plt.suptitle('各指数间落后交叉相关系数')
plt.show()
