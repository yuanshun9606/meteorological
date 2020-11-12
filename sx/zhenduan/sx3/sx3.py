from sx.functions import *

# 读取8时资料
data_air = loaddata('D:\mywork\meteorological\sx\zhenduan\sx2\data//air', 4).reshape(11, 7, 37, 73)  # 11个层次，7个时次
data_height = loaddata('D:\mywork\meteorological\sx\zhenduan\sx2\data\height', 4).reshape(11, 7, 37, 73)
data_uv = loaddata('D:\mywork\meteorological\sx\zhenduan\sx2\data//uv', 3).reshape(11, 7, 37 * 2, 73)
data_rh = loaddata('D:\mywork\meteorological\sx\zhenduan\sx2\data//rh', 4).reshape(7, 7, 37, 73)
data_u = data_uv[:, :, 0:37, :]  # u 分量
data_v = data_uv[:, :, 37:, :]  # v 分量


#  重写loaddata
def loaddata(path, n):  # n是跳过的行数
    data = []
    files = os.listdir(path)
    files.sort(key=int)

    for file in files:
        level = path + '/' + file
        positions = os.listdir(level)
        for position in positions:
            if position[8:10] == '12':  # 仅读取指定时次数据
                f = level + '/' + position
                with open(f, 'r', errors='ignore') as f1:
                    lines = f1.readlines()[n:]  # 跳过前多少行
                    for line in lines:
                        linelist = [float(s) for s in line.split()]
                        data.extend(linelist)
    data = np.array(data)
    return data


data_air_20 = loaddata('D:\mywork\meteorological\sx\zhenduan\sx2\data//air', 4).reshape(11, 7, 37, 73)  # 11个层次，7个时次
data_height_20 = loaddata('D:\mywork\meteorological\sx\zhenduan\sx2\data\height', 4).reshape(11, 7, 37, 73)
data_uv_20 = loaddata('D:\mywork\meteorological\sx\zhenduan\sx2\data//uv', 3).reshape(11, 7, 37 * 2, 73)
data_rh_20 = loaddata('D:\mywork\meteorological\sx\zhenduan\sx2\data//rh', 4).reshape(7, 7, 37, 73)
data_u_20 = data_uv_20[:, :, 0:37, :]  # u 分量
data_v_20 = data_uv_20[:, :, 37:, :]  # v 分量

# 各层次散度计算
D8, D20 = [], []
for i in range(10, -1, -1):
    D8.append(f_D(data_u[i, :, :, :], data_v[i, :, :, :]))
    D20.append(f_D(data_u_20[i, :, :, :], data_v_20[i, :, :, :]))
w8, w20 = [], []
deltP = [75, 75, 150, 200, 100, 100, 50, 50, 50, 50]
w8.append(0), w20.append(0)
for i in range(1, 11):
    w8.append(w8[i - 1] + 0.5 * (D8[i] + D8[i - 1]) * deltP[i - 1])
    w20.append(w20[i - 1] + 0.5 * (D20[i] + D20[i - 1]) * deltP[i - 1])
fig, ax = creatmap()
lon, lat = D8[0].lon, D8[0].lat  # 设置范围
lon_range = lon[(lon >= 50) & (lon <= 160)]
lat_range = lat[(lat >= 10) & (lat <= 80)]
# 绘制垂直速度分布
# w8[4].sel(time='2020-06-29', lat=lat_range, lon=lon_range). \
#     plot.contourf(levels=np.arange(-0.015,0.015+0.005,0.005) ,ax=ax, transform=ccrs.PlateCarree())
# plt.title('6月29日08时500hPa 垂直速度')
# plt.savefig('6月29日08时500hPa垂直速度.png')
# plt.show()

# 第二种修正方案
def f_dw(D, w):
    M = 55
    deltw = -w[-1]
    D_, w_ = [], []
    D_.append(D[0]), w_.append(w[0])
    for i in range(1, 11):
        D_.append(D[i] - (i / M / deltP[i - 1]) * deltw)
        w_.append(w[i] - (i * (i + 1) / 2 / M) * deltw)

    return D_, w_


# D_8, w_8 = f_dw(D8, w8)  # 对8时的数据进行第二种方案修正
#
# w_8[4].sel(time='2020-06-29', lat=lat_range, lon=lon_range). \
#     plot.contourf(levels=np.arange(-0.015,0.015+0.005,0.005),ax=ax, transform=ccrs.PlateCarree())
# plt.title('修正后6月29日08时500hPa 垂直速度')
# plt.savefig('修正后6月29日08时500hPa垂直速度.png')
# plt.show()
# 对比后只有少量差异


# 垂直剖面图
w32N = np.empty((11, 7, 73))  # 11个层次  7个时次 73个经向
for i in range(11):
    for j in range(7):
        for k in range(73):
            w32N[i, j, k] = D20[i].values[j, 24, k] +(D20[i].values[j, 23, k] - D20[i].values[j, 24, k]) * 4 / 5
            # 32.5N-30N * (4/5)+32.5N

level = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.2, 0.1, 0.7, 0.8])
ax.invert_yaxis()
ca = ax.contourf(lon[:], level[:], w32N[:, 3, :],cmap='Spectral_r',)
fig.colorbar(ca)
ax.set(xlabel='经度', ylabel='高度/ hPa ', title='6月27日20时沿32°N散度的垂直剖面图')
plt.savefig('6月27日20时沿32°N散度的垂直剖面图.png')
plt.show()

# 绘制高度场
h = toxarray(data_height_20[2, :, :, :])
h = h.sel(time='2020-06-29', lat=lat_range, lon=lon_range).plot.contour(levels=np.arange(1140, 1300, 4), ax=ax,
                                                                        linewidths=1, colors='k',
                                                                        transform=ccrs.PlateCarree())
plt.clabel(h, inline=1, fontsize=10, fmt='%1.0f')

# 高空急流
V = toxarray(np.sqrt(data_v_20[2,:,:,:]**2+data_u_20[2,:,:,:]**2))
V.sel(time='2020-06-29', lat=lat_range, lon=lon_range).plot.contourf(levels=[30,40,50,60,70],ax=ax,cmap=plt.cm.Blues,
                                                                     transform=ccrs.PlateCarree())
# 绘制风场
img_extent = [50, 160, 10, 80]
ax.set_extent(img_extent, crs=ccrs.PlateCarree())
ax.barbs(lon[::3], lat[::3], data_u_20[2, 5, ::3, ::3], data_v_20[2, 5, ::3, ::3],
         linewidth=0.4, flagcolor='k', linestyle='-', length=5,
         pivot='tip', barb_increments=dict(half=2, full=4, flag=20),
         sizes=dict(spacing=0.15, height=0.5, width=0.12), transform=ccrs.PlateCarree())
plt.title('6月29日20时200hPa位势高度场、风场和高空急流图')
plt.savefig('6月29日20时200hPa综合.png')
fig.show()
