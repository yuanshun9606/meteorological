import xarray as xr
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)  # 关闭科学计数法
matplotlib.rc("font", family='MicroSoft YaHei', weight='bold', size=12)  # 设置中文
f = xr.open_dataset('sx1.nc')
pre = f['pre']

d = pre.mean(axis=-1).values
# 计算权重数组
coslat = lambda i: np.cos(4 * np.arctan(1) / 180 * i)
w=[]
for i in range(6,23):
   w.append(coslat((i*2.5+0.5)))
# 创建蒙版数组 目的:忽略nan值得影响 average函数不会忽略nan
masked_data = np.ma.masked_array(d, np.isnan(d))
average = np.ma.average(masked_data, axis=-1, weights=w)
y = (d - d.mean()) / d.std() # 标准化

# data = np.array([[1,2,3], [4,5,np.NaN], [np.NaN,6,np.NaN], [0,0,0]])
#
# masked_data = np.ma.masked_array(data, np.isnan(data))
# #calculate your weighted average here instead
#
# weights = [1, 1, 1]
# average = np.ma.average(masked_data, axis=1, weights=weights)
# #this gives you the result
# result = average.filled(np.nan)
# print(result)