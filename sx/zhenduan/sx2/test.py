import os
import numpy as np
import xarray as xr
import pandas as pd

# np.set_printoptions(suppress=True) # 关闭科学计数法
# data = []
# with open('data/air/100/2020062400.txt', 'r',errors='ignore') as f1:
#     lines = f1.readlines()[4:]  # 跳过前多少行
#     for line in lines:
#         linelist = [float(s) for s in line.split()]
#         data.extend(linelist)
# data = np.array(data)
# print(data.reshape(73,-1).shape)
#
# dataug = np.empty((4,3))
# for i in range(0,4):
#     for j in range(0,3):
#         if i == 0 or j == 0 or i ==3:
#             dataug[i,j] = np.nan
#         else:
#             dataug[i,j]=i+j
#
# print(dataug)
# import numpy as np

#
# def fill(t):
#     for i in range(t.shape[1]):
#         temp_col = t[:, i]
#         nan_num = np.count_nonzero(temp_col != temp_col)
#
#         if nan_num != 0:
#             temp_not_nan_col = temp_col[temp_col == temp_col]
#
#             temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
#     return t
#
#
# t = np.arange(24).reshape((4, 6)).astype("float")
# t[1, 2:] = np.nan
# print(t)
# t = fill(t)
#
# print(t)
print(np.tan(np.pi/4))

