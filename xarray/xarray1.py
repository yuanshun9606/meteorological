import xarray as xr
import numpy as np
import pandas as pd
# 设置数据
np.random.seed(1)  # 随机种子
temp = 15 + 8 * np.random.randn(2, 2, 3)  # 温度
precip = 10 * np.random.rand(2, 2, 3)  # 降水
lon = [[-99.83, -99.32], [-99.79, -99.23]]
lat = [[42.25, 42.21], [42.63, 42.59]]

# 创建DataSet  传入两个字典作为参数
ds = xr.Dataset(
    {
        'temperature': (['x', 'y', 'time'], temp),
        'precipitation': (['x', 'y', 'time'], precip)
    },  # 设置data_vars 传入字典每个键作为变量的名称，每个值作为以下之一
    coords={
        'lon': (['x', 'y'], lon),
        'lat': (['x', 'y'], lat),
        'time':pd.date_range('2014-09-06', periods=3),
        'reference_time': pd.Timestamp('2014-09-05'),
    }, # 设置coors
)
print(ds)