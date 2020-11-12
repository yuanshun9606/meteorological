import pandas as pd
import numpy as np
# d1 = {'x': 3, 'y': 7.0}
# d2 = {'y': 1, 'x': 4.5, 'z': 62}
# z = pd.DataFrame([d1,d2])
# print(z)
# print(z[['x']])
np.random.seed(1010)
name1 = ['x1','x2','x3','y']
w= pd.DataFrame(np.random.randn(7,4),columns=name1,index=range(10,17))
# print(w)
# print(w.describe())
# print(w.head())

# print(w-w.iloc[0])
# print(w-w[:1])
# 这两个结果不相等