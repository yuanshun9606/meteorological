import numpy as np
from scipy.special import comb
x = np.array([1, 2, 1, 3, 3])
y = np.array([-1, 1, 0, 2, 3])

f = lambda p,r,n: comb(n,r)*(p**r)*((1-p)**(n-r))
s=0
for r in range(10,16):
    s += f(16/35,r,15)

print(s)