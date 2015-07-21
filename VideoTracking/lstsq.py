import numpy as np
import matplotlib.pyplot as plt

x= np.array([0,1,2,3])
y= np.array([-1,0.2,0.9,2.1])
z= np.array([2,3,4,5])
A = np.vstack([x,np.ones(len(x))]).T

# print A

m = np.linalg.lstsq(A, y)
n = np.linalg.lstsq(A, z)
print m
print n

# print m,c

# plt.plot(x,y,'o', markersize= 10)
# plt.plot(x,m*x +c,'r')

# plt.show()