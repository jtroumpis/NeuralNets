import math
from utilities import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from numpy.linalg import inv

lamda = 1

x, y = readCSV('data.csv')
x = np.asarray(x)
print(np.amin(x, axis=0))
print(np.amax(x, axis=0))
y = np.asarray(y)

# print(x.shape)
# for lamda in [0.1,1,10,100]:
#     print("LAMDA=",lamda)
for c in range(2,30):
    doThePolyNet(lamda,c,x,y)
