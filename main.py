# import math
from utilities import *
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()  # for plot styling
import numpy as np
# from numpy.linalg import inv
import pso

print("(1). Polynomial RBF with PSO for (centers, sigmas).")
print("(2). RBF with PSO for (centers, sigmas, weights).")
print("(3). Feed-Forward with PSO.")
print("(4). RBF with k-means.")
print("(5). Polynomial RBF with k-means.")
selection = int(input("Please choose an option: "))

x, y = readCSV('data.csv')
x = np.asarray(x)
y = np.asarray(y)

lamda=1

if selection==1:
    print("Starting Polynomial RBF...")
    pso.PSO_RBF(x,y)
elif selection==2:
    print("Starting RBF swarm...")
    pso.PSO_RBF(x,y,False)
elif selection==3:
    print("Feed-Forward under construction...")
elif selection==4:
    for c in range(2,20):
        doTheNet(c,x,y)
elif selection==5:
    for c in range(2,20):
        doThePolyNet(c,x,y)
