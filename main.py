# import math
from utilities import *
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()  # for plot styling
import numpy as np
# from numpy.linalg import inv
import pso

print("(1). Polynomial RBF with PSO for (centers, sigmas).")
print("(2). RBF with PSO for (centers, sigmas, weights).")
selection = int(input("Please choose an option: "))

x, y = readCSV('data.csv')
x = np.asarray(x)
y = np.asarray(y)

if selection==1:
    print("Starting Polynomial RBF...")
    pso.PSO_PRBF(x,y)
elif selection==2:
    print("Starting RBF swarm...")

elif selection==3:
    print("Feed-Forward under construction...")

# print(str(sys.argv))
#
# exit()

# lamda = 1



# print(x.shape)
# for lamda in [0.1,1,10,100]:
#     print("LAMDA=",lamda)
# for c in range(2,30):
#     doThePolyNet(lamda,c,x,y)
