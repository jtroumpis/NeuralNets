from utilities import readCSV
from nets import *
import numpy as np
import pso,sys

x, y = readCSV('data.csv')
x = np.asarray(x)
y = np.asarray(y)

iterations = 1000

if len(sys.argv)!=1:
    if sys.argv[2]:
        iterations = int(sys.argv[2])

    if sys.argv[1] == 'prbf':
        print("Starting Polynomial RBF...")
        pso.PSO_RBF(x,y,iterations)
    elif sys.argv[1] == 'rbf':
        print("Starting RBF swarm...")
        pso.PSO_RBF(x,y,iterations,False)
    elif sys.argv[1]=='ff':
        print("Feed-Forward under construction...")
    elif sys.argv[1]=='srbf':
        for c in range(2,20):
            doTheNet(c,x,y)
    elif sys.argv[1]=='sprbf':
        for c in range(2,20):
            doThePolyNet(c,x,y)
    exit()


print("(1). Polynomial RBF with PSO for (centers, sigmas).")
print("(2). RBF with PSO for (centers, sigmas, weights).")
print("(3). Feed-Forward with PSO.")
print("(4). RBF with k-means.")
print("(5). Polynomial RBF with k-means.")
selection = int(input("Please choose an option: "))

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
