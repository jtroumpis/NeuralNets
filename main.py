from utilities import readCSV
from nets import *
import numpy as np
import pso,sys

x, y = readCSV('data.csv')
x = np.asarray(x)
y = np.asarray(y)

iterations = 1000

if len(sys.argv)!=1:
    try:
        iterations = int(sys.argv[2])
    except IndexError:
        pass

    if sys.argv[1] == 'prbf':
        selection = 1
    elif sys.argv[1] == 'rbf':
        selection = 2
    elif sys.argv[1]=='ff':
        selection = 3
    elif sys.argv[1]=='srbf':
        selection = 4
    elif sys.argv[1]=='sprbf':
        selection = 5
else:
    print("(1). Polynomial RBF with PSO for (centers, sigmas).")
    print("(2). RBF with PSO for (centers, sigmas, weights).")
    print("(3). Feed-Forward with PSO.")
    print("(4). RBF with k-means.")
    print("(5). Polynomial RBF with k-means.")
    selection = int(input("Please choose an option: "))

if selection==1:
    # print("Starting Polynomial RBF...")
    pso.PSO(x,y,iterations)
elif selection==2:
    # print("Starting RBF swarm...")
    pso.PSO(x,y,iterations,'rbf')
elif selection==3:
    # print("Starting Feed-Forward swarm...")
    pso.PSO(x,y,iterations,'ff')
elif selection==4:
    for c in range(2,20):
        doTheNet(c,x,y)
elif selection==5:
    for c in range(2,20):
        doThePolyNet(c,x,y)
