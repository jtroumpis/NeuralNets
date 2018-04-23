from utilities import readCSV
from nets import *
import numpy as np
import pso,sys

import argparse


parser = argparse.ArgumentParser(description='Welcome my friend.')

parser.add_argument('-s --SELECT', action="store_true", dest='select',
                    help='CHOOSE THE NN')
parser.add_argument('-n', action="store", dest='nn', default='prbf',
                    help='which NN should be used: options (prbf, rbf, ff)')
parser.add_argument('-c', action="store", dest='n_clusters', type=int, default=10,
                    help='The number of clusters that should be used')
parser.add_argument('-i', action="store", dest='iterations', type=int, default=1000,
                    help='The number of iterations')
parser.add_argument('-q --QUIET', action="store_true", dest='quiet',
                    help='Does not print anything but gBests')
parser.add_argument('-e --EXPLICIT', action="store_true", dest='explicit',
                    help='Prints fitness for each particle every iteration')
args = parser.parse_args()
# print(args.accumulate(args.integers))


# print(args.nn)

x, y = readCSV('data.csv')
x = np.asarray(x)
y = np.asarray(y)

iterations = args.iterations
n_clusters = args.n_clusters
nn_type = args.nn
quiet = args.quiet
expl = args.explicit

if args.select:
    print("(prbf). Polynomial RBF with PSO for (centers, sigmas).")
    print("(rbf). RBF with PSO for (centers, sigmas, weights).")
    print("(ff). Feed-Forward with PSO.")
    print("(srbf). RBF with k-means.")
    print("(sprbf). Polynomial RBF with k-means.")
    nn_type = (input("Please choose an option: "))

if nn_type == 'prbf':
    # print("Starting Polynomial RBF...")
    pso.PSO(x,y,iterations,n_clusters,quiet=quiet,explicit=expl)
elif nn_type == 'rbf':
    # print("Starting RBF swarm...")
    pso.PSO(x,y,iterations,n_clusters,'rbf',quiet=quiet,explicit=expl)
elif nn_type == 'ff':
    # print("Starting Feed-Forward swarm...")
    pso.PSO(x,y,iterations,n_clusters,'ff',quiet=quiet,explicit=expl)
elif nn_type == 'srbf':
    for c in range(2,n_clusters):
        doTheNet(c,x,y)
elif nn_type == 'sprbf':
    for c in range(2,n_clusters):
        doThePolyNet(c,x,y)
