from utilities import readCSV
from nets import *
import numpy as np
import pso,sys, json
from datetime import datetime
import argparse

def selectNN(nn_type,run,x,y,iterations,n_clusters,quiet,expl):
    errors = []
    if nn_type == 'prbf':
        # print("Starting Polynomial RBF...")
        for i in range(run):
            errors.append(pso.PSO(x,y,iterations,n_clusters,quiet=quiet,explicit=expl))

        mean , std = getMeanSTD(errors)
        res = {'nn_type': nn_type, 'c': n_clusters, 'mean': mean, 'std': std}

        with open('res.txt','a+') as f:
            json.dump(res, f)
    elif nn_type == 'rbf':
        # print("Starting RBF swarm...")
        for i in range(run):
            errors.append(pso.PSO(x,y,iterations,n_clusters,'rbf',quiet=quiet,explicit=expl))
        mean , std = getMeanSTD(errors)
        res = {'nn_type': nn_type, 'c': n_clusters, 'mean': mean, 'std': std}

        with open('res.txt','a+') as f:
             json.dump(res, f)

    elif nn_type == 'ff':
        for i in range(run):
            errors.append(pso.PSO(x,y,iterations,n_clusters,'ff',quiet=quiet,explicit=expl))
        mean , std = getMeanSTD(errors)
        res = {'nn_type': nn_type, 'c': n_clusters, 'mean': mean, 'std': std}

        with open('res.txt','a+') as f:
            json.dump(res, f)

    elif nn_type == 'srbf':
        for c in range(2,n_clusters):
            for i in range(run):
                errors.append(doTheNet(c,x,y))
                mean , std = getMeanSTD(errors)
            res = {'nn_type': nn_type, 'c': c, 'mean': mean, 'std': std}

            with open('res.txt','a+') as f:
                json.dump(res, f)
    elif nn_type == 'sprbf':
        for c in range(2,n_clusters):
            for i in range(run):
                errors.append(doThePolyNet(c,x,y))
                mean , std = getMeanSTD(errors)
            res = {'nn_type': nn_type, 'c': c, 'mean': mean, 'std': std}

            with open('res.txt','a+') as f:
                json.dump(res, f)
    with open('res.txt','a+') as f:
        f.write('/n')

parser = argparse.ArgumentParser(description='Welcome my friend.')

parser.add_argument('-s --SELECT', action="store_true", dest='select',
                    help='CHOOSE THE NN')
parser.add_argument('-n', action="store", dest='nn', default='prbf',
                    help='which NN should be used: options (prbf, rbf, ff)')
parser.add_argument('-c', action="store", dest='n_clusters', type=int, default=10,
                    help='The number of clusters that should be used')
parser.add_argument('-i', action="store", dest='iterations', type=int, default=500,
                    help='The number of iterations')
parser.add_argument('-q --QUIET', action="store_true", dest='quiet',
                    help='Does not print anything but gBests')
parser.add_argument('-e --EXPLICIT', action="store_true", dest='explicit',
                    help='Prints fitness for each particle every iteration')
parser.add_argument('-r --RUN', action="store", dest='run', type=int, default=10,
                    help='Number of repeats')
parser.add_argument('-a --ALL', action="store_true", dest='all',
                    help='Runs all the networks')
args = parser.parse_args()

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


with open('res.txt','a+') as f:
    f.write(str(datetime.now()+'\n'))

if args.all:
    for nn in ['prbf','rbf','ff']:
        for c in [2,4,6,8,10,12,14]:
            selectNN(nn,args.run,x,y,args.iterations,c,args.quiet,args.explicit)

selectNN(args.nn,args.run,x,y,args.iterations,args.n_clusters,args.quiet,args.explicit)
