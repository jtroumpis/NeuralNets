from utilities import readCSV
from nets import *
import numpy as np
import pso,sys, json
from datetime import datetime
import argparse

def selectNN(nn_type, data,run,iterations,n_clusters,quiet,expl):
    errors_test = []
    errors_train = []
    if nn_type == 'prbf':
        # print("Starting Polynomial RBF...")
        for i in range(run):
            errors.append(pso.PSO(data,iterations,n_clusters,quiet=quiet,explicit=expl))

        mean , std = getMeanSTD(errors)
        res = {'nn_type': nn_type, 'c': n_clusters, 'mean': mean, 'std': std}

        with open('res.txt','a+') as f:
            json.dump(res, f)
    elif nn_type == 'rbf':
        # print("Starting RBF swarm...")
        for i in range(run):
            test_error, train_error = pso.evolution(data,iterations,n_clusters,'rbf',quiet=quiet,explicit=expl)
            errors_test.append(test_error)
            errors_train.append(train_error)
        mean , std = getMeanSTD(errors_test)
        res = {'nn_type': nn_type, 'c': n_clusters, 'mean': mean, 'std': std}

        with open('res.txt','a+') as f:
             json.dump(res, f)

    elif nn_type == 'ff':
        for i in range(run):
            test_error, train_error = pso.PSO(data,iterations,n_clusters,'ff',quiet=quiet,explicit=expl)
            errors_test.append(test_error)
            errors_train.append(train_error)

        mean , std = getMeanSTD(errors_test)
        res = {'nn_type': nn_type, 'c': n_clusters, 'mean': mean, 'std': std, 'testing': True}

        mean , std = getMeanSTD(errors_train)
        res_train = {'nn_type': nn_type, 'c': n_clusters, 'mean': mean, 'std': std, 'testing': False}

        with open('res.txt','a+') as f:
            json.dump(res, f)
            f.write('\n')
            json.dump(res_train, f)

    # elif nn_type == 'srbf':
    #     for c in range(2,n_clusters):
    #         for i in range(run):
    #             errors.append(doTheNet(c,x,y))
    #             mean , std = getMeanSTD(errors)
    #         res = {'nn_type': nn_type, 'c': c, 'mean': mean, 'std': std}
    #
    #         with open('res.txt','a+') as f:
    #             json.dump(res, f)
    # elif nn_type == 'sprbf':
    #     for c in range(2,n_clusters):
    #         for i in range(run):
    #             errors.append(doThePolyNet(c,x,y))
    #             mean , std = getMeanSTD(errors)
    #         res = {'nn_type': nn_type, 'c': c, 'mean': mean, 'std': std}
    #
    #         with open('res.txt','a+') as f:
    #             json.dump(res, f)
    with open('res.txt','a+') as f:
        f.write('\n')

parser = argparse.ArgumentParser(description='Welcome my friend.')

parser.add_argument('-s --SELECT', action="store_true", dest='select',
                    help='CHOOSE THE NN')
parser.add_argument('-n', action="store", dest='nn', default='',
                    help='which NN should be used: options (prbf, rbf, ff)')
parser.add_argument('-c', action="store", dest='n_clusters', type=int, default=-1,
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
parser.add_argument('-g --AR_AGGR', action="store_false", dest='aa',
                    help='Does Arithmetic Aggregation')
parser.add_argument('-f --FILE', action="store", dest='filename', default='data.csv',
                    help='Chooses the input file.')
parser.add_argument('--TEST', action='store', dest='test_file', default=None,
                    help='Chooses the test input file')
parser.add_argument('--TRAIN', action='store', dest='train_file', default=None,
                    help='Chooses the train input file')
args = parser.parse_args()

if not args.train_file:
    x, y = readCSV(args.filename,args.aa)
    x = np.asarray(x)
    y = np.asarray(y)
    x_test, x_train, y_test, y_train = separateToTestTrain(0.6,x,y)

else:
    x_train, y_train = readFromFile(args.train_file)
    x_test, y_test = readFromFile(args.test_file)
    x_train = np.asarray(x_train)
    # print(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    # print(x_train.shape,y_train.shape)

# printTestTrainToFile(x_test, x_train, y_test, y_train)
# exit()
# with open('new_out.csv','w') as f:
#     for i in range(len(x_train)):
#         s = ""
#         for j in x_train[i]:
#             s += str(j) + ", "
#         s += str(y_train[i][0]) +'\n'
#         # print(y_train[i])
#         f.write(s)



iterations = args.iterations
n_clusters = args.n_clusters
nn_type = args.nn
quiet = args.quiet
expl = args.explicit
# filename = args.filename
aa = args.aa

if args.select:
    print("(prbf). Polynomial RBF with PSO for (centers, sigmas).")
    print("(rbf). RBF with PSO for (centers, sigmas, weights).")
    print("(ff). Feed-Forward with PSO.")
    print("(srbf). RBF with k-means.")
    print("(sprbf). Polynomial RBF with k-means.")
    nn_type = (input("Please choose an option: "))


with open('res.txt','a+') as f:
    f.write(str(datetime.now())+'\n')

if args.all:
    nn_list = ['prbf','rbf','ff']
    c_list = [3,4,5,6,7,8,9,10,11]

if args.nn!='':
    nn_list = [args.nn]
else:
    nn_list = ['prbf']

if args.n_clusters>0:
    c_list = [args.n_clusters]
else:
    c_list = [3,4,5,6,7,8,9,10,11]

for nn in nn_list:
    for c in c_list:
        selectNN(nn,(x_train,y_train, x_test, y_test),args.run,args.iterations,c,args.quiet,args.explicit)

# selectNN(args.nn,args.run,x,y,args.iterations,args.n_clusters,args.quiet,args.explicit)
