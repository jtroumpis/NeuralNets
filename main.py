from utilities import readCSV
from nets import *
import numpy as np
import pso,sys, json
from datetime import datetime
import argparse
from mail import sendMail

def create_JSON_output(name,nn_type,n_clusters,train,test):
    mean , std = getMeanSTD(test)
    res = {'name': name, 'nn_type': nn_type, 'c': n_clusters, 'mean': mean, 'std': std, 'run_type': 'Testing'}

    mean , std = getMeanSTD(train)
    res_train = {'name': name, 'nn_type': nn_type, 'c': n_clusters, 'mean': mean, 'std': std, 'run_type': 'Training'}

    with open('res.txt','a+') as f:
        json.dump(res, f)
        f.write('\n')
        json.dump(res_train, f)
        f.write('\n')

def selectNN(name,nn_type, data,run,iterations,n_clusters,quiet,expl):
    errors_test = []
    errors_train = []
    if nn_type == 'prbf':
        # print("Starting Polynomial RBF...")
        for i in range(run):
            test_error, train_error = (pso.PSO(name,data,iterations,n_clusters,quiet=quiet,explicit=expl))
            errors_test.append(test_error)
            errors_train.append(train_error)
        create_JSON_output(name,nn_type,n_clusters,errors_train,errors_test)

    elif nn_type == 'rbf':
        # print("Starting RBF swarm...")
        for i in range(run):
            test_error, train_error = pso.evolution(name,data,iterations,n_clusters,'rbf',quiet=quiet,explicit=expl)
            errors_test.append(test_error)
            errors_train.append(train_error)

        create_JSON_output(name,nn_type,n_clusters,errors_train,errors_test)

    elif nn_type == 'ff':
        for i in range(run):
            test_error, train_error = pso.PSO(name,data,iterations,n_clusters,'ff',quiet=quiet,explicit=expl)
            errors_test.append(test_error)
            errors_train.append(train_error)

        create_JSON_output(name,nn_type,n_clusters,errors_train,errors_test)

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
parser.add_argument('--FILE_LIST', action='store', dest='file_list', default=None,
                    help='Reads a number of different input files.')
args = parser.parse_args()

file_list = []
if args.file_list:
    with open(args.file_list,'r') as f:
        for file in f:
            file_tile = file.split('/')[-1].strip()
            file_list.append({'name':file_tile, 'train': file.strip()+'_train.csv', 'test':file.strip()+'_test.csv'})

    print(file_list)
else:
    if not args.train_file:
        x, y = readCSV(args.filename,args.aa)
        x = np.asarray(x)
        y = np.asarray(y)
        x_test, x_train, y_test, y_train = separateToTestTrain(0.6,x,y)

    else:
        file_tile = args.train_file.split('.')[-2]
        file_list.append({'name':file_tile, 'train': args.train_file, 'test':args.test_file})
for filename in file_list:
    x_train, y_train = readFromFile(filename['train'])
    x_test, y_test = readFromFile(filename['test'])
    # x_train = np.asarray(x_train)
    # print(x_train)
    filename['x_train'] = np.asarray(x_train)
    filename['y_train'] = np.asarray(y_train)
    filename['x_test'] = np.asarray(x_test)
    filename['y_test'] = np.asarray(y_test)

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

if args.nn!='':
    nn_list = [args.nn]
else:
    nn_list = ['rbf']

if args.all:
    nn_list = ['rbf','ff']
    c_list = [3,4,5,6,7,8,9,10,11]


if args.n_clusters>0:
    c_list = [args.n_clusters]
else:
    c_list = [2,4,6,8,10,12]


for data_file in file_list:
    for nn in nn_list:
        for c in c_list:
            selectNN(filename['name'],nn,(filename['x_train'],filename['y_train'], filename['x_test'], filename['y_test']),args.run,args.iterations,c,args.quiet,args.explicit)
        subj = 'Finished %s of %s' % (nn, data_file['test'])
        attach = ['/home/jtroumpis/Programming/neuralnet/res.txt',
        '/home/jtroumpis/Programming/neuralnet/complete_res.json']
        with open('res.txt', 'r') as f:
            message = f.read()
        sendMail(subj,attach,message)
# selectNN(args.nn,args.run,x,y,args.iterations,args.n_clusters,args.quiet,args.explicit)
