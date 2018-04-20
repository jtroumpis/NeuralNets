import csv, math
from sklearn.cluster import KMeans
from utilities import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from numpy.linalg import inv

def separateToTestTrain(factor, x, y):
    v = int(len(x)*factor)
    x_test = x[:v]
    x_train = x[v:]
    y_test = y[:v]
    y_train = y[v:]

    return  x_test, x_train, y_test, y_train

def isInput(s):
    return 'input' in s

def addToNetDict(dictionary,key,value):
    try:
        dictionary[key]['l'].append(value)
        dictionary[key]['sum'] += value
    except KeyError:
        dictionary[key] = {'l': [value],'sum': value}

    return value

def calculateArithmeticProgression(net_dic,total):
    for key, value in net_dic.items():
        weight = value['sum'] / total
        net_dic[key][weight] = weight
        # print(weight)
    net_list = []
    while True:
        try:
            net_total = 0
            for key in net_dic:
                # print(net_dic[key]['l'].pop())
                net_total += net_dic[key]['l'].pop() * weight
                # total += weight * value
            net_list.append(net_total)
        except IndexError:
            break
    return net_list
    # return net_dic

def readCSV(filename = 'data.csv', keep_this=None):
    x = []
    y = []
    net_in = {}
    net_out = {}
    total_sum_in = 0
    total_sum_out = 0
    with open(filename) as f:
        reader = csv.DictReader(f)

        for row in reader:
            x_sub_list = []
            y_sub_list = []
            for key, value in row.items():
                try:
                    value = int(value)
                except ValueError:
                    pass
                except TypeError:
                    break

                if 'net_in' in key:
                    total_sum_in += addToNetDict(net_in,key,value)
                elif 'net_out' in key:
                    total_sum_out += addToNetDict(net_out,key,value)
                # elif key=='time' or 'tempFPGA' in key or 'ytempCPU'  in key:
                elif key=='time'  in key:
                    continue
                else:
                    if isInput(key):
                        # print(key)
                        x_sub_list.append(value)
                    else:
                        # print(key)
                        y_sub_list.append(value)

            x.append(x_sub_list)
            y.append(y_sub_list)
        # print(x)
        net_in = calculateArithmeticProgression(net_in,total_sum_in)
        net_out = calculateArithmeticProgression(net_out,total_sum_out)

        # print(net_in)

        for i in range(len(x)):
            x[i].append(net_in[i])
            x[i].append(net_out[i])
        return x, y

# Calculates the distance
def calcDistance(x,center):
    # print(len(x))
    dist = 0
    for j in range(0,len(x)):
        dist += math.pow(x[j] - center[j],2)
    return math.sqrt(dist)

# Calculate the maximum distance of each cluster (dmax)
def multidistance(array,n_clusters,cluster_array,center_array):
    max_dist = {}
    for i in range(0,n_clusters):
        max_dist[i] = 0

    for i in range(0,len(array)):
        dist = calcDistance(array[i],center_array[cluster_array[i]])
        max_dist[cluster_array[i]] =  max(dist,max_dist[cluster_array[i]])

    return (max_dist)

# Calculates the gaussian function for a vector
def gaussianFunction(x, center, sigma):
    dist = calcDistance(x, center)
    try:
        fraction = math.pow(dist / sigma,2)
    except ZeroDivisionError:
        fraction = 0
    res = math.exp(-fraction)
    return res

def kMeans(trainX,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(trainX)
    y_kmeans = kmeans.predict(trainX)
    centers = kmeans.cluster_centers_

    return y_kmeans, centers

# Makes the gaussian matrix
def gaussianMatrix(x,centers,sigma_array):
    gaussian_array = []
    for n in range(len(x)):
        small_g = []
        for i in range(len(centers)):
            g = gaussianFunction(x[n], centers[i], sigma_array[i])
            small_g.append(g)
        gaussian_array.append(small_g)
    return np.asarray(gaussian_array)

# Calculate root mean error
def rootMeanError(error_array):
    souma = 0
    for i in error_array:
        souma += i**2

    return math.sqrt(souma/len(error_array))

def calculateWeightsPolynomial(lamda,n_clusters,G,y,centers,sigma_array,p):
    gamma = lamda * np.identity(n_clusters*(p+1))

    gtg = G.transpose().dot(G)
    gammatgamma = gamma.transpose().dot(gamma)

    inversed = inv(np.add(gtg, gammatgamma))

    temp = inversed.dot(G.transpose())
    W = temp.dot(y)

    # print("W=",W)

    return W

def calculateWeights(lamda,n_clusters,G,y,centers,sigma_array):
    gamma = lamda * np.identity(n_clusters)

    gtg = G.transpose().dot(G)
    gammatgamma = gamma.transpose().dot(gamma)

    inversed = inv(np.add(gtg, gammatgamma))

    temp = inversed.dot(G.transpose())
    W = temp.dot(y)

    # print("W=",W)

    return W

def calculateSigmaArray(centers):

    dist_array = np.array([])
    for c in centers:
        min_dist = math.inf
        for i in centers:
            # print(i,c)
            if np.array_equal(i,c): continue
            dist = calcDistance(c,i)
            # print(dist)
            if dist < min_dist:
                min_dist = dist
        dist_array = np.append(dist_array,[min_dist])
    return dist_array

def calculateSigma_withDmax(dmaxes):
    sigma_array = []
    # print(sigma_array)
    for i in range(len(dmaxes)):
        try:
            sigma_array.append(2/3*dmaxes[i])
        except ZeroDivisionError:
            pass

    return (sigma_array)

def calculateSmallL(x, center, sigma):
    g = gaussianFunction(x, center, sigma)
    lamda = [g]
    for p in range(len(x)):
        lamda.append(g * x[p])
    # print(len(lamda))
    return lamda

def calculateLAMDA(x,centers,sigmas):
    bigL = []
    for n in range(len(x)):
        L=[]
        for c in range(len(centers)):
            L.extend(calculateSmallL(x[n],centers[c],sigmas[c]))
        bigL.append(L)
    # print(len(L))
    return np.array(bigL)

# def polynomialRBF(lamda, n_clusters,x,y):
#     y_kmeans, centers = kMeans(x, n_clusters)
#     sigma_array = (calculateSigmaArray(centers))
#
#     L = calculateLAMDA(x,centers,sigma_array, y_kmeans)
#     L = np.array(L)
#     print(L.shape)
#
#     W = calculateWeights(lamda,n_clusters,L,y,centers, sigma_array, len(x[0]))
#     Y = L.dot(W)
