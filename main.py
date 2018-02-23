import csv, math

def isInput(s):
    return s[0]=='x'

def readCSV(filename = 'data.csv', keep_this=None):
    x = []
    y = []

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

                if key=='time' or 'tempFPGA'in key or 'ytempCPU' in key:
                    continue

                if isInput(key):
                    x_sub_list.append(value)
                else:
                    y_sub_list.append(value)

            x.append(x_sub_list)
            y.append(y_sub_list)
        return x, y

# Calculates the distance
def calcDistance(k,array,center):
    dist = 0
    for j in range(0,len(array[k])):
        dist += math.pow(array[k][j] - center[j],2)
    return math.sqrt(dist)

# Calculate the maximum distance of each cluster (dmax)
def multidistance(array,n_clusters,cluster_array,center_array):
    max_dist = {}
    for i in range(0,n_clusters):
        max_dist[i] = 0

    for i in range(0,len(array)):
        dist = calcDistance(i,array,center_array[cluster_array[i]])
        max_dist[cluster_array[i]] =  max(dist,max_dist[cluster_array[i]])

    return (max_dist)

# Calculates the gaussian function for a vector
def gaussianFunction(array, center, sigma):
    gaussian_row = np.array([])
    for k in range(len(array)):
        dist = calcDistance(k, array, center)
        try:
            fraction = math.pow(dist / sigma,2)
        except ZeroDivisionError:
            fraction = 0
        res = math.exp(-fraction)
        gaussian_row = np.append(gaussian_row,[res])

    return gaussian_row

# Makes the gaussian matrix
def gaussianMatrix(array,center_array,sigma_array):
    for i in range(len(center_array)):
        g = gaussianFunction(array, center_array[i], sigma_array[i])
        try:
            gaussian_array = np.vstack([gaussian_array, g])
        except UnboundLocalError:
            gaussian_array = np.array(g)
    return gaussian_array.transpose()

# Calculate root mean error
def rootMeanError(error_array):
    souma = 0
    for i in error_array:
        souma += i**2

    return math.sqrt(souma/len(error_array))

def kMeans(trainX,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(trainX)
    y_kmeans = kmeans.predict(trainX)
    centers = kmeans.cluster_centers_

    return y_kmeans, centers

def calculateWeights(lamda,n_clusters,G,y,centers,sigma_array):
    gamma = lamda * np.identity(n_clusters)

    gtg = G.transpose().dot(G)
    gammatgamma = gamma.transpose().dot(gamma)

    inversed = inv(np.add(gtg, gammatgamma))

    temp = inversed.dot(G.transpose())
    W = temp.dot(y)

    # print("W=",W)

    return W

def doTheNet(lamda,n_clusters,x,y):
    y_kmeans, centers = kMeans(x, n_clusters)

    dmaxes = multidistance(x,n_clusters,y_kmeans,centers)

    # print(dmaxes)
    sigma_array = []
    # print(sigma_array)
    for i in range(len(dmaxes)):
        try:
            sigma_array.append(2/3*dmaxes[i])
        except ZeroDivisionError:
            pass

    G = gaussianMatrix(x,centers,sigma_array)
    print(G.shape)
    W = calculateWeights(lamda,n_clusters,G,y,centers, sigma_array)
    Y = G.dot(W)

    # print("Y=",Y)

    error = np.subtract(y,Y)

    # print("error=", error)

    print("rootMeanError for c=%d: %f" %(n_clusters,rootMeanError(error)))

    return rootMeanError(error)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import inv

n_clusters = 3

lamda = 1

x, y = readCSV()

x = np.asarray(x)
y = np.asarray(y)
for c in range(2,10):
    doTheNet(lamda,c,x,y)
