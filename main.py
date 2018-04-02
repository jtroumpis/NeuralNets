import math
from utilities import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from numpy.linalg import inv

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

def calculateLAMDA(x,centers,sigmas,y_kmeans):
    bigL = []
    for n in range(len(x)):
        L=[]
        for c in range(len(centers)):
            L.extend(calculateSmallL(x[n],centers[c],sigmas[c]))
        bigL.append(L)
    # print(len(L))
    return bigL

def polynomialRBF(lamda, n_clusters,x,y):
    y_kmeans, centers = kMeans(x, n_clusters)
    sigma_array = (calculateSigmaArray(centers))

    L = calculateLAMDA(x,centers,sigma_array, y_kmeans)
    L = np.array(L)
    print(L.shape)

    W = calculateWeights(lamda,n_clusters,L,y,centers, sigma_array, len(x[0]))
    Y = L.dot(W)

def doTheNet(lamda,n_clusters,x,y):
    y_kmeans, centers = kMeans(x, n_clusters)

    sigma_array = (calculateSigmaArray(centers))
    # print(sigma_array)

    G = gaussianMatrix(x,centers,sigma_array)
    print(G.shape)
    W = calculateWeights(lamda,n_clusters,G,y,centers, sigma_array)

    Y = G.dot(W)

    error = np.subtract(y,Y)
    print("rootMeanError for c=%d: %f" %(n_clusters,rootMeanError(error)))
    return rootMeanError(error)

def doThePolyNet(lamda, n_clusters,x,y):
    y_kmeans, centers = kMeans(x, n_clusters)

    sigma_array = (calculateSigmaArray(centers))

    L = calculateLAMDA(x,centers,sigma_array, y_kmeans)
    L = np.array(L)
    print(L.shape)

    W = calculateWeightsPolynomial(lamda,n_clusters,L,y,centers, sigma_array, len(x[0]))
    Y = L.dot(W)
    error = np.subtract(y,Y)

    print("rootMeanError for c=%d: %f" %(n_clusters,rootMeanError(error)))
    return rootMeanError(error)

lamda = 1

x, y = readCSV('data.csv')
x = np.asarray(x)
y = np.asarray(y)

# print(x.shape)
# for lamda in [0.1,1,10,100]:
#     print("LAMDA=",lamda)
for c in range(2,30):
    doThePolyNet(lamda,c,x,y)
