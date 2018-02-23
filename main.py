import csv, math

def readCSV(filename = 'data.csv', keep_this=None):
    d = {}
    train = []
    test = []
    l=[]
    i = 0
    with open(filename) as f:
        # reader = csv.reader(f)
        reader = csv.DictReader(f)

        for row in reader:
            # print (row['time'])
            # d[row['time']] = {}
            if i > 700:
                return train, l
            if i == 500:
                train = list(l)
                l=[]
            i+=1
            l.append(row)
            for key, value in row.items():
                try:
                    value = int(value)
                except ValueError:
                    pass
                except TypeError:
                    break

                if keep_this:
                    if key in keep_this:
                        try:
                            d[key].append(value)
                        except KeyError:
                            d[key] = [value]
                else:
                    try:
                        d[key].append(value)
                    except KeyError:
                        d[key] = [value]
        return l

def calcDistance(k,array,center):
    dist = 0
    for j in range(0,len(array[k])):
        dist += math.pow(array[k][j] - center[j],2)
    return math.sqrt(dist)

def multidistance(array,n_clusters,cluster_array,center_array):
    max_dist = {}
    for i in range(0,n_clusters):
        max_dist[i] = 0

    # print(center_array)

    for i in range(0,len(array)):
        dist = calcDistance(i,array,center_array[cluster_array[i]])
        max_dist[cluster_array[i]] =  max(dist,max_dist[cluster_array[i]])

    return (max_dist)

def listToArrays(input_list):
    item = input_list.pop(0)
    X = np.array([[int(item['CPU']),int(item['MemoryUsed'])]])
    Y = np.array([[int(item['tempCPU'])]])
    for item in input_list:
        # print(X,[item['CPU'],item['MemoryUsed']])
        if not item['CPU']:
            break

        X = np.append(X,[[int(item['CPU']),int(item['MemoryUsed'])]], axis=0)
        Y = np.append(Y,[[int(item['tempCPU'])]], axis=0)
    return X,Y



def gaussianFunction(array, center, sigma):
    gaussian_row = np.array([])
    for k in range(len(array)):
        dist = calcDistance(k, array, center)

        fraction = math.pow(dist / sigma,2)

        res = math.exp(-fraction)
        gaussian_row = np.append(gaussian_row,[res])

    return gaussian_row

def gaussianMatrix(array,center_array,sigma_array):
    # gaussian_array = np.array([[]])
    # print(center_array)
    for i in range(len(center_array)):
        g = gaussianFunction(array, center_array[i], sigma_array[i])
        # print('g',g.shape)
        try:
            gaussian_array = np.vstack([gaussian_array, g])
        except UnboundLocalError:
            gaussian_array = np.array(g)


    # print('array',gaussian_array.transpose().shape)
    return gaussian_array.transpose()

def rootMeanError(error_array):
    souma = 0
    for i in error_array:
        souma += i**2

    return math.sqrt(souma/len(error_array))

train, test = (readCSV(keep_this=['CPU','MemoryUsed','tempCPU']))
# print(d)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)

trainX, trainY = listToArrays(train)
testX, testY = listToArrays(test)

n_clusters = 3
# print(X)

# plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(trainX)
y_kmeans = kmeans.predict(trainX)
centers = kmeans.cluster_centers_



dmaxes = multidistance(trainX,n_clusters,y_kmeans,centers)

sigma_array = dmaxes
for i in sigma_array:
    i = 2/3*(i)

G = (gaussianMatrix(trainX,centers,sigma_array))

print("G dimensions: ",G.shape)

lamda = 1

gamma = lamda * np.identity(n_clusters)

to_inverse = G.transpose().dot(G)
temp = gamma.dot(gamma.transpose())

to_inverse = np.add(to_inverse, temp)

print('G=', G)

from numpy.linalg import inv

inversed = inv(to_inverse)

print("Inversed = ",inversed)
print("Inversed*to_inverse = ",inversed.dot(to_inverse))

temp = inversed.dot(G.transpose())
W = temp.dot(trainY)

print("W=",W)

Y = G.dot(W)

print("Y=",Y)

error = np.subtract(trainY,Y)

print("error=", error)

print("rootMeanError=", rootMeanError(error))
