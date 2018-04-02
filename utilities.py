import csv, math
from sklearn.cluster import KMeans
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
