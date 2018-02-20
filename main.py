import csv
def readCSV(filename = 'data.csv', keep_this=None):
    d = {}
    l = []
    i = 0
    with open(filename) as f:
        # reader = csv.reader(f)
        reader = csv.DictReader(f)
        # headers = reader.next()
        # print(headers)
        # for row in reader:
        #     print (row)
        if i > 100:
            return l
        for row in reader:
            # print (row['time'])
            # d[row['time']] = {}
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

def multidistance(array,n_clusters,cluster_array,center_array):
    max_dist = {}
    for i in range(0,n_clusters):
        max_dist[i] = 0

    # print(center_array)

    for i in range(0,len(array)):
        center = center_array[cluster_array[i]]
        # print(center)
        dist = 0
        for j in range(0,len(array[i])):
            dist += array[i][j] - center[j]
            # print(array[i][j],center[j])
        max_dist[cluster_array[i]] =  max(dist,max_dist[cluster_array[i]])

        # print(array[i],cluster_array[i])

    print(max_dist)

d = (readCSV(keep_this=['CPU','MemoryUsed','tempCPU']))
# print(d)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

item = d.pop(0)
X = np.array([[int(item['CPU']),int(item['MemoryUsed'])]])

for item in d:
    # print(X,[item['CPU'],item['MemoryUsed']])
    if not item['CPU']:
        break

    X = np.append(X,[[int(item['CPU']),int(item['MemoryUsed'])]], axis=0)

n_clusters = 2
# print(X)
plt.scatter(X[:,0], X[:, 1])
# plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

multidistance(X,n_clusters,y_kmeans,centers)

plt.scatter(X[:,0], X[:, 1], c=y_kmeans, cmap='viridis')


plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.show()
