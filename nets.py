from utilities import *
import json

def doTheNet(n_clusters,x,y, lamda=1):
    x_test, x_train, y_test, y_train = separateToTestTrain(0.4,x,y)
    y_kmeans, centers = kMeans(x_train, n_clusters)

    sigma_array = (calculateSigmaArray(centers))
    # print(sigma_array)

    G = gaussianMatrix(x_train,centers,sigma_array)
    # print(G.shape)
    W = calculateWeights(lamda,n_clusters,G,y_train,centers, sigma_array)

    print(W.shape)
    G = gaussianMatrix(x_test,centers,sigma_array)
    Y = G.dot(W)

    error = np.subtract(y_test,Y)
    print("rootMeanError for c=%d: %f" %(n_clusters,rootMeanError(error)))
    return rootMeanError(error)

def particleNet(x, y, centers, sigma_array, W, lamda=1):
    # x_test, x_train, y_test, y_train = separateToTestTrain(0.4,x,y)

    G = gaussianMatrix(x,centers,sigma_array)
    W = W.reshape(len(W),1)
    Y = G.dot(W)
    error = np.subtract(y,Y)

    return rootMeanError(error)

def doThePolyNet(n_clusters,x,y, lamda=0.01):
    x_test, x_train, y_test, y_train = separateToTestTrain(0.4,x,y)

    y_kmeans, centers = kMeans(x_train, n_clusters)

    sigma_array = (calculateSigmaArray(centers))

    L = calculateLAMDA(x_train,centers,sigma_array)
    L = np.array(L)
    # print(L.shape)

    W = calculateWeightsPolynomial(lamda,n_clusters,L,y_train,centers, sigma_array, len(x[0]))

    test_L = calculateLAMDA(x_test,centers,sigma_array)
    Y = test_L.dot(W)

    error = np.subtract(y_test,Y)

    print("rootMeanError for c=%d: %f" %(n_clusters,rootMeanError(error)))
    return rootMeanError(error)

def particlePolyRBF(x, y, centers, sigmas, W=None, lamda=1000):
    # x_test, x_train, y_test, y_train = separateToTestTrain(0.4,x,y)
    # print(sigmas.shape)
    # print(sigmas)
    try:
        # Needed for reasons...

        W = W.reshape(len(W),1)
        print("HI")
    except:
        L = calculateLAMDA(x,centers,sigmas)
        try:
            W = calculateWeightsPolynomial(L, y, centers, sigmas, len(x[0]), lamda)
        except np.linalg.linalg.LinAlgError:
            raise np.linalg.linalg.LinAlgError

    # print(W.shape)
    # print(W[2])
    Y=[]
    for n in range(len(x)):
        index = 0
        souma = 0
        for c in range(len(centers)):
            g = gaussianFunction(x[n],centers[c],sigmas[c])
            # print(g,index,len(x[0]))
            # print(index)
            factor = W[index]
            temp_s = 0
            for j in range(len(x[0])):
                # print(len(x[0]))
                index += 1
                temp_s += W[index]*x[n][j]
            index += 1
            souma += g * (factor + temp_s)
        Y.extend(souma)

    # print(y.shape)

    Y = np.asarray(Y)
    # print(Y.shape)
    Y = Y.reshape(len(Y),1)

    # Y = Y.transpose()
    error = np.subtract(y,Y)
    # print(error.shape)
    return rootMeanError(error)


def feedForward(x,y, n_clusters,a,b):
    # x_test, x_train, y_test, y_train = separateToTestTrain(0.4,x,y)
    errors = []
    for i in range(len(x)):
        sums = calcWeightedSum(x[i], n_clusters,a)
        total_sum = 0
        for c in range(len(sums)):
            temp = np.tanh(sums[c]/2)
            total_sum += temp * b[c]
        errors.append(y[i] - total_sum)

    return rootMeanError(errors)
    # print("rootMeanError for c=%d: %f" %(n_clusters,rootMeanError(errors)))

def save(centers,sigmas,W):
    c_write = []
    for i in centers:
        temp= []
        for j in i:
            temp.append(j)
        c_write.append(temp)

    s_write = []
    for i in sigmas:
        s_write.append(i)

    w_write = []
    print(W)
    for i in W:
        w_write.append(i)

    with open('net.var','w') as f:
        json.dump({'centers': c_write, 'sigmas': s_write, 'weights':w_write},f)


if __name__ == "__main__":
    import random
    x, y = readCSV('data.csv')
    x = np.asarray(x)
    y = np.asarray(y)

    n_clusters = 5
    a = []

    for j in range(n_clusters):
        a.append([random.uniform(-100, 100) for _ in range(len(x[0]))])
    b = [random.uniform(-100, 100) for _ in range(n_clusters)]

    feedForward(x,y,n_clusters,a,b)
