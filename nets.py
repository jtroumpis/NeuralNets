from utilities import *

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

def doThePolyNet(n_clusters,x,y, lamda=1):
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

def particlePolyRBF(x, y, centers, sigmas, W=None, lamda=1):
    # x_test, x_train, y_test, y_train = separateToTestTrain(0.4,x,y)
    # print(sigmas.shape)
    try:
        # Needed for reasons...
        W = W.reshape(len(W),1)
    except:
        L = calculateLAMDA(x,centers,sigmas)
        W = calculateWeightsPolynomial(lamda,len(centers),L,y,centers, sigmas, len(x[0]))

    # print(W.shape)
    # test_L = calculateLAMDA(x_test,centers,sigmas)
    Y = L.dot(W)
    # print(Y.shape)
    error = np.subtract(y,Y)

    # print("rootMeanError for c=%d: %f" %(len(centers),rootMeanError(error)))
    return rootMeanError(error)

def calcWeightedSum(x, n_clusters,a):
    sums = []
    for c in range(n_clusters):
        node_sum = 0
        for p in range(len(x)):
            node_sum += x[p]*a[c][p]
        sums.append(node_sum)
    return sums


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
