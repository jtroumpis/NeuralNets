

def PSO(data,iterations=500,n_clusters=10,nn='rbf', n_of_particles=20,quiet=False,explicit=False):
    x_train,y_train, x_test, y_test = data
