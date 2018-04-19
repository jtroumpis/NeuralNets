from Particle import Particle, Full_Particle
from utilities import *
import random
lamda = 1

def PSO_RBF(x,y,polynomial=True):
    # x, y = readCSV('data.csv')
    # x = np.asarray(x)
    # y = np.asarray(y)
    inertia = random.uniform(0.5,1)
    n_of_particles = 20
    n_clusters = 5
    p_list = []
    gbest = None

    #Initialise particles
    for i in range(n_of_particles):
        # p_list.append(Full_Particle(x,y,n_clusters))
        if polynomial:
            p_list.append(Particle(x,y,n_clusters,inertia))
        else:
            p_list.append(Full_Particle(x,y,n_clusters,inertia))
        print("Creating particle", i)
        try:
            if (p_list[i].getPBest()[0]) < gbest[0]:
                gbest = p_list[i].getPBest()
                # print("New gbest = ", gbest)
        except TypeError:
            gbest = p_list[i].getPBest()
            # print("New gbest = ", gbest)
    print("Staring gbest = ", gbest)
    print("Starting the swarming")
    for i in range(100):
        print("Iteration",i)
        c=0
        for p in p_list:
            pbest = p.update(gbest)
            c+=1
            print("Particle[%d]: %f" % (c,pbest[0]))
            if  pbest[0] < gbest[0]:
                gbest = pbest
                print("New gbest = ", gbest)

    print("Finished!")
    print("gbest = ", gbest)
