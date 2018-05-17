from Particle import Particle, Full_Particle, FFParticle
from utilities import *
from nets import *
import random
lamda = 1

def PSO(x,y,iterations=1000,n_clusters=10,nn='prbf', n_of_particles=20,quiet=False,explicit=False):
    x_test, x_train, y_test, y_train = separateToTestTrain(0.6,x,y)

    inertia = random.uniform(0.5,1)
    p_list = []
    gbest = 0

    if explicit: print("Creating particles...")
    #Initialise particles
    for i in range(n_of_particles):
        # p_list.append(Full_Particle(x,y,n_clusters))
        if nn=='prbf':
            if i == 0 and not quiet: print("Starting Polynomial RBF (c=%d)" % (n_clusters))
            p_list.append(Particle(x_train,y_train,n_clusters,inertia))
        elif nn=='rbf':
            if i == 0 and not quiet: print("Starting RBF swarm (c=%d)" % (n_clusters))
            p_list.append(Full_Particle(x_train,y_train,n_clusters,inertia))
        elif nn=='ff':
            if i == 0 and not quiet: print("Starting FF swarm (c=%d)" % (n_clusters))
            p_list.append(FFParticle(x_train,y_train,n_clusters,inertia))
        else:
            raise ValueError('No such NN type.')
        # print("Creating particle", i)
        # try:
        if (p_list[i].getPBest()[0]) < p_list[gbest].getPBest()[0]:
            gbest = i
                # print("New gbest = ", gbest)
        # except TypeError:
        #     gbest = i
            # print("New gbest = ", gbest)
    if explicit: print("Staring gbest = ", p_list[gbest].getPBest()[0])
    if explicit: print("Starting the swarming")
    try:
        for i in range(iterations):
            if not quiet: print("Iteration",i)
            c=0
            for p in p_list:
                pbest, to_print, fitness = p.update(p_list[gbest].getPBest())


                if not quiet and to_print:
                    print("Particle[%d] - New pBest: %f" % (c,pbest[0]))
                elif explicit:
                    print("Particle[%d] - Fitness: %f - pBest: %f" % (c,fitness,pbest[0]))
                if  pbest[0] < p_list[gbest].getPBest()[0]:
                    gbest = c
                    print("Iteration[%d] New gbest = %s" % (i,p_list[gbest].getPBest()[0]))
                c+=1
        if explicit: print("Finished!")
        if explicit: print("gbest = ", p_list[gbest].getPBest()[0])
    except KeyboardInterrupt:
        print("interrupted! running test data now.")

    pbest = p_list[gbest]
    pbest.setPositionToBest()

    if explicit: print("Now using testing data set...")
    if nn=='prbf':
        error = particlePolyRBF(x_test,y_test,pbest.getCenters(),pbest.getSigmas())
    elif nn=='rbf':
        error = particleNet(x_test,y_test,pbest.getCenters(),pbest.getSigmas(),pbest.getW())
    elif nn=='ff':
        error = feedForward(x_test,y_test,n_clusters,pbest.getA(),pbest.getB())

    if not quiet: print("RMSE=",error)
    return error
