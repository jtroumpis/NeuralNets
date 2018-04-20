from Particle import Particle, Full_Particle, FFParticle
from utilities import *
from nets import *
import random
lamda = 1

def PSO(x,y,iterations=1000,nn='prbf',n_clusters=10, n_of_particles=20):
    inertia = random.uniform(0.5,1)
    p_list = []
    gbest = None

    print("Creating particles...")
    #Initialise particles
    for i in range(n_of_particles):
        # p_list.append(Full_Particle(x,y,n_clusters))
        if nn=='prbf':
            p_list.append(Particle(x,y,n_clusters,inertia))
        elif nn=='rbf':
            p_list.append(Full_Particle(x,y,n_clusters,inertia))
        elif nn=='ff':
            p_list.append(FFParticle(x,y,n_clusters,inertia))
        else:
            raise ValueError('No such NN type.')
        # print("Creating particle", i)
        try:
            if (p_list[i].getPBest()[0]) < gbest[0]:
                gbest = p_list[i].getPBest()
                # print("New gbest = ", gbest)
        except TypeError:
            gbest = p_list[i].getPBest()
            # print("New gbest = ", gbest)
    print("Staring gbest = ", gbest)
    print("Starting the swarming")
    for i in range(iterations):
        print("Iteration",i)
        c=0
        for p in p_list:
            pbest, to_print = p.update(gbest)
            c+=1
            if to_print:
                print("Particle[%d] - New pBest: %f" % (c,pbest[0]))
            if  pbest[0] < gbest[0]:
                gbest = pbest
                print("New gbest = ", gbest)

    print("Finished!")
    print("gbest = ", gbest)
