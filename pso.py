from Particle import Particle, Full_Particle, FFParticle, Pokemon
from utilities import *
from nets import *
import random, json
lamda = 1

def runTestData(nn,x_test, y_test, pbest):
    pbest.setPositionToBest()

    # if explicit: print("Now using testing data set...")
    if nn=='prbf':
        error = particlePolyRBF(x_test,y_test,pbest.getCenters(),pbest.getSigmas())
    elif nn=='rbf':
        error = particleNet(x_test,y_test,pbest.getCenters(),pbest.getSigmas(),pbest.getW())
    elif nn=='ff':
        error = feedForward(x_test,y_test,pbest.n_clusters,pbest.getA(),pbest.getB())
    return error

def saveToFile(name,nn, n_clusters, train_error, testing_error, res_list):
    d = {'name': name, 'nn': nn, 'c': n_clusters, 'train_error': train_error,
         'testing_error': testing_error, 'iter_list': res_list}
    with open('complete_res.json', 'a+') as f:
        json.dump(d,f)
        f.write('\n')

def evolution(name,data,iterations=500,n_clusters=10,nn='prbf', n_of_particles=20,quiet=False,explicit=False):
    x_train,y_train, x_test, y_test = data
    p_list = []
    gbest = 0

    best_res_list={'train': [], 'test': []}

    if explicit: print("Creating particles...")
    #Initialise particles
    for i in range(n_of_particles):
        # p_list.append(Full_Particle(x,y,n_clusters))
        if nn=='rbf':
            if i == 0: print("Starting RBF evolution (c=%d) for %s" % (n_clusters, name))
            p_list.append(Pokemon(x_train,y_train,n_clusters))
        else:
            raise ValueError('No such NN type available.')
        # print("Creating particle", i)
        # try:
        if p_list[i].pbest < p_list[gbest].pbest:
            gbest = i
                # print("New gbest = ", gbest)
        # except TypeError:
        #     gbest = i
            # print("New gbest = ", gbest)
    if explicit: print("Staring gbest = ", p_list[gbest].pbest)
    if explicit: print("Starting the evolution")
    stop_forever = False

    best_res_list['train'].append(p_list[gbest].pbest)
    best_res_list['test'].append(runTestData(nn,x_test,y_test,p_list[gbest]))

    for i in range(iterations):
        try:
            if not quiet: print("Iteration",i)
            c=0
            for p in p_list:
                a = random.randrange(0,n_of_particles)
                b = random.randrange(0,n_of_particles)
                while b == a:
                    b = random.randrange(0,n_of_particles)

                fitness, to_print = p.evolve(p_list[gbest],p_list[a], p_list[b])

                if not quiet and to_print:
                    print("Particle[%d] - New pBest: %f" % (c,fitness))
                elif explicit:
                    print("Particle[%d] - Fitness: %f - pBest: %f" % (c,fitness,p.pbest))
                if  p.pbest < p_list[gbest].pbest:
                    gbest = c
                    print("Iteration[%d] New gbest = %s" % (i,p_list[gbest].pbest))
                c+=1

            best_res_list['train'].append(p_list[gbest].pbest)
            testing_error = runTestData(nn,x_test,y_test,p_list[gbest])
            best_res_list['test'].append(testing_error)

            stop_forever = False
        except KeyboardInterrupt:
            print("interrupted! running test data now.")
            error = runTestData(nn,x_test,y_test,p_list[gbest])
            print("RMSE=",error)
            if stop_forever:
                break
            else:
                stop_forever = True

    if explicit: print("Finished!")
    if explicit: print("gbest = ", p_list[gbest].pbest)

    if explicit: print("Now using testing data set...")

    saveToFile(name,nn,n_clusters,p_list[gbest].pbest,testing_error,best_res_list)
    # saveToFile(name,nn,n_clusters,testing_error,testing=True)
    # save(p_list[gbest].getCenters(),p_list[gbest].getSigmas(),p_list[gbest].getW())
    if not quiet: print("%s: c=%d RMSE=%f" % (name,n_clusters,testing_error))
    return testing_error, p_list[gbest].pbest

def PSO(name,data,iterations=500,n_clusters=10,nn='prbf', n_of_particles=20,quiet=False,explicit=False):
    x_train,y_train, x_test, y_test = data

    inertia = random.uniform(0.5,1)
    p_list = []
    gbest = 0

    best_res_list={'train': [], 'test': []}

    if explicit: print("Creating particles...")
    #Initialise particles
    for i in range(n_of_particles):
        # p_list.append(Full_Particle(x,y,n_clusters))
        if nn=='prbf':
            if i == 0: print("Starting Polynomial RBF (c=%d) for %s" % (n_clusters,name))
            p_list.append(Particle(x_train,y_train,n_clusters,inertia))
        elif nn=='rbf':
            if i == 0: print("Starting RBF swarm (c=%d) for %s" % (n_clusters,name))
            p_list.append(Full_Particle(x_train,y_train,n_clusters,inertia))
        elif nn=='ff':
            if i == 0: print("Starting FF swarm (c=%d) for %s" % (n_clusters,name))
            p_list.append(FFParticle(x_train,y_train,n_clusters,inertia))
        else:
            raise ValueError('No such NN type.')
        # print("Creating particle", i)
        # try:
        if (p_list[i].getPBest()[0]) < p_list[gbest].getPBest()[0]:
            gbest = i


    best_res_list['train'].append(p_list[gbest].getPBest()[0])
    best_res_list['test'].append(runTestData(nn,x_test,y_test,p_list[gbest]))

    if explicit: print("Staring gbest = ", p_list[gbest].getPBest()[0])
    if explicit: print("Starting the swarming")
    stop_forever = False
    for i in range(iterations):
        try:
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

            testing_error = runTestData(nn,x_test,y_test,p_list[gbest])
            best_res_list['train'].append(p_list[gbest].getPBest()[0])
            best_res_list['test'].append(testing_error)

            stop_forever = False
        except KeyboardInterrupt:
            print("interrupted! running test data now.")
            error = runTestData(nn,x_test,y_test,p_list[gbest])
            print("RMSE=",error)
            if stop_forever:
                break
            else:
                stop_forever = True

    if explicit: print("Finished!")
    if explicit: print("gbest = ", p_list[gbest].getPBest()[0])

    if explicit: print("Now using testing data set...")
    saveToFile(name,nn,n_clusters,p_list[gbest].getPBest()[0],testing_error,best_res_list)
    # saveToFile(name,nn,n_clusters,testing_error,testing=True)
    # save(p_list[gbest].getCenters(),p_list[gbest].getSigmas(),p_list[gbest].getW())
    if not quiet: print("%s: c=%d RMSE=%f" % (name,n_clusters,testing_error))
    return testing_error, p_list[gbest].getPBest()[0]
