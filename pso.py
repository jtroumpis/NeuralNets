from Particle import Particle
from utilities import *
lamda = 1

x, y = readCSV('data.csv')
x = np.asarray(x)
y = np.asarray(y)


# print(len(x),len(x_test),len(x_train))

# print(int(len(x)*0.4))

# print(x.shape)
# for lamda in [0.1,1,10,100]:
#     print("LAMDA=",lamda)
# for c in range(2,30):
#     doThePolyNet(lamda,c,x,y)

n_of_particles = 20
n_clusters = 10
p_list = []
gbest = None

#Initialise particles
for i in range(n_of_particles):
    p_list.append(Particle(x,y,n_clusters))
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
