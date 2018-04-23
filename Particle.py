from utilities import *
from nets import *
import random

class Particle():
    def __init__(self,x,y,n_clusters,inertia):
        self.n_clusters = n_clusters
        self.x = x
        self.y = y
        self.position = []
        self.vel = []
        self.inertia = inertia
        self.p = len(x[0])

        x_min = np.amin(x, axis=0)
        x_max = np.amax(x, axis=0)

        self.position.extend(self.initCenters(n_clusters,x_min,x_max))
        self.vel.extend(self.initCenters(n_clusters,x_min,x_max))
        self.position.extend(self.initSigmas(n_clusters,1000000,1000000000))
        self.vel.extend(self.initSigmas(n_clusters,1000000,1000000000))

        self.pbest = (particlePolyRBF(x,y,self.getCenters(),self.getSigmas()),self.position)


        # for i in range(len(self.position)):
        #     self.vel.append(random.uniform((-10),(10)))

    # Add to the position c*p variables as the centers
    def initCenters(self,n_clusters,x_min,x_max):
        # print('p=',len(x_min))
        pos = []
        for c in range(n_clusters):
            for p in range(self.p):
                pos.append(random.uniform(x_min[p],x_max[p]))
        return pos

    # Add to the position c variables as the sigmas
    def initSigmas(self,n_clusters,minimum,maximum):
        pos = []
        for c in range(n_clusters):
            pos.append(random.uniform(minimum,maximum))
        return pos

    def initW(self,n_clusters):
        pos = []
        for c in range(n_clusters*(self.p+1)):
            pos.append(random.uniform(0,1))
        return pos

    def getPBest(self):
        return self.pbest

    def getCenters(self):
        centers = []
        for c in range(self.n_clusters):
            center = []
            for p in range(self.p):
                center.append(self.position[c*p+p])
            centers.append(center)
        return np.asarray(centers)

    def getSigmas(self):
        sigmas = []
        for c in range(self.n_clusters*self.p,self.n_clusters*self.p+self.n_clusters):
            sigmas.append(self.position[c])

        return np.asarray(sigmas)

    def getW(self):
        ws = []
        for c in range(self.n_clusters*self.p+self.n_clusters,len(self.position)):
            ws.append(self.position[c])

        return np.asarray(ws)

    def updateVelocity(self,gBest):
        w = self.inertia
        c = 2
        r = random.uniform(0,1)
        for i in range(len(self.vel)):
            self.vel[i] = w * self.vel[i]
            self.vel[i] += c * r * (self.pbest[1][i] - self.position[i])
            self.vel[i] += c * r * (gBest[1][i] - self.position[i])

    def updatePosition(self):
        for i in range(len(self.position)):
            self.position[i] += self.vel[i]

    def update(self,gBest):
        self.updatePosition()
        self.updateVelocity(gBest)

        # print(self.vel)
        fitness = particlePolyRBF(self.x,self.y,self.getCenters(),self.getSigmas())
        # print(fitness)
        if fitness < self.pbest[0]:
            # print("New pbest!", fitness)
            self.pbest = (fitness,self.position)
            return self.pbest, True, fitness
        return self.pbest, False, fitness

class Full_Particle(Particle):
    def __init__(self,x,y,n_clusters, inertia):
        self.n_clusters = n_clusters
        self.x = x
        self.y = y
        self.position = []
        self.vel = []
        self.inertia = inertia
        self.p = len(x[0])

        x_min = np.amin(x, axis=0)
        x_max = np.amax(x, axis=0)

        self.position.extend(self.initCenters(n_clusters,x_min,x_max))
        self.vel.extend(self.initCenters(n_clusters,x_min,x_max))
        self.position.extend(self.initSigmas(n_clusters,1000000,1000000000))
        self.vel.extend(self.initSigmas(n_clusters,1000000,1000000000))
        self.position.extend(self.initW(n_clusters))
        self.vel.extend(self.initW(n_clusters))

        self.pbest = (particleNet(x,y,self.getCenters(),self.getSigmas(),self.getW()),self.position)

        # self.vel = []
        # # print(len(self.position))
        # for i in range(len(self.position)):
        #     self.vel.append(random.uniform((-10),(10)))

    def initW(self,n_clusters):
        pos = []
        for c in range(n_clusters):
            pos.append(random.uniform(0,1))
        return pos

    def getW(self):
        ws = []
        for c in range(self.n_clusters*self.p+self.n_clusters,len(self.position)):
            ws.append(self.position[c])

        return np.asarray(ws)

    def update(self,gBest):
        self.updatePosition()
        self.updateVelocity(gBest)
        # print(self.vel)
        fitness = particleNet(self.x,self.y,self.getCenters(),self.getSigmas(),self.getW())
        # print(fitness)
        if fitness < self.pbest[0]:
            # print("New pbest!", fitness)
            self.pbest = (fitness,self.position)
            return self.pbest, True, fitness
        return self.pbest, False, fitness

class FFParticle(Particle):
    def __init__(self,x,y,n_clusters,inertia):
        self.n_clusters = n_clusters
        self.x = x
        self.y = y
        self.position = []
        self.vel = []
        self.inertia = inertia
        self.p = len(x[0])

        x_min = np.amin(x, axis=0)
        x_max = np.amax(x, axis=0)

        #Initialise position
        # This is the As
        for j in range(n_clusters):
            self.position.extend([random.uniform(-100, 100) for _ in range(self.p)])
            self.vel.extend([random.uniform(-100, 100) for _ in range(self.p)])
        # This is the Bs
        self.position.extend([random.uniform(-100, 100) for _ in range(n_clusters)])
        self.vel.extend([random.uniform(-100, 100) for _ in range(n_clusters)])

        self.pbest = (feedForward(x,y,self.n_clusters,self.getA(),self.getB()),self.position)

        # self.vel = []
        # for i in range(len(self.position)):
        #     self.vel.append(random.uniform((-10),(10)))

    def getA(self):
        pos = []
        for i in range(self.n_clusters):
            part_pos = []
            for j in range(self.p):
                part_pos.append(self.position[i*self.p+j])
            pos.append(part_pos)
        return pos

    def getB(self):
        pos = []
        for i in range(self.n_clusters*self.p,len(self.position)):
            pos.append(self.position[i])
        return pos

    def update(self,gBest):
        self.updatePosition()
        self.updateVelocity(gBest)
        # print(self.vel)

        fitness = feedForward(self.x,self.y,self.n_clusters,self.getA(),self.getB())
        # print (fitness)
        if fitness < self.pbest[0]:
            # print("New pbest!", fitness)
            self.pbest = (fitness,self.position)
            return self.pbest, True, fitness
        return self.pbest, False, fitness
