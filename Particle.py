from utilities import *
from nets import *
import random

class Particle():
    def __init__(self,x,y,n_clusters,inertia,netType=particlePolyRBF):
        self.n_clusters = n_clusters
        self.x = x
        self.y = y
        self.position = []
        self.vel = []
        self.inertia = inertia
        self.p = len(x[0])
        self.net = netType

        x_min = np.amin(x, axis=0)
        x_max = np.amax(x, axis=0)

        self.position.extend(self.initCenters(n_clusters,0,1))
        self.vel.extend(self.initCenters(n_clusters,0,1))
        self.position.extend(self.initSigmas(n_clusters,0.1,0.5))
        self.vel.extend(self.initSigmas(n_clusters,0.1,0.5))

        self.pbest = (self.net(x,y,self.getCenters(),self.getSigmas()),list(self.position))


    def setPositionToBest(self):
        self.position = self.pbest[1]

    # Add to the position c*p variables as the centers
    def initCenters(self,n_clusters,x_min,x_max):
        # print('p=',len(x_min))
        pos = []
        for c in range(n_clusters):
            for p in range(self.p):
                temp = random.uniform(x_min[p],x_max[p])
                pos.append(temp)
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

    def getCenters(self, ar = None):
        if not ar: ar = self.position
        centers = []
        for c in range(self.n_clusters):
            center = []
            for p in range(self.p):
                center.append(ar[c*p+p])
            centers.append(center)
        return np.asarray(centers)

    def getSigmas(self, ar = None):
        if not ar: ar = self.position
        sigmas = []
        for c in range(self.n_clusters*self.p,self.n_clusters*self.p+self.n_clusters):
            sigmas.append(ar[c])

        return np.asarray(sigmas)

    def getW(self, ar = None):
        if not ar: ar = self.position
        ws = []
        for c in range(self.n_clusters*self.p+self.n_clusters,len(self.position)):
            ws.append(ar[c])

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

    def checkPBest(self,fitness):
        if fitness < self.pbest[0]:
            # print("New pbest!", fitness)
            self.pbest = (fitness,list(self.position))
            return self.pbest, True, fitness
        return self.pbest, False, fitness

    def update(self,gBest):
        self.updatePosition()
        self.updateVelocity(gBest)

        # print(self.vel)
        try:
            fitness = self.net(self.x,self.y,self.getCenters(),self.getSigmas())
        except np.linalg.linalg.LinAlgError:
            fitness = self.pbest[0]+1
        # print(fitness)

        return self.checkPBest(fitness)

class Full_Particle(Particle):
    def __init__(self,x,y,n_clusters, inertia, netType=particleNet):
        self.n_clusters = n_clusters
        self.x = x
        self.y = y
        self.position = []
        self.vel = []
        self.inertia = inertia
        # print(len(x[0]))
        self.p = len(x[0])


        x_min = np.amin(x, axis=0)
        x_max = np.amax(x, axis=0)

        self.position.extend(self.initCenters(n_clusters,x_min,x_max))
        self.vel.extend(self.initCenters(n_clusters,x_min,x_max))
        self.position.extend(self.initSigmas(n_clusters,1000000,1000000000))
        self.vel.extend(self.initSigmas(n_clusters,1000000,1000000000))
        self.position.extend(self.initW(n_clusters))
        self.vel.extend(self.initW(n_clusters))

        self.pbest = (particleNet(x,y,self.getCenters(),self.getSigmas(),self.getW()),list(self.position))

        # self.vel = []
        # # print(len(self.position))
        # for i in range(len(self.position)):
        #     self.vel.append(random.uniform((-10),(10)))

    def initW(self,n_clusters):
        pos = []
        for c in range(n_clusters):
            pos.append(random.uniform(-1,1))
        return pos

    def getW(self, ar=None):
        if not ar: ar = self.position
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
        return self.checkPBest(fitness)

class Pokemon(Full_Particle):
    def __init__(self,x,y,n_clusters, Fr=0.6, Cr=0.9, netType=particleNet):
        self.n_clusters = n_clusters
        self.x = x
        self.y = y
        self.position = []
        self.Fr = Fr
        self.Cr = Cr
        self.p = len(x[0])


        x_min = np.amin(x, axis=0)
        x_max = np.amax(x, axis=0)

        self.position.extend(self.initCenters(n_clusters,x_min,x_max))
        self.position.extend(self.initSigmas(n_clusters,1000000,1000000000))
        self.position.extend(self.initW(n_clusters))

        self.temp_pos = self.position

        self.pbest = particleNet(self.x,self.y,self.getCenters(),self.getSigmas(),self.getW())

    def setPositionToBest(self):
        pass

    def mutate(self,j):
        rand = random.uniform(0,1)
        j0 = random.randrange(0,self.p)
        return rand < self.Cr or j0 == j

    def evolve(self,best,a,b):
        for j in range(len(self.position)):
            if self.mutate(j):
                self.temp_pos[j] = best.position[j] + self.Fr*(a.position[j] - b.position[j])

        fitness = particleNet(self.x,self.y,self.getCenters(self.temp_pos),
                            self.getSigmas(self.temp_pos),self.getW(self.temp_pos))

        if fitness < self.pbest:
            self.position = self.temp_pos
            self.pbest = fitness
            return fitness, True
        return fitness, False

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
            self.position.extend([random.uniform(-0.5, 0.5) for _ in range(self.p)])
            self.vel.extend([random.uniform(-0.5, 0.5) for _ in range(self.p)])
        # This is the Bs
        self.position.extend([random.uniform(-0.5, 0.5) for _ in range(n_clusters)])
        self.vel.extend([random.uniform(-0.5, 0.5) for _ in range(n_clusters)])

        self.pbest = (feedForward(x,y,self.n_clusters,self.getA(),self.getB()),list(self.position))

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
        return self.checkPBest(fitness)
