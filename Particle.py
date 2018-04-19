from utilities import *
import random
# class Point():
#     def __init__(pos=None):
#         self.position = pos
#
#     def randomPosition(x_min,x_max):
#         self.position = []
#         for p in range(len(x_min)):
#             self.position.append(random.uniform(x_min[p],x_max[p]))

class Particle():
    def __init__(self,x,y,n_clusters,inertia):
        self.n_clusters = n_clusters
        self.x = x
        self.y = y
        self.position = []
        self.inertia = inertia
        self.p = len(x[0])
        # self.posDict = {'centers': [], 'sigmas': [], 'W': []}

        x_min = np.amin(x, axis=0)
        x_max = np.amax(x, axis=0)

        self.position.extend(self.initCenters(n_clusters,x_min,x_max))
        self.position.extend(self.initSigmas(n_clusters,x_min,x_max))

        self.pbest = (particlePolyRBF(x,y,self.getCenters(),self.getSigmas()),self.position)

        self.vel = []
        for i in range(len(self.position)):
            self.vel.append(random.uniform(-max(x_max),max(x_max)))

    # Add to the position c*p variables as the centers
    def initCenters(self,n_clusters,x_min,x_max):
        # print('p=',len(x_min))
        pos = []
        for c in range(n_clusters):
            for p in range(self.p):
                pos.append(random.uniform(x_min[p],x_max[p]))
        return pos

    # Add to the position c variables as the sigmas
    def initSigmas(self,n_clusters,x_min,x_max):
        pos = []
        for c in range(n_clusters):
            pos.append(random.uniform(min(x_min),max(x_max)))
        return pos

    def initW(self,n_clusters):
        pos = []
        for c in range(n_clusters*(self.p+1)):
            pos.append(random.uniform(0.000000000001,10000000000))
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

        self.updateVelocity(gBest)
        self.updatePosition()
        # print(self.vel)
        fitness = particlePolyRBF(self.x,self.y,self.getCenters(),self.getSigmas())
        if fitness < self.pbest[0]:
            print("New pbest!")
            self.pbest = (fitness,self.position)

        return self.pbest

class Full_Particle(Particle):
    def __init__(self,x,y,n_clusters, inertia):
        self.n_clusters = n_clusters
        self.x = x
        self.y = y
        self.position = []
        self.inertia = inertia
        self.p = len(x[0])

        x_min = np.amin(x, axis=0)
        x_max = np.amax(x, axis=0)

        self.position.extend(self.initCenters(n_clusters,x_min,x_max))
        self.position.extend(self.initSigmas(n_clusters,x_min,x_max))
        self.position.extend(self.initW(n_clusters))

        self.pbest = (particlePolyRBF(x,y,self.getCenters(),self.getSigmas(),self.getW()),self.position)

        self.vel = []
        # print(len(self.position))
        for i in range(len(self.position)):
            self.vel.append(random.uniform(-max(x_max),max(x_max)))

    def update(self,gBest):

        self.updateVelocity(gBest)
        self.updatePosition()
        # print(self.vel)
        fitness = particlePolyRBF(self.x,self.y,self.getCenters(),self.getSigmas(),self.getW())
        if fitness < self.pbest[0]:
            print("New pbest!")
            self.pbest = (fitness,self.position)

        return self.pbest
