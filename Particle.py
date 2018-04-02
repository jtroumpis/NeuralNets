from utilities import *
import random
class Point():
    def __init__(pos=None):
        self.position = pos

    def randomPosition(x_min,x_max):
        self.position = []
        for p in range(len(x_min)):
            self.position.append(random.uniform(x_min[p],x_max[p]))

class Particle():
    def __init__(self,x,y,n_clusters):
        self.n_clusters = n_clusters
        self.x = x
        self.y = y

        x_min = np.amin(x, axis=0)
        x_max = np.amax(x, axis=0)

        pos = []
        # Add to the position c*p variables as the centers
        for c in range(n_clusters):
            for p in range(len(x_min)):
                pos.append(random.uniform(x_min[p],x_max[p]))

        # Add to the position c variables as the sigmas
        for c in range(n_clusters):
            pos.append(random.uniform(min(x_min),max(x_max)))

        self.position = pos

        self.pbest = (doTheParticleNet(x,y,self.getCenters(),self.getSigmas()),pos)

        self.vel = []
        # print(min(x_min),max(x_max))
        for i in range(n_clusters*len(x_min)+n_clusters):
            self.vel.append(random.uniform(-max(x_max),max(x_max)))

    def getPBest(self):
        return self.pbest

    def getCenters(self):
        centers = []
        for c in range(self.n_clusters):
            center = []
            for p in range(len(self.x[0])):
                center.append(self.position[c*p+p])
            centers.append(center)
        return np.asarray(centers)

    def getSigmas(self):
        sigmas = []
        for c in range(len(self.position)-self.n_clusters,len(self.position)):
            sigmas.append(self.position[c])

        return np.asarray(sigmas)

    def updateVelocity(self,gBest):
        w = random.uniform(0.5,1)
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
        fitness = doTheParticleNet(self.x,self.y,self.getCenters(),self.getSigmas())
        if fitness < self.pbest[0]:
            print("New pbest!")
            self.pbest = (fitness,self.position)

        return self.pbest
