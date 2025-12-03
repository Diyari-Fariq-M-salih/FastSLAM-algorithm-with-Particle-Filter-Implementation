# coding: utf-8

'''
Created on 26 nov 2025
Modified on 28 oct 2025

@author: Fabien Bonardi
'''

import random
from copy import deepcopy
import numpy as np
from scipy.ndimage import rotate


class RobotSim:
    '''
    classdocs
    '''
    def __init__(self):
        self.x = 30
        self.y = 30
        self.theta = 0
        self.sigmaDTheta = 3
        self.sigmaDx = 2
        self.sigmaSensor = 1
        self.map = self.generateMap()

    def commandAndGetData(self, dx, dtheta):
        self.theta += dtheta + np.random.normal(scale=self.sigmaDTheta)
        if self.theta > np.pi:
            self.theta -= 2*np.pi
        if self.theta < -np.pi:
            self.theta += 2*np.pi
        dxTrue = dx + np.random.normal(scale=self.sigmaDx)
        self.x += np.sin(self.theta/180*np.pi)*dxTrue
        self.y += np.cos(self.theta/180*np.pi)*dxTrue
        print(f"val x {self.x} val y {self.y} val theta {self.theta}")
        if self.map[int(self.x), int(self.y)] == 1:
            raise Exception("CRASH ON OBSTACLE!")
        if int(self.x) == 450 and int(self.y) == 450:
            raise Exception("Goal reached!")
        return self.generateData(), (self.x, self.y)

    def generateMap(self, map_size=500):
        map = np.zeros((map_size, map_size))
        wBound = 25
        map[:wBound,:] = 1
        map[:,-wBound:] = 1
        map[-wBound:,:] = 1
        map[:,:wBound] = 1
        for i in range(50):
            xObs = int(random.random()*(map_size-100)+50)
            yObs = int(random.random()*(map_size-100)+50)
            map[xObs-10:xObs+11, yObs-10:yObs+11] = 1
        return map

    def generateData(self):
        xt = int(self.x)
        yt = int(self.y)
        sensorImage = self.map[xt-25:xt+25, yt-25:yt+25]
        sensorImage = rotate(sensorImage,
                             angle=self.theta
                                    + 90
                                    + np.random.normal(scale=self.sigmaSensor),
                             reshape=False)
        sensorImage = 0.4*sensorImage + 0.6*np.random.random((50, 50))
        return sensorImage