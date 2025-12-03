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
        self.x = 200
        self.y = 200
        self.theta = 0
        self.sigmaDTheta = 3
        self.sigmaDx = 2
        self.sigmaSensor = 1
        self.map = self.generateMap()

    def commandAndGetData(self, dx, dtheta):
        # Update orientation in DEGREES
        self.theta += dtheta + np.random.normal(scale=self.sigmaDTheta)

        # Normalize theta to [-180, 180] degrees
        self.theta = (self.theta + 180) % 360 - 180

        # Forward noise
        dxTrue = dx + np.random.normal(scale=self.sigmaDx)

        # Use degrees consistently: theta_deg → radians
        rad = self.theta * np.pi / 180.0
        self.x += np.sin(rad) * dxTrue
        self.y += np.cos(rad) * dxTrue

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
        sensorImage = 0.9*sensorImage + 0.1*np.random.random((50, 50))
        return sensorImage