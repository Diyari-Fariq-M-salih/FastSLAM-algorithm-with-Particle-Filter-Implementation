#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2

from particlesFilter import ParticlesFilter
from robot_simulator import RobotSim
# from robot import RobotModel


if __name__ == "__main__":

    # part_filter = ParticlesFilter(100, RobotModel, map_w=500, map_h=500)

    sim = RobotSim()

    # Initial coordinates : (x, y, theta) = (30, 30, 0)
    # x and y: pixels
    # theta: degrees

    # TODO: Particle filter with a (x, y, theta) vector and a 500x500 pixels
    #       occupancy map for each particle

    plt.ion()
    i = 0
    
    while True:
        
        print('iteration ', i)
        try:
            data, coordGT = sim.commandAndGetData(3, 6)
            # data: scan of the surrounding of the robot
            #       (50x50 image in robot frame)
            # coordGT: Ground Truth of the actual position
            #           of the robot
            # Parameters of the function: dx and dtheta, control
            #                               command of the robot
        except Exception as e:
            print(repr(e))
            break
        sim.map[int(coordGT[0]), int(coordGT[1])] = 0.5
        plt.clf()
        plt.subplot(121)
        plt.imshow(sim.map, interpolation="None", vmin=0, vmax=1)
        plt.subplot(122)
        plt.imshow(data, interpolation="None", vmin=0, vmax=1)
        plt.show()
        i += 1
        plt.pause(0.01)

    plt.ioff()
    plt.show()