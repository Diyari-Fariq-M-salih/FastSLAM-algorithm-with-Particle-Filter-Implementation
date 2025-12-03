#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2

from particle_robot import Particle
from particlesFilter import ParticlesFilter
from robot_simulator import RobotSim

# Compute safe direction using obstacle gradient
def compute_safe_direction(sensor_img):
    # Smooth noise
    blur = sensor_img - np.mean(sensor_img)

    h, w = blur.shape
    cx, cy = h // 2, w // 2

    gx, gy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    vx = gx - cx
    vy = gy - cy

    ox = np.sum(blur * vx)
    oy = np.sum(blur * vy)

    fx = -ox
    fy = -oy

    norm = np.sqrt(fx*fx + fy*fy) + 1e-6
    fx /= norm
    fy /= norm
    return fx, fy


if __name__ == "__main__":

    sim = RobotSim()
    pf = ParticlesFilter(50, Particle, map_size=500)

    plt.ion()
    i = 0

    # INITIAL SENSOR READING
    data = sim.generateData()

    forward = 0
    turn = 0
    trajectory_x = []
    trajectory_y = []

    while True:
        print("iteration", i)

        try:

            # 1. MOVE THE ROBOT FIRST

            data, coordGT = sim.commandAndGetData(forward, turn)
            sim.theta = (sim.theta + 180) % 360 - 180
            trajectory_x.append(coordGT[0])
            trajectory_y.append(coordGT[1])
            


            # 2. SAFE ROTATION-INVARIANT AVOIDANCE

            h, w = data.shape
            cx, cy = h//2, w//2

            num_dirs = 16
            angles = np.linspace(-180, 180, num_dirs)

            best_score = 999
            best_angle = 0

            for a in angles:
                ang = np.radians(a)     # rotation-invariant
                sx = int(cx + 15*np.sin(ang))
                sy = int(cy + 15*np.cos(ang))

                if sx < 0 or sx >= h or sy < 0 or sy >= w:
                    continue

                score = data[sx, sy]
                if score < best_score:
                    best_score = score
                    best_angle = a


            # 2b. WALL REPULSION FORCE (push robot away from walls)

            repel_x = 0
            repel_y = 0
            margin = 100    # how close before repulsion starts

            rx, ry = coordGT  # robot true coordinates

            if rx < margin:
                repel_x += (margin - rx)
            if rx > 500 - margin:
                repel_x -= (rx - (500 - margin))

            if ry < margin:
                repel_y += (margin - ry)
            if ry > 500 - margin:
                repel_y -= (ry - (500 - margin))

            # Normalize repulsion
            mag = np.sqrt(repel_x**2 + repel_y**2) + 1e-6
            fx_repel = repel_x / mag
            fy_repel = repel_y / mag

            # Safe-direction unit vector
            fx_safe = np.sin(np.radians(best_angle))
            fy_safe = np.cos(np.radians(best_angle))

            # Blend safe + repulsion
            fx_total = 0.7 * fx_safe + 0.3 * fx_repel
            fy_total = 0.7 * fy_safe + 0.3 * fy_repel


            # Convert total vector to heading (+90° shift)

            desired_theta = np.degrees(np.arctan2(fx_total, fy_total)) + 90
            desired_theta = (desired_theta + 180) % 360 - 180

            dtheta = desired_theta - sim.theta
            if dtheta > 180: dtheta -= 360
            if dtheta < -180: dtheta += 360

            turn = np.clip(dtheta, -10, 10)
            forward = 1.5
            # --- Emergency obstacle avoidance straight ahead ---
            # Look at a small window in the top of the sensor patch (in robot frame)
            front_strip = data[0:10, 20:30]   # rows near "forward", center columns
            front_intensity = np.mean(front_strip)

            if front_intensity > 0.4:
                # Big obstacle very close ahead: turn hard and slow down
                turn = np.sign(turn) * 25 if turn != 0 else 25  # strong turn
                forward = 0.1                                   # creep forward



            # 3. PF update

            pf.compute_state_transition(forward, turn)
            pf.compute_weights(data)

            best_index = np.argmax(pf.weights)
            best_particle = pf.particles_set[best_index]
            slam_map = best_particle.map

            pf.resample_w()

        except Exception as e:
            print(repr(e))
            break

        # Mark robot GT on map
        sim.map[int(coordGT[0]), int(coordGT[1])] = 0.5

        # Visualization
        plt.clf()

        plt.subplot(121)
        plt.title("Ground Truth Map")
        plt.imshow(sim.map.T, origin='lower', vmin=0, vmax=1)

        # Draw trajectory
        if len(trajectory_x) > 1:
            plt.plot(trajectory_x, trajectory_y, color='white', linewidth=1)

        plt.subplot(122)
        plt.title("Sensor Patch")
        plt.imshow(data, vmin=0, vmax=1)

        # plt.subplot(133)
        # plt.title("SLAM (Best Particle Map)")
        # plt.imshow(slam_map.T, origin='lower', vmin=0, vmax=1)

        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.show()
        plt.pause(0.01)

        i += 1

    plt.ioff()
    plt.show()
