from robot_simulator import RobotSim
from fastslam_filter import FastSLAM
from planner import potential_field_planner
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    sim = RobotSim()
    slam = FastSLAM(num_particles=30, map_size=500)

    plt.ion()

    step = 0
    while True:
        step += 1
        print("STEP", step)

        
        # 1. Grab best estimate
        
        best = slam.best_particle()

        
        # 2. Planner
        
        dx, dtheta = potential_field_planner(best.map, best.x, best.y, best.theta)

        
        # 3. Move robot and get measurement
        
        try:
            z, gt = sim.commandAndGetData(dx, dtheta)
        except Exception as e:
            print("Simulation stopped:", e)
            break

        
        # 4. FastSLAM update
        
        slam.motion_update(dx, dtheta)
        slam.measurement_update(z)

        # resampling
        Neff = 1.0 / np.sum(slam.weights**2)
        if Neff < slam.N / 2:
            slam.resample()

        
        # 5. Display
        
        if step % 5 == 0:
            plt.clf()
            plt.subplot(131)
            plt.title("Ground truth")
            plt.imshow(sim.map, vmin=0, vmax=1)

            plt.subplot(132)
            plt.title("Best Particle Map")
            plt.imshow(best.map, vmin=-2, vmax=2)

            plt.subplot(133)
            plt.title("Sensor Patch")
            plt.imshow(z, vmin=0, vmax=1)

            plt.pause(0.01)

    plt.ioff()
    plt.show()
