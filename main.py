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

        
        # 1. Use best particle's map to plan motion
        
        best = slam.best_particle()
        dx, dtheta = potential_field_planner(best.map, best.x, best.y, best.theta)

        
        # 2. Move robot & get measurement
        
        try:
            z, gt = sim.commandAndGetData(dx, dtheta)
        except Exception as e:
            print("Simulation ended:", e)
            break

        
        # 3. FastSLAM updates
        
        slam.motion_update(dx, dtheta)
        slam.measurement_update(z)

        # resample if needed
        if 1.0 / np.sum(slam.weights**2) < slam.N / 2:
            slam.resample()

        
        # 4. Show maps
        
        if step % 5 == 0:
            plt.clf()
            plt.subplot(121)
            plt.title("Ground truth")
            plt.imshow(sim.map, vmin=0, vmax=1)

            plt.subplot(122)
            plt.title("SLAM map")
            plt.imshow(best.map, vmin=-2, vmax=2)
            plt.pause(0.01)

    plt.ioff()
    plt.show()
