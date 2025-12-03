# particle_robot.py
import numpy as np
from scipy.ndimage import rotate   # <--- NEW

class Particle:
    def __init__(self, params):
        self.map_size = params.get("map_size", 500)
        self.x = 200
        self.y = 200
        self.theta = 0      # degrees, like RobotSim
        self.weight = 1.0

        # SLAM map (occupancy grid)
        self.map = np.zeros((self.map_size, self.map_size))

        # MATCH RobotSim motion noise
        self.sigmaDx = 1            # RobotSim.sigmaDx
        self.sigmaDTheta = 1.5        # RobotSim.sigmaDTheta

    # MOTION MODEL — EXACTLY MATCH RobotSim
    def state_transition(self, command):
        dx, dtheta = command

        # Update orientation (degrees)
        self.theta += dtheta + np.random.normal(scale=self.sigmaDTheta)

        # Normalize heading to [-180, 180]
        self.theta = (self.theta + 180) % 360 - 180

        # Forward noise
        dxTrue = dx + np.random.normal(scale=self.sigmaDx)

        # Update position like RobotSim (degrees → radians)
        rad = self.theta * np.pi / 180
        self.x += np.sin(rad) * dxTrue
        self.y += np.cos(rad) * dxTrue

        # Clamp inside map
        self.x = np.clip(self.x, 25, self.map_size - 25)
        self.y = np.clip(self.y, 25, self.map_size - 25)

    # SENSOR UPDATE + WEIGHT COMPUTATION
    def get_measurement_probability(self, measure):
        # measure is passed as a single argument: (data,)
        sensor_img = measure[0]     # 50×50 noisy patch in ROBOT FRAME

        px, py = int(self.x), int(self.y)

        # If the particle is too close to borders, don't try to update
        if px < 25 or px > self.map_size - 25 or py < 25 or py > self.map_size - 25:
            return 1.0

        # ---- Transform sensor into WORLD frame for this particle ----
        # RobotSim rotated by (theta_true + 90 + noise). For each particle,
        # we "unrotate" using its own heading estimate (theta_particle + 90).
        angle_world = -(self.theta + 90.0)
        sensor_world = rotate(sensor_img, angle=angle_world, reshape=False)

        # Local map patch around particle
        patch = self.map[px-25:px+25, py-25:py+25]

        # If uninitialized region -> just write sensor and give neutral weight
        if np.sum(patch) < 1e-6:
            self.map[px-25:px+25, py-25:py+25] = sensor_world
            return 1.0

        # Simple likelihood: squared-error between map patch and measurement
        diff = patch - sensor_world
        score = np.exp(-np.sum(diff * diff) / 150)

        # Map fusion (exponential moving average)
        fused = 0.7 * patch + 0.3 * sensor_world
        self.map[px-25:px+25, py-25:py+25] = fused

        return score
