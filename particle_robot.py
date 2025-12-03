# particle_robot.py
import numpy as np

class Particle:
    def __init__(self, params):
        self.map_size = params.get("map_size", 500)
        self.x = 30
        self.y = 30
        self.theta = 0      # degrees, like RobotSim
        self.weight = 1.0

        # SLAM map (occupancy grid)
        self.map = np.zeros((self.map_size, self.map_size))

        # MATCH RobotSim motion noise
        self.sigmaDx = 2            # RobotSim.sigmaDx
        self.sigmaDTheta = 3        # RobotSim.sigmaDTheta

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
        self.x = np.clip(self.x, 1, self.map_size - 2)
        self.y = np.clip(self.y, 1, self.map_size - 2)

    # SENSOR UPDATE + WEIGHT COMPUTATION
    def get_measurement_probability(self, measure):
        sensor_img = measure[0]     # 50×50 noisy patch
        px, py = int(self.x), int(self.y)

        # Skip if close to borders
        if px < 25 or px > self.map_size - 25 or py < 25 or py > self.map_size - 25:
            return 1.0

        patch = self.map[px-25:px+25, py-25:py+25]

        # If uninitialized region
        if np.sum(patch) < 1e-6:
            self.map[px-25:px+25, py-25:py+25] = 0.5 * sensor_img
            return 1.0

        # Simple similarity metric (NCC is unstable here)
        diff = patch - sensor_img
        score = np.exp(-np.sum(diff * diff) / 200.0)

        # Map fusion
        fused = 0.7 * patch + 0.3 * sensor_img
        self.map[px-25:px+25, py-25:py+25] = fused

        return score
