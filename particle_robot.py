# particle_robot.py
import numpy as np
from scipy.ndimage import rotate

class Particle:
    def __init__(self, params):
        self.map_size = params.get("map_size", 500)
        self.x = 30
        self.y = 30
        self.theta = 0
        self.weight = 1.0
        
        # occupancy grid map for the particle
        self.map = np.zeros((self.map_size, self.map_size))

        # noise parameters
        self.sigmaDx = 1.5
        self.sigmaDTheta = 2.0

    def state_transition(self, command):
        dx, dtheta = command

        # Add motion noise
        noisy_dx = dx + np.random.normal(scale=self.sigmaDx)
        noisy_dtheta = dtheta + np.random.normal(scale=self.sigmaDTheta)

        # Update orientation
        self.theta += noisy_dtheta
        if self.theta > 180: self.theta -= 360
        if self.theta < -180: self.theta += 360

        # Update position
        self.x += np.sin(np.radians(self.theta)) * noisy_dx
        self.y += np.cos(np.radians(self.theta)) * noisy_dx

        # Clamp inside map
        self.x = np.clip(self.x, 1, self.map_size-2)
        self.y = np.clip(self.y, 1, self.map_size-2)

    def get_measurement_probability(self, measure):
        sensor_img = measure[0]  # 50x50 patch
        px, py = int(self.x), int(self.y)

        # Extract predicted patch from particle map
        patch = self.map[px-25:px+25, py-25:py+25]

        # If map still empty, return neutral weight
        if patch.shape != (50, 50):
            return 1.0

        # Compute similarity (simple SSD)
        diff = patch - sensor_img
        score = np.exp(-np.sum(diff*diff) / 50.0)

        # Update map: Bayesian fusion
        self.map[px-25:px+25, py-25:py+25] = (
            0.6 * self.map[px-25:px+25, py-25:py+25] +
            0.4 * sensor_img
        )

        return score
