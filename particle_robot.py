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

    # MEASUREMENT UPDATE + WEIGHT COMPUTATION
    def get_measurement_probability(self, measure):
        sensor_img = measure[0]   # 50×50 image patch
        px, py = int(self.x), int(self.y)

        # If too close to border → cannot compare → neutral weight
        if px < 25 or px > self.map_size - 25 or py < 25 or py > self.map_size - 25:
            return 1.0

        patch = self.map[px-25:px+25, py-25:py+25]

        # If this part of the map is empty → initialize only
        if np.sum(patch) < 1e-6:
            self.map[px-25:px+25, py-25:py+25] = 0.5 * sensor_img
            return 1.0

        # Normalized Cross-Correlation (NCC)
        pred = patch.flatten().astype(np.float32)
        meas = sensor_img.flatten().astype(np.float32)

        pred_norm = (pred - np.mean(pred)) / (np.std(pred) + 1e-6)
        meas_norm = (meas - np.mean(meas)) / (np.std(meas) + 1e-6)

        corr = np.dot(pred_norm, meas_norm) / len(pred_norm)

        # Convert correlation to a positive weight
        weight = np.exp(corr * 4.0)

        # Map Fusion
        fused_patch = 0.7 * patch + 0.3 * sensor_img
        self.map[px-25:px+25, py-25:py+25] = fused_patch

        return weight
