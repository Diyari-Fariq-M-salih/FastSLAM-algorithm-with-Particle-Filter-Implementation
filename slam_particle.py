import numpy as np
from scipy.ndimage import rotate

class SLAMParticle:
    def __init__(self, init_pose, map_size=500):
        self.x, self.y, self.theta = init_pose
        self.map_size = map_size
        self.map = np.zeros((map_size, map_size))  # log-odds map

    # 1. Motion update
    def motion_update(self, dx, dtheta, sigma_dx=1.0, sigma_dtheta=2.0):
        dx_noisy = dx + np.random.normal(scale=sigma_dx)
        dtheta_noisy = dtheta + np.random.normal(scale=sigma_dtheta)

        self.theta += dtheta_noisy

        # keep angle in degrees normalized
        if self.theta > 180:  self.theta -= 360
        if self.theta < -180: self.theta += 360

        # move using teacher’s sin/cos convention
        self.x += np.sin(self.theta / 180 * np.pi) * dx_noisy
        self.y += np.cos(self.theta / 180 * np.pi) * dx_noisy

        # stay inside map bounds
        self.x = np.clip(self.x, 1, self.map_size - 2)
        self.y = np.clip(self.y, 1, self.map_size - 2)

    # 2. Predict expected sensor patch (from particle map)
    def expected_sensor_patch(self, patch_size=50):
        half = patch_size // 2
        xs = int(self.x)
        ys = int(self.y)

        # Get local map patch
        local = self.map[xs-half:xs+half, ys-half:ys+half]

        # If outside, pad with zeros
        if local.shape != (patch_size, patch_size):
            padded = np.zeros((patch_size, patch_size))
            x0 = max(0, half-xs)
            y0 = max(0, half-ys)
            x1 = x0 + local.shape[0]
            y1 = y0 + local.shape[1]
            padded[x0:x1, y0:y1] = local
            local = padded

        # Rotate according to particle heading
        return rotate(local, angle=self.theta + 90, reshape=False)

    # 3. Measurement likelihood
    def measurement_likelihood(self, z):
        z_est = self.expected_sensor_patch()
        diff = (z - z_est) ** 2
        return np.exp(-np.sum(diff) / (2 * 0.5))

    # 4. Map update (inverse sensor model)
    def update_map(self, z):
        patch = 50
        half = patch // 2
        xs = int(self.x)
        ys = int(self.y)

        # Unrotate measurement into world frame
        z_world = rotate(z, angle=-(self.theta + 90), reshape=False)

        # Convert to log-odds
        log_odds_update = np.log(z_world + 1e-3) - np.log(1 - z_world + 1e-3)

        # Update map region
        x1, x2 = xs - half, xs + half
        y1, y2 = ys - half, ys + half

        if 0 <= x1 < self.map_size and 0 <= x2 < self.map_size and \
           0 <= y1 < self.map_size and 0 <= y2 < self.map_size:
            self.map[x1:x2, y1:y2] += log_odds_update
