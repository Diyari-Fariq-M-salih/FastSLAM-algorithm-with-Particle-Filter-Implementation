import numpy as np
from scipy.ndimage import rotate

class SLAMParticle:
    def __init__(self, init_pose, map_size=500):
        self.x, self.y, self.theta = init_pose
        self.map = np.zeros((map_size, map_size))  # log-odds map


    # 1. Motion model

    def motion_update(self, dx, dtheta, sigma_xy=1.0, sigma_theta=2.0):
        dx_noisy = dx + np.random.normal(scale=sigma_xy)
        dtheta_noisy = dtheta + np.random.normal(scale=sigma_theta)

        # Teacher sim uses sin(theta/180*pi)
        self.theta += dtheta_noisy
        self.x += np.sin(self.theta / 180 * np.pi) * dx_noisy
        self.y += np.cos(self.theta / 180 * np.pi) * dx_noisy


    # 2. Expected sensor patch

    def expected_sensor_patch(self, patch_size=50):
        xs = int(self.x)
        ys = int(self.y)
        half = patch_size // 2

        # extract map locally around particle pose
        local = self.map[xs-half:xs+half, ys-half:ys+half]

        # rotate according to particle heading
        rotated = rotate(local, angle=self.theta+90, reshape=False)

        return rotated


    # 3. Compute likelihood

    def measurement_likelihood(self, z):
        z_est = self.expected_sensor_patch()
        # compare patches using Gaussian likelihood
        diff = (z - z_est)**2
        return np.exp(-np.sum(diff) / (2 * 0.5))


    # 4. Map update (inverse patch model)

    def update_map(self, z):
        xs = int(self.x)
        ys = int(self.y)
        half = 25

        # unrotate z into world frame
        z_world = rotate(z, angle=-(self.theta+90), reshape=False)

        # convert z to log-odds
        log_odds_update = np.log(z_world + 1e-3) - np.log(1 - z_world + 1e-3)

        # update local area in map
        self.map[xs-half:xs+half, ys-half:ys+half] += log_odds_update
