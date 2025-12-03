import numpy as np
from slam_particle import SLAMParticle

class FastSLAM:
    def __init__(self, num_particles=40, map_size=500):
        self.N = num_particles
        init_pose = (30, 30, 0)
        self.particles = [SLAMParticle(init_pose, map_size) for _ in range(num_particles)]
        self.weights = np.ones(num_particles) / num_particles

    #  MOTION UPDATE 
    def motion_update(self, dx, dtheta):
        for p in self.particles:
            p.motion_update(dx, dtheta)

    #  MEASUREMENT UPDATE 
    def measurement_update(self, z):
        new_w = []
        for p in self.particles:
            w = p.measurement_likelihood(z)
            new_w.append(w)
            p.update_map(z)

        new_w = np.array(new_w) + 1e-12
        self.weights = new_w / np.sum(new_w)

    #  RESAMPLE 
    def resample(self):
        idx = np.random.choice(self.N, self.N, p=self.weights)
        self.particles = [self.particles[i] for i in idx]
        self.weights = np.ones(self.N) / self.N

    #  BEST
    def best_particle(self):
        return self.particles[np.argmax(self.weights)]
