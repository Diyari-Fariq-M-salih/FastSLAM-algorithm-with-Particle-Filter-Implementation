import numpy as np
from slam_particle import SLAMParticle

class FastSLAM:
    def __init__(self, num_particles=50, map_size=500):
        self.N = num_particles
        self.particles = [
            SLAMParticle((30,30,0), map_size) for _ in range(num_particles)
        ]
        self.weights = np.ones(num_particles) / num_particles

    
    # 1. Motion update
    
    def motion_update(self, dx, dtheta):
        for p in self.particles:
            p.motion_update(dx, dtheta)

    
    # 2. Measurement update
    
    def measurement_update(self, z):
        new_weights = []
        for p in self.particles:
            w = p.measurement_likelihood(z)
            new_weights.append(w)
            p.update_map(z)

        w = np.array(new_weights)
        w += 1e-300
        self.weights = w / np.sum(w)

    
    # 3. Resampling
    
    def resample(self):
        idx = np.random.choice(self.N, self.N, p=self.weights)
        self.particles = [self.particles[i] for i in idx]
        self.weights = np.ones(self.N) / self.N

    
    # Utility: best estimate
    
    def best_particle(self):
        return self.particles[np.argmax(self.weights)]
