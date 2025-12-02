# coding: utf-8

'''
Created on 10 jan 2021

@author: Fabien Bonardi
'''

import random
import copy


class ParticlesFilter:
    '''
    classdocs
    '''
    def __init__(self, particles_nbr, particles_type, **init_params):
        self.particles_number = particles_nbr
        self.particles_set = []
        self.weights = []
        self.max_weight = 0
        # Sampling step
        for _ in range(self.particles_number):
            particle = particles_type(init_params)
            self.particles_set.append(particle)
            self.weights.append(1.0)

    def compute_state_transition(self, *command):
        for particle in self.particles_set:
            particle.state_transition(command)

    def compute_weights(self, *measure):
        new_weights = []
        self.max_weight = 0
        for particle in self.particles_set:
            weight = particle.get_measurement_probability(measure)
            new_weights.append(weight)
            if weight > self.max_weight:
                self.max_weight = weight
        self.weights = new_weights

    def normalize_weights(self):
        max_weight = max(self.weights)
        for weight in self.weights:
            weight /= max_weight

    def resample(self):
        new_particles_set = []
        for index, particle in enumerate(self.particles_set):
            x, y = particle.x, particle.y
            if self.weights[index] > self.max_weight - 5 and \
                    self.weights[index] != 0 and \
                    len(new_particles_set) < self.particles_number:
                new_particles_set.append(copy.deepcopy(particle))
                new_particles_set.append(particle)
        nbr = len(new_particles_set)
        i = 0
        nbr_left = self.particles_number - nbr
        for index, particle in enumerate(self.particles_set):
            if self.weights[index] == 0:
                new_particles_set.append(particle)
                i += 1
            if i == nbr_left:
                self.particles_set = new_particles_set
                break
        self.particles_set = new_particles_set

    # Resampling wheel
    def resample_w(self):
        new_particles_set = []
        index = int(random.random() * self.particles_number)
        beta = 0.0
        max_weight = max(self.weights)
        for _ in range(self.particles_number):
            beta += random.random() * 2.0 * max_weight
            while beta > self.weights[index]:
                beta -= self.weights[index]
                index = (index + 1) % self.particles_number
            new_particles_set.append(copy.deepcopy(self.particles_set[index]))
        self.particles_set = new_particles_set

