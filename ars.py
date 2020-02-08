import os
import numpy as np


class Hp():
	def __init__(self):
		self.nb_steps = 1000
		self.episode_lenght = 100
		self.learning_rate = 0.02
		self.nb_directions = 16
		self.nb_best_directions = 16
		assert self.nb_best_directions <= self.nb_directions
		self.noise = 0.03
		self.seed = 1
		self.env_name = ''
