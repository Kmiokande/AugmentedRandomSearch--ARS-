import os
import numpy as np


class Hp():
	def __init__(self):
		self.nbSteps = 1000
		self.episodeLenght = 100
		self.learningRate = 0.02
		self.nbDirections = 16
		self.nbBestDirections = 16
		assert self.nbBestDirections <= self.nbDirections
		self.noise = 0.03
		self.seed = 1
		self.envName = ''
