
import unittest
import sys
import numpy as np
import matplotlib.pyplot as plt

class PDelta:
	"""
	Class for implementing the p-delta learning rule in Auer et. al. 2002
	"""	
	def __init__(self, rho, eps, eta, gamma, mu):
		self.rho = rho
		self.eps = eps
		self.eta = eta
		self.gamma = gamma
		self.mu = mu

	def f(self, x):
		"""
		Map vector x to {-1, 1}
		"""
		fx = np.zeros(x.shape[0])
		for i in range(0, len(fx)):
			if (x[i] >= 0.0):
				fx[i] = 1.0
			else:
				fx[i] = -1.0
		return fx
	
	def squash(self, p):
		"""
		Squash integer p to [-1,1]
		"""
		# squash the vote (p) to [-1, 1]
		sp = 0.0
		if (p < -self.rho):
			sp = -1.0
		elif (-self.rho <= p) and (p <= self.rho):
			sp = float(p / self.rho)
		else:
			sp = 1.0
		return sp
	
	def vote(self, x):
		"""
		Determine population vote from array x {-1 ,1}
		"""
		return np.sum(x)

"""
Unit Tests
"""

class TestPDelta(unittest.TestCase):
	
	def setUp(self):
		eps = 0.1 # some error > 0
		rho = 1.0 / (2.0 * eps) # resolution of squashing function
		eta = 0.001 # the learning rate
		gamma =  0.1 # clear margin for dot product (can be set by learning algorithm)
		mu = 1.0 # importance of clear margin
		self.pdelta = PDelta(eps, rho, eta, gamma, mu)

	def test_f(self):	
		"""Test that f is correct"""
		x = np.array([0.1, -0.2, 0.0])
		y = np.array([1, -1, 1])
		self.assertEqual(self.pdelta.f(x).all(), y.all())

	def test_vote(self):	
		"""Test that vote is correct"""
		x = [-1, 1, -1]
		self.assertEqual(self.pdelta.vote(x), -1)

	def test_squash(self):	
		"""Test that squash is correct"""
		self.pdelta.rho = 2
		self.assertEqual(self.pdelta.squash(-3.0), -1)
		self.assertEqual(self.pdelta.squash(3.0), 1)
		self.assertEqual(self.pdelta.squash(1.0), 0.5)




if __name__ == '__main__':
	unittest.main()

