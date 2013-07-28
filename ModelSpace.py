"""
Classes and methods for representing neurons in some spatial configuration

Author: Sina Masoud-Ansari
"""

import math
import unittest

class Point3D:
	"""A point in 3D Euclidian space"""
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z
	
	def __eq__(self, other):
		return	(	isinstance(other, self.__class__)
					and self.__dict__ == other.__dict__	)

	def __ne__(self, other):
		return not self.__eq__(other)	

class Cuboid3D:
	"""Class for 3D Euclidean space"""
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	def pointFrom1D(self, p):
		"""Find the 3D coordinates of a 1D point"""
		x = p % self.x
		y = (p / self.x) % self.y
		z = p / (self.x * self.y)
		return Point3D(x, y, z)

	def distance (self, i, j):
		"""Find the distance between two 3D points"""
		pi = self.pointFrom1D(i)
		pj = self.pointFrom1D(j)
		dx = pi.x - pj.x
		dy = pi.y - pj.y
		dz = pi.z - pj.z
		return math.sqrt(dx*dx + dy*dy + dz*dz)

"""
Unit Tests
"""


class TestPoint3D(unittest.TestCase):

	def test_eq(self):
		"""Test equality between Point3D objects"""
		self.assertEquals(Point3D(1,2,3), Point3D(1,2,3))
		self.assertNotEquals(Point3D(0,1,2), Point3D(1,2,3))
	

class TestSpace3D(unittest.TestCase):

	def setUp(self):
		self.space = Cuboid3D(2,3,2)
	
	def test_pointFrom1D(self):	
		"""Test that 1D point maps correctly to a 3D point in the space"""
		self.assertEqual(self.space.pointFrom1D(14), Point3D(0,1,2))

	def test_distance(self):	
		"""Test that distance between two points is correct"""
		self.assertEqual(self.space.distance(10, 3), math.sqrt(3))

if __name__ == '__main__':
	unittest.main()
