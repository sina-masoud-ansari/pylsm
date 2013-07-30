"""
Classes and methods for representing neurons in some spatial configuration

Author: Sina Masoud-Ansari

Classes:
	Point3D
	Cuboid3D
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

def distance3D(i, j):
	"""Distance between two 3D euclidean points"""
	dx = i.x - j.x
	dy = i.y - j.y
	dz = i.z - j.z
	return math.sqrt(dx*dx + dy*dy + dz*dz)


class Cuboid3D:
	"""Class for 3D Euclidean space"""
	def __init__(self, x, y, z, offset=Point3D(0,0,0)):
		self.x = x
		self.y = y
		self.z = z
		self.size = x * y * z
		self.offset = offset

	def pointFrom1D(self, i):
		"""Find the 3D coordinates of a 1D point"""
		x = i % self.x
		y = (i / self.x) % self.y
		z = i / (self.x * self.y)
		ox = self.offset.x
		oy = self.offset.y
		oz = self.offset.z
		return Point3D(x+ox, y+oy, z+oz)
		

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

	def test_size(self):	
		"""Test that size of space is correct"""
		self.assertEqual(self.space.size, 12)
	
	def test_pointFrom1D(self):	
		"""Test that 1D point maps correctly to a 3D point in the space"""
		self.assertEqual(self.space.pointFrom1D(14), Point3D(0,1,2))

	def test_distance3D(self):	
		"""Test that distance between two points is correct"""
		i = self.space.pointFrom1D(10)
		j = self.space.pointFrom1D(3)
		self.assertEqual(distance3D(i, j), math.sqrt(3))

	def test_distance3DWithOffset(self):	
		"""Test that distance between two points in two model spaces is correct"""
		offset = Point3D(5,0,0) 
		adjacent = Cuboid3D(2,3,2, offset=offset)
		i = self.space.pointFrom1D(10)
		j = adjacent.pointFrom1D(3)
		self.assertEqual(distance3D(i, j), math.sqrt(38))

if __name__ == '__main__':
	unittest.main()
