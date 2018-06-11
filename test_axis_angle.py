"""
Axis Angle
Copyright (C) 2010  sarasola

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Unit test for axis angle module
"""

import unittest 

from math import cos, sin, pi
from numpy import matrix, array, cross, linalg, complex

from axis_angle import axis_angle, _index_of_unitary_eigenvalue, _rotation_matrix, TOLERANCE, AxisAngleException 

def normalize(vector):
    """
    Normalise a vector
    vector can be matrix, or array
    """
    norm = linalg.linalg.norm(vector)
    if isinstance(vector, matrix):
        vector = vector / norm
    else:
        vector = array(vector)
        vector = vector / norm
        vector = tuple(vector)
    return vector

#####################################################
# Rotation Matrices
#####################################################

def vector_rotation_matrix(axis, phi):
    """Rotation matrix about arbitrary vector"""
    axis_n = normalize( axis )
    u_1 = axis_n[0, 0]
    u_2 = axis_n[0, 1]
    u_3 = axis_n[0, 2]
    cos_phi = cos( phi )
    sin_phi = sin( phi )
    return matrix([[ cos_phi + ((1 - cos_phi) * u_1 * u_1), 
                    (1 - cos_phi) * u_1 * u_2 - u_3 * sin_phi, 
                    (1 - cos_phi) * u_1 * u_3 + u_2 * sin_phi ],
                   [ (1 - cos_phi) * u_1 * u_2 + u_3 * sin_phi, 
                    cos_phi + (1 - cos_phi) * u_2 * u_2, 
                    (1 - cos_phi) * u_2 * u_3 - u_1 * sin_phi ],
                   [ (1 - cos_phi) * u_1 * u_3 - u_2 * sin_phi, 
                        (1 - cos_phi) * u_2 * u_3 + u_1 * sin_phi, 
                        cos_phi + (1 - cos_phi) * u_3 * u_3 ]])
 
def x_rotation_matrix(phi):
    """x Rotation Matrix for angle phi in radians"""
    c = cos(phi)
    s = sin(phi)
    return matrix(((1.0, 0.0, 0.0),(0.0, c, -s), (0.0, s, c)))

def y_rotation_matrix(phi):
    """y Rotation Matrix for angle phi in radians"""
    c = cos(phi)
    s = sin(phi)
    return matrix(((c, 0.0, s),(0.0, 1, 0), (-s, 0, c)))

def z_rotation_matrix(phi):
    """z Rotation Matrix for angle phi in radians"""
    c = cos(phi)
    s = sin(phi)
    return matrix(((c, -s, 0),(s, c, 0), (0, 0, 1)))

####################################################################
# TEST
####################################################################


class TestAxisAngle(unittest.TestCase):

    def test_orthonormal_basis_rotate_x(self):
        """
        Test for two axes that are related by a rotation about x
        """
        axis1 = matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        phi = 10.0/ 180.0 *pi
        axis2 = axis1 * x_rotation_matrix(phi)
        ax, ang = axis_angle(axis1, axis2)
        self.assertEquals((1, 0, 0), ax)
        self.assertAlmostEquals(10, ang)
        rot_m = vector_rotation_matrix(matrix(ax), ang/180.0*pi)
        self.assertTrue((abs(rot_m*axis1-axis2)<TOLERANCE).all())

    def test_orthonormal_basis_rotate_y(self):
        """
        Test for two axes that are related by a rotation about y
        """
        axis1 = matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        phi = 35.0/ 180.0 *pi
        axis2 = axis1 * y_rotation_matrix(phi)
        ax, ang = axis_angle(axis1, axis2)
        self.assertEquals((0, 1, 0), ax)
        self.assertAlmostEquals(35, ang)
        rot_m = vector_rotation_matrix(matrix(ax), ang/180.0*pi)
        self.assertTrue((abs(rot_m*axis1-axis2)<TOLERANCE).all())

    def test_orthonormal_basis_rotate_z(self):
        """
        Test for two axes that are related by a rotation about z
        """
        axis1 = matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        a = 35.0/ 180.0 *pi        
        axis2 = axis1 * z_rotation_matrix(a)
        ax, ang = axis_angle(axis1, axis2)
        self.assertEquals((0, 0, 1), ax)
        self.assertAlmostEquals(35, ang)
        rot_m = vector_rotation_matrix(matrix(ax), ang/180.0*pi)
        self.assertTrue((abs(rot_m*axis1-axis2)<TOLERANCE).all())

    def test_orthonormal_basis_rotate_arbitrary_axis(self):
        """
        Test for two axes that are related by a rotation about an arbitrary axis
        """
        x = (1.0, 2.0, -3.0)
        y = (-4.0, 8.0, 4.0) 
        z = cross(x, y)
        axis1 = matrix([normalize(x), normalize(y), normalize(z)])
        a = 10.0/ 180.0 *pi
        b = 35.0/ 180.0 *pi
        axis2 = x_rotation_matrix(a)*y_rotation_matrix(b)*axis1
        ax, ang = axis_angle(axis1, axis2)
        rot_m = vector_rotation_matrix(matrix(ax), ang/180.0*pi)*axis1
        self.assertTrue((abs(rot_m-axis2)<TOLERANCE).all())
        
    def test_index_of_unitary_eigenvalue(self):
        no_unitary_ev = array([complex(0.5, 0.5), complex(0.5, 0.5), complex(0.5, 0.5)])
        self.assertRaises(AxisAngleException, _index_of_unitary_eigenvalue, no_unitary_ev)
        e_values = array([complex(0.5, 0.5), complex(1.0, 0.5), complex(0.5, 0.5)])
        self.assertEquals(1, _index_of_unitary_eigenvalue(e_values))
        

if __name__ == "__main__":
    unittest.main()
