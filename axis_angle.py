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


Module to calculate the axis angle between two orthonormal bases

The axis angle representation of a rotation, also known as the exponential 
coordinates of a rotation, parameterizes a rotation by two values: a unit 
vector indicating the direction of a directed axis (straight line), and an 
angle describing the magnitude of the rotation about the axis. The rotation 
occurs in the sense prescribed by the right hand grip rule.
"""

from numpy import linalg
from math import acos, pi, fabs

TOLERANCE = 1e-8

class AxisAngleException(Exception):
    pass

###################################
# HELPERS
###################################

def _index_of_unitary_eigenvalue(eigenvalues):
    """
    From a numpy array of eigenvalues 
    returns the index of the 
    eigenvalue lamba = 1
    """
    idx = None
    r_eq_1 = list(abs(eigenvalues.real-1.0) < TOLERANCE)
    try:
        idx = r_eq_1.index(True)
    except ValueError:
        raise AxisAngleException("Unitary eigenvalue not found")
    return idx

def _rotation_matrix(axis1, axis2):
    """
    Calculate the rotation matrix
    M2 = R * M1
    R = M^-1 * M2
    returns a rotation matrix
    """
    rotation_matrix = axis2 * linalg.inv(axis1)
    determinant = float(linalg.det(rotation_matrix))
    if fabs((1.0-determinant))<TOLERANCE:
        AxisAngleException("Candidate rotation matrix has det != 1") 
    return rotation_matrix

############################################
# AXIS ANGLE
############################################


def axis_angle(axis1, axis2):
    """
    From two matrices
    of orthonormal basis of form:-
    [[ix jx kx]
    [iy jy ky]
    [iz jz kz]]
    return axis, angle
    """
    eigenvalues, eigenvectors = linalg.eig(_rotation_matrix(axis1, axis2))
    unitary_ev_idx = _index_of_unitary_eigenvalue(eigenvalues)
    #eigenvectors with vector v[:,i] corresponds to eigenvalue u[i]
    axis = eigenvectors[:,unitary_ev_idx] 
    axis = tuple([dir_vec[0,0] for dir_vec in axis.real])
    eigenvalues = list(eigenvalues)
    eigenvalues.pop(unitary_ev_idx)
    angle = acos((eigenvalues[0]+ eigenvalues[1]) / 2.0)/pi * 180.0
    return axis, angle
