import numpy as np


def normalize(v):
    '''Normalize a vector based on its 2 norm.'''
    if 0 == np.linalg.norm(v):
        return v
    return v / np.linalg.norm(v)

def create_frame_from_three_points(p1, p2, p3):
    '''Create a left-handed coordinate frame from 3 points. 
    The p2 is the origin; the y-axis is the vector from p2 to p3; 
    the z-axis is the cross product of the vector from p2 to p1
    and the y-axis.
    
    Return a matrix where the axis vectors are the rows.
    '''
    
    y = normalize(p3 - p2)
    z = normalize(np.cross(p1 - p2, y))
    x = np.cross(y, z)
    return np.array([x, y, z])

def rotation_matrix_to_axis_and_angle(M):
    '''Calculate the axis and angle of a rotation matrix.'''
    u = np.array([M[2][1] - M[1][2],
                  M[0][2] - M[2][0],
                  M[1][0] - M[0][1]])

    sin_theta = np.linalg.norm(u) / 2
    cos_theta = (np.trace(M) - 1) / 2

    return normalize(u), np.arctan2(sin_theta, cos_theta)

def rotation_matrix_from_axis_and_angle(u, theta):
    '''Calculate a rotation matrix from an axis and an angle.'''

    u = normalize(u)
    x = u[0]
    y = u[1]
    z = u[2]
    s = np.sin(theta)
    c = np.cos(theta)

    return np.array([[c + x**2 * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y**2 * (1 - c), y * z * (1 - c) - x * s ],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z**2 * (1 - c) ]])

def cartesian_coord_from_internal_coord(p1, p2, p3, d, theta, tau):
    '''Calculate the cartesian coordinates of an atom from 
    three internal coordinates and three reference points.
    '''
    axis1 = np.cross(p1 - p2, p3 - p2)
    axis2 = p3 - p2

    M1 = rotation_matrix_from_axis_and_angle(axis1, theta - np.pi)
    M2 = rotation_matrix_from_axis_and_angle(axis2, tau)

    return p3 + d * np.dot(M2, np.dot(M1, normalize(p3 - p2)))

def rotation_matrix_to_superimpose_two_vectors(v1, v2, theta=0):
    '''Get a rotation matrix that superimpose v1 to v2.
    Because there are infinite number of matrices that can do
    so, change the value of theta to get different results.
    '''
    v1 = normalize(v1)
    v2 = normalize(v2)

    axis = np.cross(v1, v2)
    sin_ang = np.linalg.norm(axis)
    cos_ang = np.dot(v1, v2)

    if np.linalg.norm(axis) < 0.01:
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    else:
        M = rotation_matrix_from_axis_and_angle(axis, np.arctan2(sin_ang, cos_ang))

    return np.dot(rotation_matrix_from_axis_and_angle(v2, theta), M)
