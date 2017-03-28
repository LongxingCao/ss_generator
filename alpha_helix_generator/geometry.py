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
                     [z * x * (1 - c) + y * s, z * y * (1 - c) + x * s, c + z**2 * (1 - c) ]])
