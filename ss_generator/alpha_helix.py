import numpy as np

from . import geometry


def theta_tau_to_rotation_matrix(theta, tau):
    '''Get the rotation matrix corresponding to the
    bond angle theta and dihedral tau.
    '''
    return np.dot(geometry.rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), tau),
                  geometry.rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), theta - np.pi))

def axis_to_theta_tau(axis):
    '''Get the bond angle theta and dihedral tau,
    from a rotation axis.
    '''
    theta = 2 * np.arctan(-axis[1] / axis[0])
    tau = 2 * np.arctan(axis[0] / axis[2])

    return theta, tau
