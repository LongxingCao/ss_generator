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

def generate_alpha_helix_from_internal_coordinates(ds, thetas, taus):
   '''Generate an alpha helix from a set of internal coordinates.
   Return a list of Ca coordinates.
   '''
   # Make sure that the sizes of internal coordinates are correct

   if len(ds) < 3 or len(thetas) < 2 or len(taus) < 1 \
      or len(ds) != len(thetas) + 1 or len(ds) != len(taus) + 2:
          raise Exception("Incompatible sizes of internal coordinates.")

   # Make the first three Ca atoms

   ca_list = []
   ca_list.append(ds[0] * np.array([np.sin(thetas[0]),np.cos(thetas[0]), 0]))
   ca_list.append(np.array([0, 0, 0]))
   ca_list.append(np.array([0, ds[1], 0]))

   # Make the rest of Ca atoms

   for i in range(len(taus)):
      ca_list.append(geometry.cartesian_coord_from_internal_coord(
          ca_list[i], ca_list[i + 1], ca_list[i + 2], ds[i + 2], thetas[i + 1], taus[i]))

   return ca_list
