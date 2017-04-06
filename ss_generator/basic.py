import numpy as np

from . import geometry


def get_internal_coordinates_from_ca_list(ca_list):
    '''Get the list of ds, thetas and taus from a ca list.'''
    ds = []
    thetas = []
    taus = []

    for i in range(len(ca_list) - 1):
        ds.append(np.linalg.norm(ca_list[i + 1] - ca_list[i]))

    for i in range(1, len(ca_list) - 1):
        thetas.append(geometry.angle(ca_list[i - 1] - ca_list[i],
            ca_list[i + 1] - ca_list[i]))

    for i in range(1, len(ca_list) - 2):
        taus.append(geometry.dihedral(ca_list[i - 1], ca_list[i],
            ca_list[i + 1], ca_list[i + 2]))

    return ds, thetas, taus

def generate_segment_from_internal_coordinates(ds, thetas, taus):
   '''Generate a protein segment from a set of internal coordinates.
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

