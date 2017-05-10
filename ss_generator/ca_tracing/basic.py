import numpy as np

from ss_generator import geometry


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

def get_peptide_bond_parameters():
    '''Print peptide parameters.'''
    d = {'c_n_length' : 1.32869,
         'n_ca_length' : 1.458,
         'ca_c_length' : 1.52326,
         'c_n_ca_angle' : np.radians(121.7),
         'n_ca_c_angle' : np.radians(111.2),
         'ca_c_n_angle' : np.radians(116.2),
         'omega' : np.radians(180)}

    p1 = np.array([0, 0, 0])
    p2 = np.array([0, 0, d['ca_c_length']])
    p3 = p2 + d['c_n_length'] * np.array([
        np.sin(d['ca_c_n_angle']), 0, -np.cos(d['ca_c_n_angle'])])
    p4 = geometry.cartesian_coord_from_internal_coord(
            p1, p2, p3, d['n_ca_length'], d['n_ca_c_angle'], d['omega'])

    d['theta_c'] = geometry.angle(p4 - p1, p2 - p1)
    d['theta_n'] = geometry.angle(p1 - p4, p3 - p4)

    return d

def get_n_for_pp_bond_forward(ca1, ca2, v_c):
    '''Get the coordinate of the N atom in a peptide bond.
    Inputs are the two ends of the peptide bond and the 
    direction from ca1 to the position of C.
    '''
    params = get_peptide_bond_parameters()

    x = geometry.normalize(ca1 - ca2)
    y = -geometry.normalize(v_c - np.dot(v_c, x) * x)

    return ca2 + params['n_ca_length'] * (np.cos(params['theta_n']) * x \
            + np.sin(params['theta_n']) * y)

def get_c_for_pp_bond_forward(ca1, ca2, v_n, z_sign=1):
    '''Get the coordinate of the C atom in a peptide bond.
    Inputs are the two ends of the peptide bond, the direction
    from ca1 to the position of the previous N and the sign
    of Z direction that is used to pick one solution from two.
    '''
    params = get_peptide_bond_parameters()

    frame = geometry.create_frame_from_three_points(ca1 + v_n, ca1, ca2)
    beta = geometry.angle(v_n, ca2 - ca1)
    
    gamma = z_sign * np.arccos((np.cos(params['n_ca_c_angle']) - np.cos(params['theta_c']) * np.cos(beta)) \
            / (np.sin(params['theta_c']) * np.sin(beta)))

    c_local = params['ca_c_length'] * np.array([np.sin(params['theta_c']) * np.cos(gamma),
        np.cos(params['theta_c']), np.sin(params['theta_c']) * np.sin(gamma)])

    return ca1 + np.dot(np.transpose(frame), c_local)

def get_o_for_peptide_bond(c, n, ca2):
    '''Get the coordinate of the O atom in a peptide bond.'''
    return geometry.cartesian_coord_from_internal_coord(ca2,
            n, c, 1.24, np.radians(125), 0)

def thread_ca_list_forward(ca_list, initial_c_direction, z_sign=1):
    '''Thread backbones through a ca list. Return a list
    for residue dictionaries.
    '''
    params = get_peptide_bond_parameters()

    # Make the initial residue

    residue_list = [{'ca' : ca_list[0], 
        'c' : ca_list[0] + params['ca_c_length'] * geometry.normalize(initial_c_direction)}]

    # Make the rest of residues

    for i in range(1, len(ca_list)):
        residue = {'ca' : ca_list[i]}
        v_c = residue_list[i - 1]['c'] - residue_list[i - 1]['ca']
        residue['n'] = get_n_for_pp_bond_forward(ca_list[i - 1], ca_list[i], v_c)

        if i < len(ca_list) - 1:
            residue['c'] = get_c_for_pp_bond_forward(ca_list[i], ca_list[i + 1], 
                    residue['n'] - residue['ca'], z_sign=z_sign)

        residue['o'] = get_o_for_peptide_bond(residue_list[i - 1]['c'],
                residue['n'], residue['ca'])

        residue_list.append(residue)

    return residue_list


