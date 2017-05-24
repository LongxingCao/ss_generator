import numpy as np

from . import geometry


def build_ideal_straight_alpha_helix(length):
    '''Build an ideal straight alpha helix.'''
    
    # Set the basic parameters

    c_n_length = 1.32869
    n_ca_length = 1.458
    ca_c_length = 1.52326
    c_o_length = 1.24
    c_n_ca_angle = np.radians(121.7)
    n_ca_c_angle = np.radians(111.2)
    ca_c_n_angle = np.radians(116.2)
    n_c_o_angle = np.radians(125)
    ca_c_o_angle = np.radians(121)
    phi = np.radians(-57)
    psi = np.radians(-47)
    ca_n_c_o_torsion = np.radians(0)
    omega = np.radians(180)

    # Build the first residue

    helix = [{'ca' : np.array([0, 0, 0]),
              'n' : n_ca_length * np.array([np.sin(n_ca_c_angle), np.cos(n_ca_c_angle), 0]),
              'c' : np.array([0, ca_c_length, 0])}]

    # Build the rest of residues

    for i in range(1, length):
        res = {}
        res['n'] = geometry.cartesian_coord_from_internal_coord(
                    helix[-1]['n'], helix[-1]['ca'], helix[-1]['c'],
                    c_n_length, ca_c_n_angle, psi)
        
        res['ca'] = geometry.cartesian_coord_from_internal_coord(
                    helix[-1]['ca'], helix[-1]['c'], res['n'],
                    n_ca_length, c_n_ca_angle, omega)

        res['c'] = geometry.cartesian_coord_from_internal_coord(
                    helix[-1]['c'], res['n'], res['ca'],
                    ca_c_length, n_ca_c_angle, phi)

        helix.append(res)

    # Add oxygen atoms

    for i in range(len(helix) - 1):
        helix[i]['o'] = geometry.cartesian_coord_from_internal_coord(
                        helix[i + 1]['ca'], helix[i + 1]['n'], helix[i]['c'],
                        c_o_length, n_c_o_angle, ca_n_c_o_torsion)

    # Add oxgen for the last residue

    helix[-1]['o'] = geometry.cartesian_coord_from_internal_coord(
                        helix[-1]['n'], helix[-1]['ca'], helix[-1]['c'],
                        c_o_length, ca_c_o_angle, np.radians(133))

    return helix[:length]


