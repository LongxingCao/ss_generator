import numpy as np

from . import geometry


def build_ideal_flat_beta_strand(length):
    '''Build an ideal flat beta strand.
    The sheet plane is the x, y plane and the direction 
    of the strand is the x axis.
    '''
    eta = np.radians(134.3)

    # Set the 2D peptide bond coordinates

    ca1_f = np.array([0, 0, 0])
    c_f = np.array([1.311, 0.788, 0])
    o_f = np.array([1.311, 2.028, 0])
    n_f = np.array([2.392, 0.031, 0])
    ca2_f = np.array([3.755, 0.582, 0])

    # Get the coordinates of the first residue

    Mx = geometry.rotation_matrix_from_axis_and_angle(
            np.array([1, 0, 0]), np.pi)

    My = geometry.rotation_matrix_from_axis_and_angle(
            np.array([0, 1, 0]), -(np.pi - eta) / 2)

    res1 = {'n' : np.dot(np.transpose(My), np.dot(Mx, n_f - ca2_f)),
            'ca' : ca1_f, 'c' : np.dot(My, c_f), 'o' : np.dot(My, o_f)}

    # Get the coordinates of the second residue

    t = np.dot(My, ca2_f)
    res2 = {}

    for key in res1.keys():
        res2[key] = np.dot(Mx, res1[key]) + t

    # Get the coordinates of the strand
    
    shift = 2 * np.array([np.sin(eta / 2) * ca2_f[0], 0, 0])
    strand =[res1, res2]
    
    for i in range(2, length):
        
        res = {}
        for key in res1.keys():
            res[key] = strand[i - 2][key] + shift

        strand.append(res)

    return strand[:length]

def build_ideal_flat_beta_sheet(length, num_strand):
    '''Build an ideal flat beta sheet.
    The sheet plane is the x, y plane and the direction 
    of strands is the x axis.
    '''
    return build_ideal_flat_beta_strand(length)
