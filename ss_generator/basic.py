import numpy as np

from . import geometry


def transform_residue(residue, M, t):
    '''Transform a residue by a rotation M and a
    translation t. Return the transformed residue.
    '''
    new_res = {}

    for key in residue.keys():
        new_res[key] = np.dot(M, residue[key]) + t

    return new_res

def transform_residue_list(res_list, M, t):
    '''Transform a residue list by a rotation M and a
    translation t. Return the new list.
    '''
    return [transform_residue(res, M, t) for res in res_list]

def change_torsions(strand, res_id, phi, psi):
    '''Change the phi, psi angles of a residue in
    a strand.
    '''

    # Rotate the psi torsion

    if res_id < len(strand) - 1:
    
        psi_old = geometry.dihedral(strand[res_id]['n'], strand[res_id]['ca'], 
                strand[res_id]['c'], strand[res_id + 1]['n'])

        # Get the rotation matrix

        axis = strand[res_id]['c'] - strand[res_id]['ca']
        M = geometry.rotation_matrix_from_axis_and_angle(axis, psi - psi_old)
        t = strand[res_id]['ca'] - np.dot(M, strand[res_id]['ca'])

        # Rotate subsequent atoms

        strand[res_id]['o'] = np.dot(M, strand[res_id]['o']) + t

        for i in range(res_id + 1, len(strand)):
            strand[i] = transform_residue(strand[i], M, t)

    # Rotate the phi torsion
    
    if res_id > 0:

        phi_old = geometry.dihedral(strand[res_id - 1]['c'], strand[res_id]['n'],
                strand[res_id]['ca'], strand[res_id]['c'])

        # Get the rotation matrix

        axis = strand[res_id]['ca'] - strand[res_id]['n']
        M = geometry.rotation_matrix_from_axis_and_angle(axis, phi - phi_old)
        t = strand[res_id]['ca'] - np.dot(M, strand[res_id]['ca'])

        # Rotate subsequent residues

        for i in range(res_id, len(strand)):
            strand[i] = transform_residue(strand[i], M, t)
