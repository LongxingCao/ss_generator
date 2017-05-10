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
