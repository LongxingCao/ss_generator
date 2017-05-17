import json

import numpy as np

from . import geometry
from . import basic


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
    h_f = np.array([2.392, -0.969, 0])
    ca2_f = np.array([3.755, 0.582, 0])

    # Get the coordinates of the first residue

    Mx = geometry.rotation_matrix_from_axis_and_angle(
            np.array([1, 0, 0]), np.pi)

    My = geometry.rotation_matrix_from_axis_and_angle(
            np.array([0, 1, 0]), -(np.pi - eta) / 2)

    res1 = {'n' : np.dot(np.transpose(My), np.dot(Mx, n_f - ca2_f)),
            'h' : np.dot(np.transpose(My), np.dot(Mx, h_f - ca2_f)),
            'ca' : ca1_f, 'c' : np.dot(My, c_f), 'o' : np.dot(My, o_f)}

    # Get the coordinates of the second residue

    t = np.dot(My, ca2_f)
    res2 = basic.transform_residue(res1, Mx, t) 

    # Get the coordinates of the strand
    
    shift = 2 * np.array([np.sin(eta / 2) * ca2_f[0], 0, 0])
    strand =[res1, res2]
    
    for i in range(2, length):
        strand.append(basic.transform_residue(strand[i - 2], np.identity(3), shift))

    return strand[:length]

def build_ideal_flat_beta_sheet(sheet_type, length, num_strand):
    '''Build an ideal flat beta sheet.
    The sheet plane is the x, y plane and the direction 
    of strands is the x axis.
    '''
    sheet = []

    if sheet_type == 'parallel':
        shift = np.array([0, 4.84, 0])
        sheet.append(build_ideal_flat_beta_strand(length))
        
        for i in range(1, num_strand):
            sheet.append(basic.transform_residue_list(
                sheet[-1], np.identity(3), shift))
   
    elif sheet_type == 'antiparallel':
        if length < 2 and num_strand > 2:
            raise Exception("Invalid parameters for antiparallel beta sheet!")
        
        shift = np.array([0, 5.24, 0])

        strand1 = build_ideal_flat_beta_strand(length)
        
        # Get the second strand

        Mx = geometry.rotation_matrix_from_axis_and_angle(
                np.array([1, 0, 0]), np.pi)
        Mz = geometry.rotation_matrix_from_axis_and_angle(
                np.array([0, 0, 1]), np.pi)

        M = Mz if length % 2 == 1 else np.dot(Mx, Mz)
        t = shift + strand1[0]['ca'] \
            - basic.transform_residue(strand1[-1], M, np.zeros(3))['ca']

        strand2 = basic.transform_residue_list(strand1, M, t)

        # Get the rest of the sheet
    
        shift2 = shift + strand2[-2]['ca'] - strand1[1]['ca']

        sheet = [strand1, strand2]
        for i in range(2, num_strand):
            sheet.append(basic.transform_residue_list(
                sheet[-2], np.identity(3), shift2))

    return sheet[:num_strand]

def calc_n_ca_c_angle_between_peptide_plan(eta, epsilon_n, epsilon_c):
    '''Calculate the n_ca_c angle given the angle eta between two peptide plane,
    epsilon_n the residual of the angle between the ca_n vector and the intersection line and
    epsilon_c the residual of the angle between the ca_c vector and the intersection line.
    '''
    v_n = np.array([np.cos(epsilon_n), 0, np.sin(epsilon_n)])
    v_c = np.array([np.cos(epsilon_c) * np.cos(eta), np.cos(epsilon_c) * np.sin(eta),
                    np.sin(epsilon_c)])

    return np.arccos(np.dot(v_n, v_c))

def get_dipeptide_bond_direction(res1, res2, res3):
    '''Caculate the direction of the dipeptide bond formed by 3 residues.
    Return a tuple (dipeptide_bond_direction, hb_direction).
    '''
    dipeptide_bond_direction = geometry.normalize(res3['ca'] - res1['ca'])

    # Get the intersection between the two peptide bond plane

    n1 = np.cross(res1['c'] - res2['n'], res2['ca'] - res2['n'])
    n2 = np.cross(res2['ca'] - res2['c'], res3['n'] - res2['c'])
    
    intersection = np.cross(n1, n2)
    if np.dot(intersection, res2['c'] - res2['ca']) < 0:
        intersection = -intersection

    # Get the hb_direction

    hb_direction = geometry.normalize(intersection 
                    - np.dot(intersection, dipeptide_bond_direction) * dipeptide_bond_direction)

    return (dipeptide_bond_direction, hb_direction)

def get_expected_dipeptide_transformation(res1, res2, dipeptide_bond_direction, hb_direction):
    '''Get the peptide bond to dipeptide bond transformation given the two
    residues that forms a peptide bond and the direction of the dipeptide
    bond.
    Return the transformation matrix in the peptide frame.
    '''

    pp_frame = geometry.create_frame_from_three_points(res1['c'], res2['n'], res2['ca'])
    dipp_frame = np.array([dipeptide_bond_direction, hb_direction,
                            np.cross(dipeptide_bond_direction, hb_direction)])

    return np.dot(pp_frame, np.transpose(dipp_frame))

def load_peptide_dipeptide_transformations(data_file):
    '''Load peptide to dipeptide transformations from a json file.'''
    with open(data_file, 'r') as f:
        transformations = json.loads(f.read())
        for d in transformations:
            d['M'] = np.array(d['M'])
        return transformations

def search_peptide_dipeptide_transformations(transformations, expected_transformation):
    '''Find the peptide to dipeptide transformation that match the expected_transformation
    best. Return the matched transformation and the norm of the difference matrix.
    '''
    best_match = None
    best_diff = 100

    for t in transformations:
        diff = np.linalg.norm(expected_transformation - t['M'])

        if diff < best_diff:
            best_match = t
            best_diff = diff

    return best_match, best_diff

def build_beta_strand_from_dipeptide_directions(di_pp_directions, pp_di_pp_transformations):
    '''Build a beta strand given a list of dipeptide bond directions and 
    peptide to dipeptide bond transformations.
    '''
    # Build a flat strand

    strand = build_ideal_flat_beta_strand(len(di_pp_directions) * 2 + 1)

    # Align the strand to the first di_pp_direction
    
    x = geometry.normalize(di_pp_directions[0][0])
    y = -geometry.normalize(di_pp_directions[0][1])
    z = np.cross(x, y) 
    M = np.transpose(np.array([x, y, z]))

    strand = basic.transform_residue_list(strand, M, np.zeros(3)) 

    # Set the torsions for the residues

    for i in range(1, len(di_pp_directions)):
        res_id = 2 * i

        # Find the best tranformation

        expected_transformation = get_expected_dipeptide_transformation(
                *strand[res_id - 1:res_id + 1], *di_pp_directions[i])

        match, best_diff = search_peptide_dipeptide_transformations(
                pp_di_pp_transformations, expected_transformation)

        print(best_diff) ###DEBUG

        # Change the torsions

        basic.change_torsions(strand, res_id, match['phi1'], match['psi1'])
        basic.change_torsions(strand, res_id + 1, match['phi2'], match['psi2'])

    return strand
