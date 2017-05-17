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

def build_beta_strand_from_dipeptide_directions(di_pp_directions):
    '''Build a beta strand given a list of dipeptide bond directions and 
    peptide to dipeptide bond transformations.
    '''
    # Build a flat strand

    strand = build_ideal_flat_beta_strand(len(di_pp_directions) * 2 + 1)

    # Set all the torsions to the peak of the beta sheet ramachanran distribution

    for i in range(len(strand)):
        basic.change_torsions(strand, i, np.radians(-120), np.radians(126))

    # Functions to create dipeptide frames

    def frame_for_dipp_direction(di_pp_id):
        x = geometry.normalize(di_pp_directions[di_pp_id][0])
        y = geometry.normalize(di_pp_directions[di_pp_id][1])
        z = np.cross(x, y)
        return np.array([x, y, z])

    def frame_for_current_dipp(res_id):
        dipp_direction = get_dipeptide_bond_direction(*strand[res_id - 1:res_id + 2])
        x = dipp_direction[0]
        y = dipp_direction[1]
        z = np.cross(x, y)
        return np.array([x, y, z])

    def transform_matrix(di_pp_id, res_id):
        '''Get matrix that transforms a dipeptide bond to a
        given direction.
        '''
        return np.dot(np.transpose(frame_for_dipp_direction(di_pp_id)), 
            frame_for_current_dipp(res_id))

    # Align the strand to the first di_pp_direction

    strand = basic.transform_residue_list(strand, transform_matrix(0, 1), np.zeros(3)) 

    # Set the torsions for the residues

    for i in range(1, len(di_pp_directions)):
        res_id = 2 * i + 1
        
        # Get the Euclidean transformation

        M = transform_matrix(i, res_id)
        t = strand[res_id - 1]['ca'] - np.dot(M, strand[res_id - 1]['ca'])
        
        # Transform the rest of the strand

        for atom in strand[res_id - 1].keys():
            if atom not in ['h', 'n']:
                strand[res_id - 1][atom] = np.dot(M, strand[res_id - 1][atom]) + t

        for j in range(res_id, len(strand)):
            strand[j] = basic.transform_residue(strand[j], M, t)

    return strand
