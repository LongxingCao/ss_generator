import json

import numpy as np
import scipy.optimize

from . import geometry
from . import basic
from . import ramachandran


# Load the beta sheet ramachandran distribution

beta_sheet_ramachandran = None

with open('../databases/beta_sheet_ramachandran.json', 'r') as f:
    beta_sheet_ramachandran = json.loads(f.read())

cumulative_beta_sheet_ramachandran = ramachandran.get_cumulative_probability(
                                            beta_sheet_ramachandran)

def random_beta_torsion():
    '''Return a random pair of phi/psi torsions for a beta strand.'''
    return ramachandran.random_torsions(beta_sheet_ramachandran,
                                cumulative_beta_sheet_ramachandran)

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

def frame_for_dipp(res1, res2, res3):
    '''Return a frame for a dipeptide bond defined by 3 residues.'''
    dipp_direction = get_dipeptide_bond_direction(res1, res2, res3)
    x = dipp_direction[0]
    y = dipp_direction[1]
    z = np.cross(x, y)
    return np.array([x, y, z])

def build_beta_strand_from_dipeptide_directions(di_pp_directions, relax=True):
    '''Build a beta strand given a list of dipeptide bond directions and 
    peptide to dipeptide bond transformations.
    '''
    # Build a flat strand

    strand = build_ideal_flat_beta_strand(len(di_pp_directions) * 2 + 1)

    # Set all the torsions to the peak of the beta sheet ramachandran distribution

    for i in range(len(strand)):
        basic.change_torsions(strand, i, np.radians(-120), np.radians(126))

    # Functions to create dipeptide frames

    def frame_for_dipp_direction(di_pp_id):
        x = geometry.normalize(di_pp_directions[di_pp_id][0])
        y = geometry.normalize(di_pp_directions[di_pp_id][1])
        z = np.cross(x, y)
        return np.array([x, y, z])

    def transform_matrix(di_pp_id, res_id):
        '''Get matrix that transforms a dipeptide bond to a
        given direction.
        '''
        return np.dot(np.transpose(frame_for_dipp_direction(di_pp_id)), 
                frame_for_dipp(*strand[res_id - 1:res_id + 2]))

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

    # Relax the strand before returning it

    if relax:
        relax_bond_angles(strand)

    return strand

def relax_bond_angles(strand, num_positions=10, num_trials=10):
    '''Relax the bond angles in a strand built from the 
    build_beta_strand_from_dipeptide_directions() function.
    Do the relaxation by minimizing a strain function.
    '''
    def strain_function(angle1, angle2):
        ideal_angle = np.radians(111.2)
        return (angle1 - ideal_angle) ** 2 + (angle2 - ideal_angle) ** 2

    def get_angle(res_id):
        '''Get the angle for a given residue.'''
        return geometry.angle(strand[res_id]['n'] - strand[res_id]['ca'],
                              strand[res_id]['c'] - strand[res_id]['ca'])

    def get_strain(position):
        '''Get the strain at a given position.'''
        angle1 = get_angle(2 * position) 
        angle2 = get_angle(2 *position + 2)
        return strain_function(angle1, angle2)

    # Make a model dipeptide to try different torsions
    
    model_dipp = build_ideal_flat_beta_strand(3)
    
    def relax_position(position):
        '''Relax the dipeptide bond at a given position.'''
        # Get the dipeptide frame

        frame = frame_for_dipp(*strand[2 * position : 2 * position + 3])

        # Get the two flanking bonds

        ca_n1 = strand[2 * position]['n'] - strand[2 * position]['ca']
        ca_c2 = strand[2 * position + 2]['c'] - strand[2 * position + 2]['ca']

        # Sample torsions

        best_torsions = (basic.get_phi(strand, 2 * position + 1),
                         basic.get_psi(strand, 2 * position + 1))
        best_strain = get_strain(position)

        for i in range(num_trials):

            # Change the model dipeptide to a random torsion

            torsions = random_beta_torsion()

            basic.change_torsions(model_dipp, 1, *torsions)

            # Get the frame of the model dipeptide and frame tranformation matrix

            model_frame = frame_for_dipp(*model_dipp)
            M = np.dot(np.transpose(frame), model_frame)

            # Get the vectors for bond angle calculation

            ca_c1 = np.dot(M, model_dipp[0]['c'] - model_dipp[0]['ca'])
            ca_n2 = np.dot(M, model_dipp[2]['n'] - model_dipp[2]['ca'])

            # Update the best torsions

            strain = strain_function(geometry.angle(ca_n1, ca_c1), geometry.angle(ca_n2, ca_c2))

            if strain < best_strain:
                best_strain = strain
                best_torsions = torsions

        # Apply the best torsions

        basic.change_torsions(model_dipp, 1, *best_torsions)
        
        M = np.dot(np.transpose(frame), frame_for_dipp(*model_dipp))
        t1 = strand[2 * position]['ca'] - np.dot(M, model_dipp[0]['ca'])
        t2 = np.dot(M, model_dipp[2]['ca']) + t1 - strand[2 * position + 2]['ca']

        for atom in strand[2 * position].keys():
            if position == 0 or (atom not in ['h', 'n']):
                strand[2 * position][atom] = np.dot(M, model_dipp[0][atom]) + t1

        strand[2 * position + 1] = basic.transform_residue(model_dipp[1], M, t1)

        for atom in strand[2 * position + 2].keys():
            if 2 * position + 3 == len(strand) or (atom not in ['c', 'o']):
                strand[2 * position + 2][atom] = np.dot(M, model_dipp[2][atom]) + t1
            else:
                strand[2 * position + 2][atom] += t2

        for i in range(2 * position + 3, len(strand)):
            strand[i] = basic.transform_residue(strand[i], np.identity(3), t2)

    # Pick positions randomly and relax them

    random_positions = np.random.randint(len(strand) // 2, size=num_positions)

    #strains = [get_strain(i) for i in range(len(strand) // 2)] ###DEBUG
    #print("Strains before relaxation:\n", strains) ###DEBUG
    #print("Total strain before relaxation:\n", sum(strains)) ###DEBUG
    #angles = [np.degrees(get_angle(i)) for i in range(len(strand))]###DEBUG
    #print("Angles before relaxation:\n", angles)###DEBUG

    for p in random_positions:
        relax_position(p)

    #strains = [get_strain(i) for i in range(len(strand) // 2)] ###DEBUG
    #print("Strains after relaxation:\n", strains) ###DEBUG
    #print("Total strain after relaxation:\n", sum(strains)) ###DEBUG
    #angles = [np.degrees(get_angle(i)) for i in range(len(strand))]###DEBUG
    #print("Angles after relaxation:\n", angles)###DEBUG

def build_beta_barrel(sheet_type, num_strand, strand_length, pitch_angle):
    '''Build a beta barrel.
    The pitch_angle should be in the range of [-pi / 2, pi / 2]. Strands
    are right handed if pitch_angle > 0.
    '''
    # Set some basic parameters
    
    DI_PEPTIDE_LENGTH = 6.7
    PARALLEL_D_INTER = 4.84
    ANTIPARALLEL_D_INTER_POS = 5.24
    ANTIPARALLEL_D_INTER_NEG = 4.50
    
    # Calculate the screw paramters

    axis = np.array([0, 0, 1])
    theta_inter = 2 * np.pi * np.cos(pitch_angle) ** 2 / num_strand
    R = PARALLEL_D_INTER * np.cos(pitch_angle) / theta_inter
    
    f = lambda x : (R * x / np.tan(pitch_angle)) ** 2 + (2 * R * np.sin(x / 2)) ** 2\
                    - DI_PEPTIDE_LENGTH ** 2
    
    theta = np.sign(pitch_angle) * np.absolute(scipy.optimize.broyden1(f, np.pi))

    tilt_angle = np.arctan2(2 * R * np.sin(theta / 2), R * theta / np.tan(pitch_angle)) 

    # Get dipeptide directions one strand

    M = geometry.rotation_matrix_from_axis_and_angle(axis, theta)

    di_pp_directions = [ (np.array([np.sin(tilt_angle), 0, np.cos(tilt_angle)]),
                        np.array([np.cos(tilt_angle), 0, -np.sin(tilt_angle)]))]

    for i in range(1, strand_length // 2): 
        di_pp_directions.append((np.dot(M, di_pp_directions[-1][0]),
                                 np.dot(M, di_pp_directions[-1][1])))

    # Generate a reference strand

    strand = build_beta_strand_from_dipeptide_directions(di_pp_directions)

    strand = basic.transform_residue_list(strand, np.identity(3),
                np.array([0, -R * np.cos(theta / 2), 0]) 
                - (strand[0]['ca'] + strand[2]['ca']) / 2)

    # Generate the rest of the strands by rotation

    sheet = [strand[:strand_length]]

    if sheet_type == 'parallel':
        pitch_inter = -np.sin(pitch_angle) * PARALLEL_D_INTER * 2 * np.pi / theta_inter 
        M_inter, t_inter = geometry.get_screw_transformation(axis, 
                            theta_inter, pitch_inter, np.zeros(3))

        for i in range(1, num_strand):
            sheet.append(basic.transform_residue_list(sheet[-1], M_inter, t_inter))
   
    #TODO: Implement antiparallel beta barrel generation

    return sheet
