import numpy as np
import scipy.optimize

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

def attach_beta_strand_to_reference_byHB(strand, ref_strand, strand_type, bp_map, direction):
    '''Attach a beta strand to a reference strand.
    The beta pairing between the two strand are defined by the bp_map
    which maps reference strand residues to residues of the strand to be attached.
    The direction variable is either 0 or 1. When direction == 0, the 
    strand is attached to the side pointed by the NH and CO vectors of
    residues in the ref_strand that have even indices. When direction == 1,
    the strand is attached to the other side.
    Return a new strand after transformation.
    '''
    if direction not in [0, 1]:
        raise Exception("direction must be either 0 or 1!")

    current_positions = []  # Current positions of hydrogen bonding atoms in the strand 
    expected_positions = [] # Expected positions of hydrogen bonding atoms in the strand

    ref_ids = [k for k in bp_map.keys() if k % 2 == direction]

    for i_ref in ref_ids:
        
        # Get the residue id that use NH to pair with the reference

        i_nh = bp_map[i_ref] if strand_type == 'antiparallel' else bp_map[i_ref] + 1
        
        # Get the residue id that use CO to pair with the reference

        i_co = bp_map[i_ref] if strand_type == 'antiparallel' else bp_map[i_ref] - 1

        # Get the expected positions of HB atoms

        if i_nh in range(len(strand)):
            nh_expected = basic.get_hb_nh_coord_from_co(ref_strand[i_ref]['c'],
                                ref_strand[i_ref]['o'])
            
            current_positions.append(strand[i_nh]['n'])
            current_positions.append(strand[i_nh]['h'])
            expected_positions.append(nh_expected[0])
            expected_positions.append(nh_expected[1])
        
        if i_co in range(len(strand)):
            co_expected = basic.get_hb_co_coord_from_nh(ref_strand[i_ref]['n'],
                                ref_strand[i_ref]['h'])
            
            current_positions.append(strand[i_co]['c'])
            current_positions.append(strand[i_co]['o'])
            expected_positions.append(co_expected[0])
            expected_positions.append(co_expected[1])
       
    # Find the Euclidean transformation

    M, t = geometry.get_superimpose_transformation(current_positions, expected_positions)

    # Transform the strand

    return basic.transform_residue_list(strand, M, t)

def get_expeceted_bp_positions_for_two_residue(strand, res_id, strand_type):
    '''
    Get the expected bp residue positions of two residues in a strand using
    a screw transformation defined by four torsions of residue at res_id and
    res_id + 1.
    Return two expected CA positions at the positive direction defined by the 
    HB atoms of the residue at res_id and two expected CA positions at the negative
    direction.
    '''
    if not (0 < res_id < len(strand) - 2):
        raise Exception("Residues must be in the interior of the strand.")

    # Get the screw transformation of the strand

    M, t = geometry.point_lists_to_screw_transformation(
            [strand[res_id - 1]['c'], strand[res_id]['n'], strand[res_id]['ca']],
            [strand[res_id + 1]['c'], strand[res_id + 2]['n'], strand[res_id + 2]['ca']])

    axis, angle, pitch, u = geometry.get_screw_parameters(M, t)

    # Get the radius of the screw for the Ca atom in res_id

    v = u - strand[res_id]['ca'] 
    R = np.linalg.norm(v - np.dot(v, axis) * axis)

    # Get the pitch for the inter strand screw

    pitch_angle_strand = geometry.pitch_to_pitch_angle(pitch, R)
    pitch_angle_inter = np.pi / 2 - pitch_angle_strand
    
    pitch_inter = geometry.pitch_angle_to_pitch(pitch_angle_inter, R)

    def get_transformed_residues(d1, d2, direction):
        inter_axis = np.sign(np.dot(direction, axis)) * axis
        angle_inter1 = -np.sign(angle) * d1 * np.sin(pitch_angle_inter) / R
        angle_inter2 = -np.sign(angle) * d2 * np.sin(pitch_angle_inter) / R

        M_inter1, t_inter1 = geometry.get_screw_transformation(
                            inter_axis, angle_inter1, pitch_inter, u)

        M_inter2, t_inter2 = geometry.get_screw_transformation(
                            inter_axis, angle_inter2, pitch_inter, u)

        return [basic.transform_residue(strand[res_id], M_inter1, t_inter1)['ca'],
                basic.transform_residue(strand[res_id + 1], M_inter2, t_inter2)['ca']]

    # Get the ca-ca distances

    d_pos = 4.84 if strand_type == 'parallel' else 5.24
    d_neg = 4.84 if strand_type == 'parallel' else 4.5
    
    hb_direction = strand[res_id]['o'] - strand[res_id]['c']

    return (get_transformed_residues(d_pos, d_neg, hb_direction), 
            get_transformed_residues(d_neg, d_pos, -hb_direction))
     
def attach_beta_strand_to_reference_by_screw(strand, ref_strand, strand_type, bp_map, direction):
    '''Attach a beta strand to a reference strand.
    The beta pairing between the two strand are defined by the bp_map
    which maps reference strand residues to residues of the strand to be attached.
    The direction variable is either 0 or 1. When direction == 0, the 
    strand is attached to the side pointed by the NH and CO vectors of
    residues in the ref_strand that have even indices. When direction == 1,
    the strand is attached to the other side.
    Return a new strand after transformation.
    '''
    if direction not in [0, 1]:
        raise Exception("direction must be either 0 or 1!")

    # Get expected residues for the ref_strand

    expected_strand = []

    for i in range(1, len(ref_strand) - 2, 2):
        expected_bp_positions = get_expeceted_bp_positions_for_two_residue(
                                    ref_strand, i, strand_type)

        expected_strand += expected_bp_positions[1 - direction]

    if len(expected_strand) == 0:
        raise Exception("Cannot attach the strand, the ref_strand is too short.")

    # Get the Euclidean transformation

    current_positions = []  
    expected_positions = [] 

    res_ids = [key for key in bp_map.keys() if 0 < key < len(ref_strand) - 2]

    for res_id in res_ids:
        current_positions.append(strand[bp_map[res_id]]['ca'])
        expected_positions.append(expected_strand[res_id - 1])

        
    M, t = geometry.get_superimpose_transformation(current_positions, expected_positions)

    # Transform the strand

    return basic.transform_residue_list(strand, M, t)

def calc_n_ca_c_angle_between_peptide_plan(eta, epsilon_n, epsilon_c):
    '''Calculate the n_ca_c angle given the angle eta between two peptide plane,
    epsilon_n the residual of the angle between the ca_n vector and the intersection line and
    epsilon_c the residual of the angle between the ca_c vector and the intersection line.
    '''
    v_n = np.array([np.cos(epsilon_n), 0, np.sin(epsilon_n)])
    v_c = np.array([np.cos(epsilon_c) * np.cos(eta), np.cos(epsilon_c) * np.sin(eta),
                    np.sin(epsilon_c)])

    return np.arccos(np.dot(v_n, v_c))

def get_flat_dipeptide_torsions(epsilon_n, eta):
    '''Get the torsions for a flat dipeptide.'''
    epsilon_c = epsilon_n + np.radians(9)

    n = 1.47 * np.array([-np.cos(epsilon_n) * np.sin(eta / 2), -np.sin(epsilon_n), 
                         -np.cos(eta / 2) * np.cos(epsilon_n)])
    ca = np.array([0, 0, 0])
    c = 1.53 * np.array([np.cos(epsilon_c) * np.sin(eta / 2), -np.sin(epsilon_c), 
                         -np.cos(eta / 2) * np.cos(epsilon_c)])

    return geometry.dihedral(np.array([0, 1, 0]), n, ca, c), \
            geometry.dihedral(n, ca, c, np.array([0, 1, 0]))

def calc_flat_dipeptide_for_n_ca_vector(n_ca_vector, di_pp_direction, hb_direction):
    '''Given the n_ca vector, the direction of a dipeptide bond, the
    direction of the hydrogen bonds, calculate the flat dipeptide after it.
    Return the ca_c vector, c_n vector and phi, psi torsions of the 
    flat dipeptide.
    '''
    # Create a frame for the dipeptide bond

    x = geometry.normalize(di_pp_direction)
    y = -geometry.normalize(hb_direction)
    z = np.cross(x, y)

    # Make a frame for the ca_c vector

    n_ca_vector = geometry.normalize(n_ca_vector)
    n_ca_x = geometry.perpendicular_vector(n_ca_vector)
    n_ca_y = np.cross(n_ca_vector, n_ca_x)

    N_CA_C_ANLGE = np.radians(111.2)

    def virtual_n_ca_c_anlge(theta):
        ca_c_vector = np.cos(np.pi - N_CA_C_ANLGE) * n_ca_vector \
                    + np.sin(np.pi - N_CA_C_ANLGE) * np.cos(theta) * n_ca_x \
                    + np.sin(np.pi - N_CA_C_ANLGE) * np.sin(theta) * n_ca_y 

        eta = np.pi - 2 * np.absolute(np.arctan2(
                np.dot(ca_c_vector, z), np.dot(ca_c_vector, x)))
        
        epsilon_c = geometry.angle(ca_c_vector, 
                np.dot(ca_c_vector, x) * x + np.dot(ca_c_vector, z) * z)
       
        return calc_n_ca_c_angle_between_peptide_plan(eta, epsilon_c - np.radians(9), epsilon_c)

    #for theta in range(-180, 180): ###DEBUG
    #    print(theta, np.degrees(virtual_n_ca_c_anlge(np.radians(theta))))

    # Solve the direction of the ca_c vector

    theta = scipy.optimize.broyden1(lambda x : virtual_n_ca_c_anlge(x) - N_CA_C_ANLGE, 0, f_tol=0.001)

    # Calculate the flat peptide torsions 

    ca_c_vector = np.cos(np.pi - N_CA_C_ANLGE) * n_ca_vector \
                + np.sin(np.pi - N_CA_C_ANLGE) * np.cos(theta) * n_ca_x \
                + np.sin(np.pi - N_CA_C_ANLGE) * np.sin(theta) * n_ca_y \

    ca_c_vector *= 1.53

    eta = np.pi - 2 * np.absolute(np.arctan2(
            np.dot(ca_c_vector, z), np.dot(ca_c_vector, x)))
    
    epsilon_c = geometry.angle(ca_c_vector, 
            np.dot(ca_c_vector, x) * x + np.dot(ca_c_vector, z) * z)
    
    phi, psi = get_flat_dipeptide_torsions(epsilon_c - np.radians(9), eta)

    # Calculate the c_n vector

    n = geometry.cartesian_coord_from_internal_coord(y, np.zeros(3), ca_c_vector,
            1.32, np.radians(114), np.pi)

    c_n_vector = n - ca_c_vector
    
    #print(ca_c_vector, n) ###DEBUG
    #print(np.degrees(phi), np.degrees(psi)) ###DEBUG

    return ca_c_vector, c_n_vector, phi, psi

def build_beta_strand_from_dipeptide_directions(di_pp_directions):
    '''Build a beta strand given a list of dipeptide directions.
    Elements in di_pp_directions have format (di_pp_direction, hb_direction).
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

        ca = strand[res_id]['ca']
        n = strand[res_id]['n']
       
        ca_c_vector, c_n_vector, phi2, psi2 = \
            calc_flat_dipeptide_for_n_ca_vector(ca - n, *di_pp_directions[i])

        phi = geometry.dihedral(strand[res_id - 1]['c'], n, ca, ca + ca_c_vector)
        psi = geometry.dihedral(n, ca, ca + ca_c_vector, ca + ca_c_vector + c_n_vector)

        basic.change_torsions(strand, res_id, phi, psi)
        basic.change_torsions(strand, res_id + 1, phi2, psi2)

    return strand
