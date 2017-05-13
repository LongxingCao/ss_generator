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

    return sheet

def attach_beta_strand_to_reference(strand, ref_strand, strand_type, bp_map, direction):
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
       
    # Find the Eucleadian transformation

    M, t = geometry.get_superimpose_transformation(current_positions, expected_positions)

    # Transform the strand

    return basic.transform_residue_list(strand, M, t)
