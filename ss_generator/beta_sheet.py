import numpy as np

from . import numeric
from . import geometry
from . import basic


D_MEAN = 3.80

def get_internal_coordinates_for_ideal_strand(R, delta, alpha, eta):
    '''Get 4 internal coordinates for an ideal beta strand.
    The inputs are the screw redius R, the angle alpha which is
    between a tangent of the screw spiral and the horizontal plane
    and the screw angle omega between Ca_i and Ca_i+2, delta which
    is the screw angle from Ca_i to Ca_i+2 and eta which is the value
    of tilting.

    Note that alpha is in the range (0, pi/2) for right handed strands
    and (pi/2, pi) for left handed strands.

    The outputs are theta1, tau1, theta2, tau2.
    '''

    # Adjust the sign of delta for left handed strand

    if alpha > np.pi / 2:
        delta = -delta

    theta1 = 2 * np.arcsin(R * np.absolute(np.tan(delta / 2)) / (D_MEAN * np.absolute(np.cos(alpha))))
    h = 2 * D_MEAN * np.sin(theta1 / 2) * np.sin(alpha)

    p2 = np.array([0, D_MEAN * np.sin(theta1 / 2) * np.cos(alpha), h / 2])

    # Get p1

    p1_0 = np.array([D_MEAN * np.cos(theta1 / 2), 0, 0])
    M1 = geometry.rotation_matrix_from_axis_and_angle(p2, eta)
    p1 = np.dot(M1, p1_0)

    # Get p3 and p4

    M_screw, t_screw = geometry.get_screw(np.array([0, 0, 1]), delta,
            2 * np.pi / np.absolute(delta) * h, np.array([-R, 0, 0]))

    p3 = np.dot(M_screw, p1) + t_screw
    p4 = np.dot(M_screw, p2) + t_screw

    theta2 = geometry.angle(p1 - p2, p3 - p2)
    tau1 = geometry.dihedral(-p2, p1, p2, p3)
    tau2 = geometry.dihedral(p1, p2, p3, p4)

    return theta1, tau1, theta2, tau2

def get_ideal_parameters_from_internal_coordinates(theta1, tau1, theta2, tau2):
    '''Get ideal parameters R, delta, alpha and eta from internal coordinates.'''

    # Always requires theta2 >= theta1

    if theta2 < theta1:
        theta1, tau1, theta2, tau2 = theta2, tau2, theta1, tau1

    # Get all the points

    p0 = D_MEAN * np.array([np.sin(theta1), np.cos(theta1), 0])
    p1 = np.array([0, 0, 0])
    p2 = np.array([0, D_MEAN, 0])
    p3 = geometry.cartesian_coord_from_internal_coord(p0, p1, p2, D_MEAN, theta2, tau1)
    p4 = geometry.cartesian_coord_from_internal_coord(p1, p2, p3, D_MEAN, theta1, tau2)

    # Get the rotation and translation of the middle point between p0 and p2

    M = np.transpose(geometry.create_frame_from_three_points(p2, p3, p4))
    t = (p4 - p0) / 2
    
    # Get parameters
    
    v, delta = geometry.rotation_matrix_to_axis_and_angle(M)
    if delta < 0:
        delta = -delta
        v = -v

    R = np.linalg.norm(t - np.dot(t, v) * v) / (2 * np.sin(delta / 2))
    alpha = np.pi / 2 - geometry.angle(v, p2 - p0)
    if alpha < 0:
        alpha += np.pi

    p1_0 = np.cross(p2 - p0, v)
    if np.dot(p1_0, p1 - (p2 + p0) / 2) < 0:
        p1_0 = -p1_0

    eta = geometry.angle(p1 - (p2 + p0) / 2, p1_0)
    if np.dot(np.cross(p1_0, p1 - (p2 + p0) / 2), p2 - p0) < 0:
        eta = -eta

    return R, delta, alpha, eta

def get_screw_transformation_for_strand(strand, distance, direction):
    '''Get the screw transformation for a strand. The first five atoms
    of the strand are used to calculate relative parameters.
    '''
    # Get internal coordinates

    theta1 = geometry.angle(strand[0] - strand[1], strand[2] - strand[1])
    theta2 = geometry.angle(strand[1] - strand[2], strand[3] - strand[2])
    tau1 = geometry.dihedral(strand[0], strand[1], strand[2], strand[3])
    tau2 = geometry.dihedral(strand[1], strand[2], strand[3], strand[4])

    # Get the direction vector

    d_v = geometry.normalize(np.cross(strand[1] - strand[2], strand[3] - strand[2]))
    if direction == '-':
        d_v = -d_v

    # Get the ideal parameters

    R, delta, alpha, eta = get_ideal_parameters_from_internal_coordinates(theta1, tau1, theta2, tau2)

    # If theta1 == theta2 or the alpha is zero, do a pure translation
    
    if np.absolute(theta1 - theta2) < 0.001 \
      or np.absolute(alpha) < 0.001 or np.absolute(alpha - np.pi) < 0.001:
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        t = distance * d_v
        return M, t
    
    # Get the anchor

    anchor = strand[:3] if theta1 <= theta2 else strand[1:4]

    # Get the outer cylinder radius
    
    R_out = R + D_MEAN * np.cos(min(theta1, theta2) / 2)
    
    # Get the screw axis
   
    rot_p2_p0 = geometry.rotation_matrix_from_axis_and_angle(anchor[2] - anchor[0], -eta)
    x = geometry.normalize(np.dot(rot_p2_p0, anchor[1] - (anchor[2] + anchor[0]) / 2))
    rot_x = geometry.rotation_matrix_from_axis_and_angle(x, np.pi / 2 - alpha)
    
    z = geometry.normalize(np.dot(rot_x, anchor[2] - anchor[0]))
    if np.dot(d_v, z) < 0:
        z = -z
    
    u = (anchor[0] + anchor[2]) / 2 - R * x
    
    # Get the pitch
    
    pitch = geometry.pitch_angle_to_pitch(alpha, R_out)
    
    # Get the screw angle
    
    screw_angle = -distance * np.sin(alpha) / R_out
    
    # Get the screw rotation and translation
    
    return geometry.get_screw(z, screw_angle, pitch, u)
    
def generate_ideal_beta_sheet_from_internal_coordinates(theta1, tau1, theta2, tau2, length, num_strands, sheet_type='parallel'):
    '''Generate an ideal beta sheet from three internal coordinates, the length of each strand
    and the number of strands.
    '''
    # Generate one strand

    ds = [D_MEAN] * (length - 1)
    thetas = ([theta1, theta2] * length)[:length - 2]
    taus = ([tau1, tau2] * length)[:length - 3]

    strand = basic.generate_segment_from_internal_coordinates(ds, thetas, taus)

    # For parallel sheets, use a simple way to generate other strands
    
    if sheet_type == 'parallel':

        PARA_BP_LEN_MEAN = 4.84
        PARA_BP_LEN_STD = 0.24

        # Get the screw rotation and translation
        # Because the equation used for screw angle calculation will
        # slightly shrink the length of bp vectors and also because
        # the outer part of the sheet should have larger bp vector length,
        # use PARA_BP_LEN_MEAN + PARA_BP_LEN_STD here.
    
        M, t = get_screw_transformation_for_strand(strand, PARA_BP_LEN_MEAN + PARA_BP_LEN_STD, '+') 

        # Generate strands
    
        sheet = [strand]
        
        for i in range(1, num_strands):
            sheet.append([np.dot(M, ca) + t for ca in sheet[i - 1]])
   
        return sheet

    else:
        raise Exception("Current implementation only support parallel sheets.")

def get_strand_parameters(strand_type):
    '''Return length, angle and torsion parameters of a strand'''
    if strand_type == 'antiparallel':
        return {'length_mean':3.798, 'length_std':0.03, 'angle_mean':np.radians(123.9),
                'angle_std':np.radians(12.6), 'torsion_mean':np.radians(195.8), 'torsion_std':np.radians(48.2)}
    
    if strand_type == 'parallel':
        return {'length_mean':3.799, 'length_std':0.03, 'angle_mean':np.radians(121.1),
                'angle_std':np.radians(10.5), 'torsion_mean':np.radians(195.5), 'torsion_std':np.radians(30.9)}

def get_bp_vector_parameters(strand_type, direction):
    '''Return d_mean, d_std, x_mean, x_std, y_mean, y_std, x_y_cov
    of a bp vector.
    '''
    if strand_type not in ['antiparallel', 'parallel'] or direction not in ['+', '-']:
        raise Exception("Invalid bp vector type: {0}, {1}".format(strand_type, direction))

    if strand_type == 'antiparallel':
        if direction == '+':
            return {'d_mean':4.50, 'd_std':0.28, 'x_mean':-0.836, 
                    'x_std':1.324, 'y_mean':0.226, 'y_std':0.261, 'x_y_cov':0.383}
        
        elif direction == '-':
            return {'d_mean':5.24, 'd_std':0.26, 'x_mean':1.177, 
                    'x_std':1.884, 'y_mean':1.280,  'y_std':0.320, 'x_y_cov':0.400}
    
    if strand_type == 'parallel':
        if direction == '+':
            return {'d_mean':4.84, 'd_std':0.24, 'x_mean':-0.277, 
                    'x_std':0.819, 'y_mean':-0.189, 'y_std':0.248, 'x_y_cov':0.286}
        
        elif direction == '-':
            return {'d_mean':4.84, 'd_std':0.24, 'x_mean':0.751, 
                    'x_std':0.918, 'y_mean':0.437, 'y_std':0.246, 'x_y_cov':0.244}

def get_bp_vector_score(bp_vector, strand_type, direction):
    '''Get the score of a bp vector, which is the probability
    density of that bp vector.'''
    # Set parameters for different strand types

    params = get_bp_vector_parameters(strand_type, direction)

    # Check the direction

    if (direction == '+' and bp_vector[2] < 0) \
            or (direction == '-' and bp_vector[2] > 0):
        return 0

    # Check the vector length

    d = np.linalg.norm(bp_vector)
    d_score = numeric.gaussian(d, params['d_mean'], params['d_std'])

    # Return the probability density

    return d_score * numeric.multivariate_gaussian(bp_vector[:2], np.array([params['x_mean'], params['y_mean']]),
            np.array([[params['x_std'], params['x_y_cov']], [params['x_y_cov'], params['y_std']]]))

def check_bp_vector(bp_vector, strand_type, direction, cutoff=0.1):
    '''Return True if a bp_vector is allowed.'''

    return get_bp_vector_score(bp_vector, strand_type, direction) > cutoff

def make_strand_seed(strand, strand_type, direction):
    '''Make a three atom seed needed to grow a strand.'''
    strand_seed = []
    strand_params = get_strand_parameters(strand_type)

    # Build the first two atoms by translating the first two atoms of the reference

    ref_frame = geometry.create_frame_from_three_points(strand[0], strand[1], strand[2])
    params = get_bp_vector_parameters(strand_type, direction)

    z_abs = np.sqrt(params['d_mean'] ** 2 - params['x_mean'] ** 2 - params['y_mean'] ** 2)
    z = z_abs if direction == '+' else -z_abs

    t = params['x_mean'] * ref_frame[0] + params['y_mean'] * ref_frame[1] + z * ref_frame[2]

    for i in range(2):
        strand_seed.append(strand[i] + t)

    # Get the third seed atom by search the best position relactive to the reference frame

    direction2 = '-' if direction == '+' else '+'

    ref_frame2 = geometry.create_frame_from_three_points(strand[1], strand[2], strand[3])

    v = geometry.normalize(strand_seed[1] - strand_seed[0])

    best_score = 0
    best_p = None

    for i in range(2000):
        theta = np.random.normal(strand_params['angle_mean'], strand_params['angle_std'])
        u = np.array([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])
        uu = geometry.normalize(u - np.dot(u, v) * v)

        p = strand_params['length_mean'] * (np.sin(theta) * uu - np.cos(theta) * v) + strand_seed[1]
        p_local = np.dot(ref_frame2, p - strand[2])

        score = get_bp_vector_score(p_local, strand_type, direction2)

        if score > best_score:
            best_p = p
            best_score = score

    strand_seed.append(best_p)

    return strand_seed

def build_a_random_strand_from_a_reference(strand, strand_type, direction, seed=None, adjust_first_atom=False): 
    '''Build a random beta strand based on a reference strand.'''
   
    if seed is None:
        seed = make_strand_seed(strand, strand_type, direction)

    new_strand = seed 

    # Add a atom to the reference strand such that we can build a frame for the last atom
    
    strand_params = get_strand_parameters(strand_type)
    extended_strand = strand + [geometry.cartesian_coord_from_internal_coord(strand[-3],
                        strand[-2], strand[-1], strand_params['length_mean'], 
                        strand_params['angle_mean'], strand_params['torsion_mean'])]

    # Extend the strand randomly
    
    for i in range(3, len(strand)):
        ref_frame = geometry.create_frame_from_three_points(
                extended_strand[i - 1], extended_strand[i], extended_strand[i + 1])

        # Try to find a proper value of theta and tau
       
        theta_best = 0
        tau_best = 0
        score_best = 0
        p_best = None

        for j in range(300):
            theta = np.random.normal(strand_params['angle_mean'],  strand_params['angle_std'])
            tau = np.random.normal(strand_params['torsion_mean'],  strand_params['torsion_std'])

            p = geometry.cartesian_coord_from_internal_coord(new_strand[i - 3],
                    new_strand[i - 2], new_strand[i - 1], strand_params['length_mean'], theta, tau)

            p_local = np.dot(ref_frame, p - strand[i])

            score = get_bp_vector_score(p_local, strand_type, direction)

            if score > score_best:
                theta_best = theta
                tau_best = tau
                score_best = score
                p_best = p

        if score_best < 0.01:
            return None
                
        new_strand.append(p_best)
        direction = '-' if direction == '+' else '+' #Flip the direction

    # Adjust the position of the first atom

    if adjust_first_atom:
        new_strand[0] = new_strand[1] + new_strand[2] - new_strand[3]

    return new_strand

def build_a_random_sheet_from_a_reference(strand, strand_type, direction, num_strands):
    '''Build a random sheet from a reference strand.'''
    new_sheet = [strand[:]]

    while len(new_sheet) < num_strands:
        for i in range(1, num_strands):
            strand = build_a_random_strand_from_a_reference(new_sheet[i - 1], 
                    strand_type, direction, adjust_first_atom=True)

            if strand is None:
                new_sheet = new_sheet[:1]
                break

            new_sheet.append(strand)

    return new_sheet

def extend_a_strand(strand, forward=True):
    '''Extend a strand by one atom. Angle and torsion of the 
    strand will be used to extend the new atom.
    '''
    if forward:
        last_p = geometry.cartesian_coord_from_internal_coord(
                strand[-3], strand[-2], strand[-1], D_MEAN, 
                geometry.angle(strand[-4] - strand[-3], strand[-2] - strand[-3]),
                geometry.dihedral(strand[-5], strand[-4], strand[-3], strand[-2]))
        
        return strand + [last_p]

    else:
        first_p = geometry.cartesian_coord_from_internal_coord(
                strand[2], strand[1], strand[0], D_MEAN,
                geometry.angle(strand[3] - strand[2], strand[1] - strand[2]),
                geometry.dihedral(strand[4], strand[3], strand[2], strand[1]))
        
        return [first_p] + strand

def get_expected_bp_position_of_three_atoms(ref_atoms, strand_type, direction):
    '''Return the expected bp position defined by three atoms.'''
    frame = geometry.create_frame_from_three_points(ref_atoms[0], ref_atoms[1], ref_atoms[2])
    
    params = get_bp_vector_parameters(strand_type, direction)
    x = params['x_mean']
    y = params['y_mean']
    z_abs = np.sqrt(params['d_mean'] ** 2 - x ** 2 - y ** 2)
    z = z_abs if direction == '+' else -z_abs

    return ref_atoms[1] + np.dot(np.transpose(frame), np.array([x, y, z]))
  
def get_expected_bp_position_of_five_atoms(ref_atoms, strand_type, direction):
    '''Return the expected bp position defined by five atoms.
    Two thetas and two taus are used to define a screw transformation.
    The bp position is calculated by screw transforming the third 
    reference atom.
    '''
    params = get_bp_vector_parameters(strand_type, direction)
    M, t = get_screw_transformation_for_strand(ref_atoms, params['d_mean'], direction)
    
    return np.dot(M, ref_atoms[2]) + t 

def get_expected_bp_positions_of_strand(strand, strand_type, direction, method='five'):
    '''Get the expected bp positions of a strand.
    The method argument specifies how many reference atoms are
    used to build each expected bp position.
    '''
    expected_positions = []
    
    if method == 'three': 
        extended_ref = extend_a_strand(strand, True)
        extended_ref = extend_a_strand(extended_ref, False)

        for i in range(len(extended_ref) - 2):
            expected_positions.append(get_expected_bp_position_of_three_atoms(
                extended_ref[i:i + 3], strand_type, direction))
                
            direction = '-' if direction == '+' else '+'
    
    elif method == 'five':
        extended_ref = extend_a_strand(strand, True)
        extended_ref = extend_a_strand(extended_ref, True)
        extended_ref = extend_a_strand(extended_ref, False)
        extended_ref = extend_a_strand(extended_ref, False)

        for i in range(len(extended_ref) - 4):
            expected_positions.append(get_expected_bp_position_of_five_atoms(
                extended_ref[i:i + 5], strand_type, direction))
            
            direction = '-' if direction == '+' else '+'

    else:
        raise Exception("Invalid method: " + method)

    return expected_positions

def build_a_strand_from_a_reference(ref_strand, strand_type, direction):
    '''Build a strand from a reference strand.'''
    # When the number of expected positions is even, extend the strand by one atom

    extended_ref = ref_strand if len(ref_strand) % 2 != 0 else extend_a_strand(ref_strand, True)

    # Get expected positions
    
    expected_positions = get_expected_bp_positions_of_strand(extended_ref, strand_type, direction)

    # Adjust the expected positions such that the bond lengths are the ideal value
    # Shift the position along the reference direction 

    center_of_mass_old = sum(expected_positions) / len(expected_positions)

    i = 1
    while i < len(expected_positions) - 1:
        tt_ref = extended_ref[i] - (extended_ref[i - 1] + extended_ref[i + 1]) / 2

        v = expected_positions[i + 1] - expected_positions[i - 1]
        vv = geometry.normalize(v)
        tt_ref_oth = geometry.normalize(tt_ref - np.dot(tt_ref, vv))

        center = (expected_positions[i - 1] + expected_positions[i + 1]) / 2
        u = expected_positions[i] - center
        t = u - np.dot(u, vv) * vv
        t_rest = t - np.dot(t, tt_ref_oth) * tt_ref_oth
        
        l = np.sqrt(D_MEAN ** 2 - (np.linalg.norm(v) / 2) ** 2 - np.linalg.norm(t_rest) ** 2)

        expected_positions[i] = center + t_rest + l * tt_ref_oth

        i += 2

    center_of_mass_new = sum(expected_positions) / len(expected_positions)

    for i in range(len(expected_positions)):
        expected_positions[i] += center_of_mass_old - center_of_mass_new

    return expected_positions[:len(ref_strand)] 

def build_a_sheet_from_a_reference(ref_strand, strand_type, direction, num_strands):
    '''Build a sheet from a reference strand.'''
    sheet = [ref_strand[:]]

    for i in range(1, num_strands):
        sheet.append(build_a_strand_from_a_reference(sheet[i - 1], strand_type, direction))

    return sheet

def change_radius_local(strand, bend_position, bend_coef):
    '''Chane the local screw radius. 
    The new screw radius will be the old one times bend_coef. The value of
    delta should also be adjusted.
    '''
    return perturb_strand_local(strand, bend_position, 
            lambda x : (x[0] * bend_coef, 2 * np.arctan(np.tan(x[1] / 2) / bend_coef), x[2], x[3]))

def change_alpha_local(strand, twist_position, alpha_change):
    '''Shift the local alpha.
    The value of delta should also be adjusted.
    '''
    return perturb_strand_local(strand, twist_position,
            lambda x : (x[0], 2 * np.arctan(np.tan(x[1] / 2) * np.cos(x[2] + alpha_change) / np.cos(x[2])), 
                x[2] + alpha_change, x[3]))

def perturb_strand_local(strand, position, perturb_function):
    '''Perturb a strand at a given position according to a 
    perturb_function.
    '''
    # get internal coordinates

    ds, thetas, taus = basic.get_internal_coordinates_from_ca_list(strand)

    # get old geometry parameters

    R, delta, alpha, eta = perturb_function(get_ideal_parameters_from_internal_coordinates(
            thetas[position - 1], taus[position - 1], thetas[position], taus[position]))

    # update internal coordinates

    thetas[position - 1], taus[position - 1], thetas[position], taus[position] = \
            get_internal_coordinates_for_ideal_strand(R, delta, alpha, eta)

    thetas[position + 1] = thetas[position - 1]
    taus[position + 1] = taus[position - 1]

    # return a new strand

    new_strand = strand[:3]

    for i in range(3, len(strand)):
        new_strand.append(geometry.cartesian_coord_from_internal_coord(
            new_strand[i - 3], new_strand[i - 2], new_strand[i - 1],
            ds[i - 1], thetas[i - 2], taus[i - 3]))

    return new_strand

def bend_strand(strand, position, bend_angle):
    '''Bend a strand at a given position.'''
    
    def bend_two_thetas(x):
        theta1, tau1, theta2, tau2 = x
        if theta1 > theta2:
            return theta1 + bend_angle / 2, tau1, theta2 - bend_angle / 2, tau2
        else:
            return theta1 - bend_angle / 2, tau1, theta2 + bend_angle / 2, tau2

    return change_strand_internal_coord_local(strand, position, bend_two_thetas)

def twist_strand(strand, position, twist_angle):
    '''Twist a strand at a given position.'''

    return change_strand_internal_coord_local(strand, position, 
            lambda x : (x[0], x[1] + twist_angle, x[2], x[3] + twist_angle))

def change_strand_internal_coord_local(strand, position, perturb_function):
    '''Perturb the internal coordinates of a strand at 
    position and position + 1 according to a perturb_function.
    '''
    # get internal coordinates

    ds, thetas, taus = basic.get_internal_coordinates_from_ca_list(strand)

    # perturb internal coordinates

    thetas[position], taus[position], thetas[position + 1], taus[position + 1] = \
            perturb_function((thetas[position], taus[position], thetas[position + 1], taus[position + 1]))

    # return a new strand

    new_strand = strand[:3]

    for i in range(3, len(strand)):
        new_strand.append(geometry.cartesian_coord_from_internal_coord(
            new_strand[i - 3], new_strand[i - 2], new_strand[i - 1],
            ds[i - 1], thetas[i - 2], taus[i - 3]))

    return new_strand

def random_perturb_strand(strand, strand_type, perturb_coef):
    '''Randomly perturb a strand.'''
    # get internal coordinates

    ds, thetas, taus = basic.get_internal_coordinates_from_ca_list(strand)

    # Get parameters for the strand

    params = get_strand_parameters(strand_type)

    # Perturb the internal coordinates

    for i in range(len(thetas)):
        thetas[i] = thetas[i] + np.random.normal(0, perturb_coef * params['angle_std'])

    for i in range(len(taus)):
        taus[i] = taus[i] + np.random.normal(0, perturb_coef * params['torsion_std'])

    # return a new strand

    new_strand = strand[:3]

    for i in range(3, len(strand)):
        new_strand.append(geometry.cartesian_coord_from_internal_coord(
            new_strand[i - 3], new_strand[i - 2], new_strand[i - 1],
            ds[i - 1], thetas[i - 2], taus[i - 3]))

    com_old = sum(strand) / len(strand)
    com_new = sum(new_strand) / len(new_strand)

    new_strand = [p + com_old - com_new for p in new_strand]

    return new_strand

def thread_backbone_for_strand(ca_list, strand_type):
    '''Thread backbone atoms for a ca list of a beta strand.'''
    # Make the N termial residue

    residue_list = [{'ca': ca_list[0],
            'n': geometry.cartesian_coord_from_internal_coord(ca_list[2], 
                ca_list[1], ca_list[0], 1.45, 
                np.radians(123.1) if strand_type == 'parallel' else np.radians(125.2), 
                np.radians(-136.0) if strand_type == 'parallel' else np.radians(-116.2)),
            'c': geometry.cartesian_coord_from_internal_coord(ca_list[2],
                ca_list[1], ca_list[0], 1.52, 
                np.radians(20.8) if strand_type == 'parallel' else np.radians(20.8), 
                np.radians(-93.0) if strand_type == 'parallel' else np.radians(-94.4))}]

    # Make middle residues

    for i in range(1, len(ca_list) - 1):
        residue_list.append({'ca': ca_list[i],
            'n': geometry.cartesian_coord_from_internal_coord(ca_list[i - 1],
                ca_list[i + 1], ca_list[i], 1.45, 
                np.radians(120.8) if strand_type == 'parallel' else np.radians(121.9), 
                np.radians(16.4) if strand_type == 'parallel' else np.radians(15.7)),
            'c': geometry.cartesian_coord_from_internal_coord(ca_list[i + 1],
                ca_list[i - 1], ca_list[i], 1.52, 
                np.radians(114.0) if strand_type == 'parallel' else np.radians(115.2), 
                np.radians(-20.0) if strand_type == 'parallel' else np.radians(-17.2))})

    # Make the N terminal residue

    residue_list.append({'ca': ca_list[-1],
        'n': geometry.cartesian_coord_from_internal_coord(ca_list[-3],
            ca_list[-2], ca_list[-1], 1.45, 
            np.radians(15.4) if strand_type == 'parallel' else np.radians(15.2), 
            np.radians(99.6) if strand_type == 'parallel' else np.radians(100.5)),
        'c': geometry.cartesian_coord_from_internal_coord(ca_list[-3],
            ca_list[-2], ca_list[-1], 1.52, 
            np.radians(113.5) if strand_type == 'parallel' else np.radians(112.8), 
            np.radians(11.6) if strand_type == 'parallel' else np.radians(-8.5))})

    # Buil O atoms

    for i in range(1, len(ca_list)):
        residue_list[i - 1]['o'] = basic.get_o_for_peptide_bond(
                residue_list[i - 1]['c'], residue_list[i]['n'], residue_list[i]['ca'])

    return residue_list

def thread_backbone_for_sheet(ca_list, sheet_type):
    '''Thread backbone atoms for a ca list of a beta sheet.'''
    sheet_list = []

    for strand in ca_list:
        sheet_list.append(thread_backbone_for_strand(strand, sheet_type))

    return sheet_list
