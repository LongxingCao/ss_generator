import numpy as np

from . import numeric
from . import geometry
from . import basic


D_MEAN = 3.80
PARA_BP_LEN_PLUS = 4.84

def get_internal_coordinates_for_ideal_sheet(R, alpha, delta):
    '''Get 4 internal coordinates for an ideal beta sheet.
    The inputs are the screw redius R, the angle alpha which is
    between a tangent of the screw spiral and the horizontal plane
    and the screw angle omega between Ca_i and Ca_i+2.

    Note that alpha is in the range (0, pi/2) for right handed strands
    and (pi/2, pi) for left handed strands.

    The outputs are theta1, tau1, theta2, tau2.
    '''

    # Adjust the sign of delta for left handed strand

    if alpha > np.pi / 2:
        delta = -delta

    theta1 = 2 * np.arcsin(R * np.absolute(np.sin(delta / 2)) / (D_MEAN * np.absolute(np.cos(alpha))))
    h = 2 * D_MEAN * np.sin(theta1 / 2) * np.sin(alpha)

    p1 = np.array([D_MEAN * np.cos(theta1 / 2), 0, 0])
    p2 = np.array([0, D_MEAN * np.sin(theta1 / 2) * np.cos(alpha), h / 2])
    q = np.array([-D_MEAN * np.sin(theta1 / 2) * np.cos(alpha) * np.sin(delta),
        D_MEAN * np.sin(theta1 / 2) * np.cos(alpha) * (1 + np.cos(delta)), h])
    p3 = q + D_MEAN * np.cos(theta1 / 2) * np.array([np.cos(delta), np.sin(delta), 0])

    theta2 = geometry.angle(p1 - p2, p3 - p2)
    tau1 = geometry.dihedral(-p2, p1, p2, p3)
    tau2 = geometry.dihedral(p1, p2, p3, 2 * q - p2)

    return theta1, tau1, theta2, tau2

def get_ideal_parameters_from_three_internal_coordinates(theta1, tau1, theta2):
    '''Get 3 ideal beta sheet parameters R, alpha and delta from
    three internal coordinates
    '''
    if theta1 > theta2:
        raise Exception("Current implementation requires theta1 < theta2.")

    p0 = D_MEAN * np.array([np.sin(theta1), np.cos(theta1), 0])
    p1 = np.array([0, 0, 0])
    p2 = np.array([0, D_MEAN, 0])
    p3 = D_MEAN * np.array([np.sin(theta2) * np.cos(tau1),
        1 - np.cos(theta2), -np.sin(theta2) * np.sin(tau1)])

    # Get the screw axis when the strand is in a plane

    s = np.array([0, 0, 1])

    # Get the screw axis in nondegenerative cases
    
    if np.absolute(p3[2]) > 0.001:

        lam = np.dot(2 * p2 - p3, p2 - p0) / np.dot(p3, np.array([0, 0, 1]))

        s = geometry.normalize(p2 - p0 + lam * np.array([0, 0, 1]))

        if np.dot(s, p3) < 0:
            s = -s

    # Get a new frame

    x = geometry.normalize(p1 - (p2 + p0) / 2)
    y = np.cross(s, x)
    z = s

    # Get the ideal parameters

    alpha = np.arctan2(np.dot(z, p2 - p0), np.dot(y, p2 - p0))
    v1 = p1 - p0
    v2 = p3 - p2
    delta = geometry.angle(v1 - np.dot(v1, z) * z, v2 - np.dot(v2, z) * z)
    R = np.absolute(D_MEAN * np.sin(theta1 / 2) * np.cos(alpha) / np.sin(delta / 2))

    return R, alpha, delta

def generate_ideal_beta_sheet_from_internal_coordinates(theta1, tau1, theta2, length, num_strands):
    '''Generate an ideal beta sheet from three internal coordinates, the length of each strand
    and the number of strands.
    '''
    # Calculate the internal coordinates
    
    R, alpha, delta = get_ideal_parameters_from_three_internal_coordinates(theta1, tau1, theta2)
    theta1, tau1, theta2, tau2 = get_internal_coordinates_for_ideal_sheet(R, alpha, delta)

    ds = [D_MEAN] * (length - 1)
    thetas = ([theta1, theta2] * length)[:length - 2]
    taus = ([tau1, tau2] * length)[:length - 3]

    # Generate one strand

    strand = basic.generate_segment_from_internal_coordinates(ds, thetas, taus)

    # Get the outer cylinder radius

    R_out = R + D_MEAN * np.cos(theta1 / 2)

    # Get the screw axis

    x = geometry.normalize(strand[1] - (strand[2] + strand[0]) / 2)
    rot_x = geometry.rotation_matrix_from_axis_and_angle(x, np.pi / 2 - alpha)
    z = geometry.normalize(np.dot(rot_x, strand[2] - strand[0]))
    u = strand[1] - R_out * x

    # Get the pitch

    pitch = geometry.pitch_angle_to_pitch(alpha, R_out)

    # Get the screw angle

    screw_angle = -PARA_BP_LEN_PLUS * np.sin(alpha) / R_out 

    # Get the screw rotation and translation

    M, t = geometry.get_screw(-z, screw_angle, pitch, u)

    # If the alpha is zero, do a pure translation

    if np.absolute(alpha) < 0.001:
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        t = -PARA_BP_LEN_PLUS * z

    # Generate strands

    sheet = [strand]
    
    for i in range(1, num_strands):
        sheet.append([np.dot(M, ca) + t for ca in sheet[i - 1]])

    return sheet

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

    if d < params['d_mean'] - params['d_std'] \
            or d > params['d_mean'] + params['d_std']:
        return 0

    # Return the probability density

    return numeric.multivariate_gaussian(bp_vector[:2], np.array([params['x_mean'], params['y_mean']]),
            np.array([[params['x_std'], params['x_y_cov']], [params['x_y_cov'], params['y_std']]]))

def check_bp_vector(bp_vector, strand_type, direction, cutoff=0.1):
    '''Return True if a bp_vector is allowed.'''

    return get_bp_vector_score(bp_vector, strand_type, direction) > cutoff

def make_strand_seed(ref_strand, strand_type, direction):
    '''Make a three atom seed needed to grow a strand.'''
    strand_seed = []
    
    # Build the first three atoms by translating the first three atoms of the reference

    ref_frame = geometry.create_frame_from_three_points(ref_strand[0], ref_strand[1], ref_strand[2])
    params = get_bp_vector_parameters(strand_type, direction)

    z_abs = np.sqrt(params['d_mean'] ** 2 - params['x_mean'] ** 2 - params['y_mean'] ** 2)
    z = z_abs if direction == '+' else -z_abs

    t = params['x_mean'] * ref_frame[0] + params['y_mean'] * ref_frame[1] + z * ref_frame[2]

    for i in range(3):
        strand_seed.append(ref_strand[i] + t)

    return strand_seed

def build_a_random_strand_from_a_reference(ref_strand, strand_type, direction, seed=None): 
    '''Build a random beta strand based on a reference strand.'''
   
    if seed is None:
        seed = make_strand_seed(ref_strand, strand_type, direction)

    new_strand = seed 

    # Add a atom to the reference strand such that we can build a frame for the last atom
    
    strand_params = get_strand_parameters(strand_type)
    extended_strand = ref_strand + [geometry.cartesian_coord_from_internal_coord(ref_strand[-3],
                        ref_strand[-2], ref_strand[-1], strand_params['length_mean'], 
                        strand_params['angle_mean'], strand_params['torsion_mean'])]

    # Extend the strand randomly
    
    for i in range(3, len(ref_strand)):
        ref_frame = geometry.create_frame_from_three_points(
                extended_strand[i - 1], extended_strand[i], extended_strand[i + 1])

        # Try to find a proper value of theta and tau
       
        theta_best = 0
        tau_best = 0
        score_best = 0
        p_best = None

        for j in range(100):
            theta = np.random.normal(strand_params['angle_mean'], 0.5 * strand_params['angle_std'])
            tau = np.random.normal(strand_params['torsion_mean'], 0.5 * strand_params['torsion_mean'])

            p = geometry.cartesian_coord_from_internal_coord(new_strand[i - 3],
                    new_strand[i - 2], new_strand[i - 1], strand_params['length_mean'], theta, tau)

            p_local = np.dot(ref_frame, p - ref_strand[i])

            score = get_bp_vector_score(p_local, strand_type, direction)

            if score > score_best:
                theta_best = theta
                tau_best = tau
                score_best = score
                p_best = p

        if score_best < 0.1:
            return None
                
        new_strand.append(p_best)
        direction = '-' if direction == '+' else '+' #Flip the direction

    return new_strand

    

