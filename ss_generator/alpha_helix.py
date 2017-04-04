import numpy as np

from . import geometry

D_MEAN = 3.81
D_STD = 0.02
THETA_MEAN = np.radians(91.8)
THETA_STD = np.radians(3.35)
TAU_MEAN = np.radians(49.5)
TAU_STD = np.radians(7.1)

def theta_tau_to_rotation_matrix(theta, tau):
    '''Get the rotation matrix corresponding to the
    bond angle theta and dihedral tau.
    '''
    return np.dot(geometry.rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), tau),
                  geometry.rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), theta - np.pi))

def axis_to_theta_tau(axis):
    '''Get the bond angle theta and dihedral tau,
    from a rotation axis.
    '''
    theta = 2 * np.arctan(-axis[1] / axis[0])
    tau = 2 * np.arctan(axis[0] / axis[2])

    return theta, tau

def check_theta_tau(theta, tau):
    '''Check that if the theta and tau are within the
    3 * STD respectively.
    '''
    if theta > THETA_MEAN + 3 * THETA_STD or theta < THETA_MEAN - 3 * THETA_STD:
        return False

    if tau > TAU_MEAN + 3 * TAU_STD or tau < TAU_MEAN - 3 * TAU_STD:
        return False

    return True

def theta_tau_for_nexus(axis, axis_new):
    '''Given an axis, find a pair of (theta, tau) such that
    after rotating the coordinate frame by M(theta, tau), the
    coordinates of the axis in the new frame is axis_new.
    '''
    vx1 = axis[0]
    vy1 = axis[1]
    vz1 = axis[2]
    vx2 = axis_new[0]
    vy2 = axis_new[1]
    vz2 = axis_new[2]

    # Calculate the tau angle

    t = 1 / (vz2 + vz1) * (vx1 + np.sign(vx1) * np.sqrt(vx1 ** 2 - (vz2 ** 2 - vz1 ** 2)))

    tau = 2 * np.arctan(t)

    # Calculate the theta angle

    s = np.sin(tau)
    c = np.cos(tau)

    q = 1 / (vx2 + s * vz1 - c * vx1) * (-vy1 \
            - np.sign(vy1) * np.sqrt(vy1 ** 2 - (vx2 ** 2 - (s * vz1 - c * vx1) ** 2)))

    theta = 2 * np.arctan(q)

    return theta, tau

def get_internal_coordinates_from_ca_list(ca_list):
    '''Get the list of ds, thetas and taus from a ca list.'''
    ds = []
    thetas = []
    taus = []

    for i in range(len(ca_list) - 1):
        ds.append(np.linalg.norm(ca_list[i + 1] - ca_list[i]))

    for i in range(1, len(ca_list) - 1):
        thetas.append(geometry.angle(ca_list[i - 1] - ca_list[i],
            ca_list[i + 1] - ca_list[i]))

    for i in range(1, len(ca_list) - 2):
        taus.append(geometry.dihedral(ca_list[i - 1], ca_list[i],
            ca_list[i + 1], ca_list[i + 2]))

    return ds, thetas, taus


### Functions to generate a new helix

def generate_alpha_helix_from_internal_coordinates(ds, thetas, taus):
   '''Generate an alpha helix from a set of internal coordinates.
   Return a list of Ca coordinates.
   '''
   # Make sure that the sizes of internal coordinates are correct

   if len(ds) < 3 or len(thetas) < 2 or len(taus) < 1 \
      or len(ds) != len(thetas) + 1 or len(ds) != len(taus) + 2:
          raise Exception("Incompatible sizes of internal coordinates.")

   # Make the first three Ca atoms

   ca_list = []
   ca_list.append(ds[0] * np.array([np.sin(thetas[0]),np.cos(thetas[0]), 0]))
   ca_list.append(np.array([0, 0, 0]))
   ca_list.append(np.array([0, ds[1], 0]))

   # Make the rest of Ca atoms

   for i in range(len(taus)):
      ca_list.append(geometry.cartesian_coord_from_internal_coord(
          ca_list[i], ca_list[i + 1], ca_list[i + 2], ds[i + 2], thetas[i + 1], taus[i]))

   return ca_list

def generate_alpha_helix_from_screw_axes(screw_axes, relieve_strain=False, M_init=None):
    '''Generate an alpha helix from a list of screw axes.
    Return a list of Ca coordinates.
    '''
    thetas, taus, M_rot = get_theta_tau_and_rotation_matrix_from_screw_axes(
            screw_axes, relieve_strain=relieve_strain, M_init=M_init) 

    ca_list = generate_alpha_helix_from_internal_coordinates(
            [D_MEAN] * (len(screw_axes) + 2), thetas, taus)

    return [np.dot(M_rot, ca) for ca in ca_list]


def get_theta_tau_and_rotation_matrix_from_screw_axes(screw_axes, relieve_strain=False, M_init=None):
    '''Get internal coordinates theta and tau from a list
    of screw axes.
    '''
    # Get the rotation matrix from the default frame to the first local frame.
    # Note that there are infinite number of possible matrices to do so.

    if M_init is None:

        axis_default = geometry.rotation_matrix_to_axis_and_angle(
                theta_tau_to_rotation_matrix(THETA_MEAN, TAU_MEAN))[0]

        M_init = geometry.rotation_matrix_to_superimpose_two_vectors(
                axis_default, screw_axes[0], theta=np.random.uniform(-np.pi, np.pi))

    # Get the internal coordinates

    thetas = [THETA_MEAN] * 2
    taus = [TAU_MEAN]

    M_rot = np.dot(M_init, theta_tau_to_rotation_matrix(THETA_MEAN, TAU_MEAN))

    for i in range(1, len(screw_axes)):
        local_axis = np.dot(np.transpose(M_rot), screw_axes[i])
       
        theta, tau = axis_to_theta_tau(local_axis)
        
        # Relieve the strain

        if relieve_strain and i % 7 == 0 and i + 1 < len(screw_axes):
            next_axis = np.dot(np.transpose(M_rot), screw_axes[i + 1])
            ideal_axis = geometry.rotation_matrix_to_axis_and_angle(
                    theta_tau_to_rotation_matrix(THETA_MEAN, TAU_MEAN))[0]

            theta, tau = theta_tau_for_nexus(next_axis, ideal_axis)

        if not check_theta_tau(theta, tau):
            raise Exception("The value of theta or tau beyond the limits.")

        M_local = theta_tau_to_rotation_matrix(theta, tau)
        M_rot = np.dot(M_rot, M_local)

        thetas.append(theta)
        taus.append(tau)

    return thetas, taus, M_init

def generate_super_coil(axis, omega, pitch_angle, length):
    '''Generate a alpha helix super coil.
    Return a list of Ca coordinates.
    '''

    axis = geometry.normalize(axis)
    M_rot = geometry.rotation_matrix_from_axis_and_angle(axis, omega)
    
    # Get the screw axes
    
    axis_perpendicular = None

    if np.abs(axis[0]) > 0.01:
        axis_perpendicular = geometry.normalize(
                np.array([axis[1], -axis[0], 0]))
    else:
        axis_perpendicular = geometry.normalize(
                np.array([0, axis[2], -axis[1]]))

    screw_seed = np.dot(geometry.rotation_matrix_from_axis_and_angle(
        axis_perpendicular, pitch_angle), axis)

    screw_axes = [screw_seed]

    for i in range(1, length):
        screw_axes.append(np.dot(M_rot, screw_axes[i - 1]))

    # Generate the helix

    return generate_alpha_helix_from_screw_axes(screw_axes, relieve_strain=True)

### Functions to perturb an existing helix

def randomize_a_helix(ca_list, ratio):
    '''Randomize internal coordinates of a helix. Only int(ratio * len(ca_list))
    residues are perturbed.
    '''
    ds, thetas, taus = get_internal_coordinates_from_ca_list(ca_list)

    num_to_perturb = int(ratio * len(ca_list))
    res_to_perturb = np.random.permutation(len(ca_list) - 3)[:num_to_perturb]

    for i in res_to_perturb:
        theta = np.random.normal(THETA_MEAN, THETA_STD)
        tau = np.random.normal(TAU_MEAN, TAU_STD)
        
        if check_theta_tau(theta, tau):
          thetas[i] = theta
          taus[i] = tau

    perturbed_ca_list = generate_alpha_helix_from_internal_coordinates(ds, thetas, taus)

    # Rotate and translate the perturbed helix, such that the first frame
    # coincide with its original frame

    M = np.transpose(geometry.create_frame_from_three_points(
        ca_list[0], ca_list[1], ca_list[2]))
    t = np.mean(ca_list, axis=0) - np.mean(perturbed_ca_list, axis=0)

    for i in range(len(perturbed_ca_list)):
        perturbed_ca_list[i] = np.dot(M, 
                perturbed_ca_list[i]) + t

    return perturbed_ca_list

def shift_helix_phase(ca_list, phase_shift):
    '''Shift the phase of a helix without changing 
    it's direction.
    '''
    # Get the screw axes
    
    screw_axes = []
    
    for i in range(1, len(ca_list) - 2):
        M1 = geometry.create_frame_from_three_points(
                ca_list[i - 1], ca_list[i], ca_list[i + 1])
        M2 = geometry.create_frame_from_three_points(
                ca_list[i], ca_list[i + 1], ca_list[i + 2])

        screw_axes.append(geometry.rotation_matrix_to_axis_and_angle(
            np.dot(np.transpose(M2), M1))[0])

    # Get the initial rotation matrix for helix generation

    M1 = geometry.create_frame_from_three_points(
            ca_list[0], ca_list[1], ca_list[2])
    M_init = np.dot(geometry.rotation_matrix_from_axis_and_angle(
        screw_axes[0], phase_shift), np.transpose(M1))

    # Calculate the Ca coordinates

    shifted_ca_list = generate_alpha_helix_from_screw_axes(screw_axes, relieve_strain=True, M_init=M_init)

    t = np.mean(ca_list, axis=0) - np.mean(shifted_ca_list, axis=0)
    for i in range(len(shifted_ca_list)):
        shifted_ca_list[i] = shifted_ca_list[i] + t

    return shifted_ca_list

def twist_helix(ca_list, axis, pitch_angle, omega, ratio):
    '''Twist a helix, making it closer to a super coil who is defined
    by axis, pitch_angle and omega. int(ratio * len(ca_list)) minimum
    twist units (each with 6 residues) are perturbed.
    '''
    ds, thetas, taus = get_internal_coordinates_from_ca_list(ca_list)

    M_init = geometry.create_frame_from_three_points(
            ca_list[0], ca_list[1], ca_list[2])

    # Get residues to perturb

    num_to_perturb = int(ratio * len(ca_list))
    res_to_perturb = sorted(np.random.permutation(len(taus) - 2)[:num_to_perturb])

    # Get the perturbed internal coordinates

    for i in range(len(taus)):
        if i in res_to_perturb:
            new_thetas, new_taus = twist_minimum_unit(thetas[i + 1: i + 4],
                    taus[i: i + 3], M_init, axis, pitch_angle, omega)
            
            if new_thetas is not None:
                for j in range(3):
                    thetas[i + 1 + j] = new_thetas[j]
                    taus[i + j] = new_taus[j]

        M_init = np.dot(M_init, theta_tau_to_rotation_matrix(thetas[i + 1], taus[i]))

    # Get new ca coordinates

    new_ca_list = generate_alpha_helix_from_internal_coordinates(ds, thetas, taus)
    
    M_rot = geometry.create_frame_from_three_points(
            ca_list[0], ca_list[1], ca_list[2])
    t = np.mean(ca_list, axis=0) - np.mean(new_ca_list, axis=0)

    for i in range(len(new_ca_list)):
        new_ca_list[i] = np.dot(M_rot, new_ca_list[i]) + t

    return new_ca_list

def twist_minimum_unit(thetas, taus, M_init, axis, pitch_angle, omega):
    '''Twist a minimum twist unit.
    Return new values for thetas and taus. Return None if the twist failed.
    '''

    if len(thetas) != 3 or len(taus) != 3:
        raise Exception("A minimum twist unit must have 3 angles and 3 torsions!")

    screw_axes = []

    # Get the new value of the first axis

    axis1_local = geometry.rotation_matrix_to_axis_and_angle(
            theta_tau_to_rotation_matrix(thetas[0], taus[0]))[0]

    axis1_global = np.dot(M_init, axis1_local)

    angle_to_rotate = pitch_angle - geometry.angle(axis, axis1_global)
    rotate_axis = geometry.normalize(np.cross(axis, axis1_global))

    for i in range(1, 5):
        M_rot = geometry.rotation_matrix_from_axis_and_angle(
                rotate_axis, angle_to_rotate)

        new_axis_global = np.dot(M_rot, axis1_global)
        new_axis_local = np.dot(np.transpose(M_init), new_axis_global)

        theta, tau = axis_to_theta_tau(new_axis_local)
        
        if check_theta_tau(theta, tau):
            screw_axes.append(new_axis_global)
            break

        angle_to_rotate /= 2

    if len(screw_axes) == 0: return None, None

    # Get the new screw axes

    M_rot = geometry.rotation_matrix_from_axis_and_angle(axis, omega)

    for i in range(1, 3):
        screw_axes.append(np.dot(M_rot, screw_axes[i - 1]))

    # Get the internal coordinates

    try:
        thetas, taus, M = get_theta_tau_and_rotation_matrix_from_screw_axes(
                screw_axes, M_init=M_init)
    except:
        return None, None

    return thetas[1:], taus


