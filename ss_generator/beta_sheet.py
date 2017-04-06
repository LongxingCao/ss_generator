import numpy as np

from . import geometry
from . import basic


D_MEAN = 3.80

def get_internal_coordinates_for_ideal_sheet(R, alpha, delta):
    '''Get 4 internal coordinates for an ideal beta sheet.
    The inputs are the screw redius R, the angle alpha which is
    between a tangent of the screw spiral and the horizontal plane
    and the screw angle omega between Ca_i and Ca_i+2.

    Note that alpha is in the range (0, pi/2) for right handed strands
    and (pi/2, pi) for left handed strands.

    The outputs are theta1, tau1, theta2, tau2.
    '''

    theta1 = 2 * np.arcsin(R * np.sin(delta / 2) / (D_MEAN * np.absolute(np.cos(alpha))))
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

    alpha = np.arctan2(np.dot(z, p2), np.dot(y, p2))
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
    
    return strand ###DEBUG

