#!/usr/bin/env python3

import pytest
import numpy as np
np.seterr(all='raise')

import ss_generator as ssg


def test_build_beta_sheet():
    print("test build beta sheet.")
    
    res_list = ssg.alpha_helix.build_ideal_straight_alpha_helix(20)
    ssg.IO.save_residue_list(res_list, "ideal_straight_helix.pdb")


    theta = np.radians(90)
    phi = np.radians(-4)
    directions = [np.array([0, np.sin(theta), np.cos(theta)])]

    for i in range(1, 50):
        directions.append(np.dot(ssg.geometry.rotation_matrix_from_axis_and_angle(
                                    np.array([0, 0, 1]), phi), directions[-1]))

    res_list = ssg.alpha_helix.build_alpha_helix_from_directions(directions)
    ssg.IO.save_residue_list(res_list, "helix_perturbed.pdb")
