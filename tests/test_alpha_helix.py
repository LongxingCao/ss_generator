#!/usr/bin/env python3

import pytest
import numpy as np

import ss_generator as ssg


def test_transformations():
    print("test transformations.")
    mean_theta = np.radians(91.8)
    std_theta = np.radians(3.35)
    mean_tau = np.radians(49.5)
    std_tau = np.radians(7.1)

    coef = [-1, 0, 1]

    for c1 in coef:
        for c2 in coef:
            theta = mean_theta + c1 * std_theta
            tau = mean_tau + c2 * std_tau

            axis, xi = ssg.geometry.rotation_matrix_to_axis_and_angle(
                    ssg.alpha_helix.theta_tau_to_rotation_matrix(theta, tau))

            c_theta, c_tau = ssg.alpha_helix.axis_to_theta_tau(axis)

            print("theta = {0:.2f}\ttau = {1:.2f}\txi = {2:.2f}\taxis = {3}\tc_theta = {4:.2f}\tc_tau = {5:.2f}".format(
                np.degrees(theta), np.degrees(tau), np.degrees(xi), axis, np.degrees(c_theta), np.degrees(c_tau)))

def test_generate_alpha_helix():
    print("test generating alpha helices.")

    ds = 20 * [3.81]
    thetas = 19 * [np.radians(91.8)]
    taus = 18 * [np.radians(49.5)]

    ca_list = ssg.alpha_helix.generate_alpha_helix_from_internal_coordinates(ds, thetas, taus)
    ssg.IO.save_ca_list(ca_list, "straight_helix.pdb")

    screw_axes = [np.array([0, 0, 1])] * 20
    ca_list = ssg.alpha_helix.generate_alpha_helix_from_screw_axes(screw_axes)
    ssg.IO.save_ca_list(ca_list, "z_helix.pdb")

    ca_list = ssg.alpha_helix.generate_super_coil(np.array([0, 0, 1]), np.radians(-4), np.radians(10), 100)
    ssg.IO.save_ca_list(ca_list, "super_coil.pdb")

