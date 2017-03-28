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

            print("theta = {0:.2f}\ttau = {1:.2f}\txi = {2:.2f}\taxis = {3}".format(
                np.degrees(theta), np.degrees(tau), np.degrees(xi), axis))
