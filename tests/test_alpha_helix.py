#!/usr/bin/env python3

import pytest
import numpy as np
np.seterr(all='raise')

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

def test_build_nexus():
    print("test build nexus.")

    theta = np.radians(91.8)
    tau = np.radians(49.5)

    axis = ssg.geometry.rotation_matrix_to_axis_and_angle(
            ssg.alpha_helix.theta_tau_to_rotation_matrix(theta, tau))[0]

    c_theta, c_tau = ssg.alpha_helix.theta_tau_for_nexus(axis, axis)

    print("theta = {0:.2f}\ttau = {1:.2f}\taxis = {2}\tc_theta = {3:.2f}\tc_tau = {4:.2f}".format(
            np.degrees(theta), np.degrees(tau), axis, np.degrees(c_theta), np.degrees(c_tau)))

def test_generate_alpha_helix():
    print("test generating alpha helices.")

    #ds = 100 * [3.81]
    #thetas = 99 * [np.radians(91.8)]
    #taus = 98 * [np.radians(49.5)]

    #ca_list = ssg.basic.generate_segment_from_internal_coordinates(ds, thetas, taus)
    #ssg.IO.save_ca_list(ca_list, "straight_helix.pdb")

    #ca_list = ssg.basic.generate_segment_from_internal_coordinates(
    #        ds, thetas + np.radians(3.35) * np.random.uniform(-1, 1, 99), taus + np.radians(7.1) * np.random.uniform(-1, 1, 98))
    #ssg.IO.save_ca_list(ca_list, "random_helix.pdb")

    #screw_axes = [np.array([0, 0, 1])] * 20
    #ca_list = ssg.alpha_helix.generate_alpha_helix_from_screw_axes(screw_axes)
    #ssg.IO.save_ca_list(ca_list, "z_helix.pdb")

    #screw_axes = [np.array([0, 0, 1])]
    #for i in range(100):
    #    screw_axes.append(ssg.geometry.normalize(screw_axes[i] + 0.001 * np.array([np.random.normal(), np.random.normal(), np.random.normal()])))

    #ca_list = ssg.alpha_helix.generate_alpha_helix_from_screw_axes(screw_axes)
    #ssg.IO.save_ca_list(ca_list, "random_screws.pdb")
    
    ca_list = ssg.alpha_helix.generate_super_coil(np.array([0, 0, 1]), np.radians(-3.6), np.radians(12), 1000)
    ssg.IO.save_ca_list(ca_list, "super_coil.pdb")

def test_perturb_alpha_helix():
    print("test perturb alpha helices.")
    
    ds = 100 * [3.81]
    thetas = 99 * [np.radians(91.8)]
    taus = 98 * [np.radians(49.5)]

    ca_list_before = ssg.basic.generate_segment_from_internal_coordinates(ds, thetas, taus)
    for ca in ca_list_before:
        ca += np.array([10, 0, 0])
    
    ssg.IO.save_ca_list(ca_list_before, "helix_before_perturb.pdb")
   
    #random_perturbed_ca_list = ssg.alpha_helix.randomize_a_helix(ca_list_before, 0.1)
    #ssg.IO.save_ca_list(random_perturbed_ca_list, "helix_random_perturb.pdb")

    #phase_shifted_ca_list = ssg.alpha_helix.shift_helix_phase(random_perturbed_ca_list, np.pi)
    #ssg.IO.save_ca_list(phase_shifted_ca_list, "helix_phase_shifted.pdb")

    #ca_to_twist = random_perturbed_ca_list
    #twisted_ca_list = ssg.alpha_helix.twist_helix(ca_to_twist, ca_to_twist[-1] - ca_to_twist[0], np.radians(12), np.radians(-3.6), 0.5)
    #ssg.IO.save_ca_list(twisted_ca_list, "helix_twisted.pdb")

def test_thread_bb():
    print("test thread bb.")
   
    n = 10
    ds = n * [3.81]
    thetas = (n - 1) * [np.radians(91.8)]
    taus = (n - 2) * [np.radians(49.5)]

    ca_list = ssg.basic.generate_segment_from_internal_coordinates(ds, thetas, taus)
    ssg.IO.save_ca_list(ca_list, 'straight_helix.pdb')
    res_list = ssg.alpha_helix.thread_backbone_for_helix(ca_list)
    ssg.IO.save_residue_list(res_list, 'straight_helix_bb.pdb')
