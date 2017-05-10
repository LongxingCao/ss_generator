#!/usr/bin/env python3

import pytest
import numpy as np
np.seterr(all='raise')

import ss_generator as ssg


def test_build_beta_sheet():
    print("test build beta sheet.")

    theta1 = np.radians(123.9 + 5)
    #theta1 = np.radians(123.9 - 10)
    #tau1 = np.radians(165)
    #tau1 = np.radians(180)
    tau1 = np.radians(190)
    #tau1 = np.radians(195.8)
    theta2 = np.radians(123.9 - 5)
    tau2 = np.radians(185)

    R, delta, alpha, eta = ssg.ca_tracing.beta_sheet.get_ideal_parameters_from_internal_coordinates(theta1, tau1, theta2, tau2)
    
    theta1_calc, tau1_calc, theta2_calc, tau2_calc = \
            ssg.ca_tracing.beta_sheet.get_internal_coordinates_for_ideal_strand(R, delta, alpha, eta)

    print("R = {0:.2f}, alpha = {1:.2f}, delta = {2:.2f}, eta={3:.2f}".format(R, np.degrees(alpha), np.degrees(delta), np.degrees(eta)))
    print("theta1 = {0:.2f}, theta1_calc = {1:.2f}".format(np.degrees(theta1), np.degrees(theta1_calc)))
    print("tau1 = {0:.2f}, tau1_calc = {1:.2f}".format(np.degrees(tau1), np.degrees(tau1_calc)))
    print("theta2 = {0:.2f}, theta2_calc = {1:.2f}".format(np.degrees(theta2), np.degrees(theta2_calc)))
    print("tau2 = {0:.2f}, tau2_calc = {1:.2f}".format(np.degrees(tau2), np.degrees(tau2_calc)))

    
    ca_list = ssg.ca_tracing.beta_sheet.generate_ideal_beta_sheet_from_internal_coordinates(theta1, tau1, theta2, tau2, 10, 5)
    ssg.IO.save_ca_list(ca_list, "ideal_sheet.pdb")
    res_list = ssg.ca_tracing.beta_sheet.thread_backbone_for_sheet(ca_list, 'parallel')
    
    #trimed_list = []
    #for i in range(len(res_list)):
    #    l = len(res_list[i])
    #    trimed_list.append(res_list[i][i:l - (len(res_list) - i)])

    #ssg.IO.save_residue_list(trimed_list, "ideal_sheet_bb.pdb")
    ssg.IO.save_residue_list(res_list, "ideal_sheet_bb.pdb")

def test_purterb_beta_sheet():
    print("test purterb beta sheet.")

    theta1 = np.radians(123.9 + 5)
    tau1 = np.radians(195)
    theta2 = np.radians(123.9 - 5)
    tau2 = np.radians(185)

    #ca_list_before_purterb = ssg.ca_tracing.beta_sheet.generate_ideal_beta_sheet_from_internal_coordinates(theta1, tau1, theta2, tau2, 11, 5)
    #ssg.IO.save_ca_list(ca_list_before_purterb, "sheet_before_purterb.pdb")
    #res_list_before_purterb = ssg.ca_tracing.beta_sheet.thread_backbone_for_sheet(ca_list_before_purterb, 'parallel')
    #ssg.IO.save_residue_list(res_list_before_purterb, "sheet_before_purterb_bb.pdb")

    #rand_strand = None
    #while rand_strand is None:
    #    rand_strand = ssg.ca_tracing.beta_sheet.build_a_random_strand_from_a_reference(ca_list_before_purterb[1], 'parallel', '-')
    #    #rand_strand = ssg.ca_tracing.beta_sheet.build_a_random_strand_from_a_reference(ca_list_before_purterb[1], 'parallel', '+', seed=ca_list_before_purterb[0][:3])
    #ssg.IO.save_ca_list(rand_strand, 'rand_strand.pdb')

    #rand_strand = None
    #while rand_strand is None:
    #    rand_strand = ssg.ca_tracing.beta_sheet.build_a_random_strand_from_a_reference(ca_list_before_purterb[1], 'parallel', '+')
    #    #rand_strand = ssg.ca_tracing.beta_sheet.build_a_random_strand_from_a_reference(ca_list_before_purterb[1], 'parallel', '+', seed=ca_list_before_purterb[2][:3])
    #ssg.IO.save_ca_list(rand_strand, 'rand_strand2.pdb')

    #rand_sheet = ssg.ca_tracing.beta_sheet.build_a_random_sheet_from_a_reference(ca_list_before_purterb[1], 'parallel', '+', 4)
    #ssg.IO.save_ca_list(rand_sheet, 'rand_sheet.pdb')
    
    #built_strand = ssg.ca_tracing.beta_sheet.build_a_strand_from_a_reference(ca_list_before_purterb[1], 'parallel', '+')
    #ssg.IO.save_ca_list(built_strand, 'built_strand.pdb')

    #built_sheet = ssg.ca_tracing.beta_sheet.build_a_sheet_from_a_reference(ca_list_before_purterb[1], 'parallel', '+', 8)
    #ssg.IO.save_ca_list(built_sheet, 'built_sheet.pdb')
    
    #random_perturbed_strand = ssg.ca_tracing.beta_sheet.random_perturb_strand(ca_list_before_purterb[0], 'parallel', 0.1)
    #ssg.IO.save_ca_list(random_perturbed_strand, 'random_perturb_strand.pdb')
    #random_perturbed_sheet = ssg.ca_tracing.beta_sheet.build_a_sheet_from_a_reference(random_perturbed_strand, 'parallel', '+', len(ca_list_before_purterb))
    #random_perturbed_sheet = ssg.ca_tracing.beta_sheet.superimpose_sheet(random_perturbed_sheet, ca_list_before_purterb)
    #ssg.IO.save_ca_list(random_perturbed_sheet, 'random_perturb_sheet.pdb')
    #random_perturbed_sheet_bb = ssg.ca_tracing.beta_sheet.thread_backbone_for_sheet(random_perturbed_sheet, 'parallel')
    #ssg.IO.save_residue_list(random_perturbed_sheet_bb, "random_perturb_sheet_bb.pdb")
    #
    #bended_strand = ssg.ca_tracing.beta_sheet.bend_strand(random_perturbed_sheet[0], np.radians(10))
    #ssg.IO.save_ca_list(bended_strand, 'bend_strand.pdb')
    #bended_sheet = ssg.ca_tracing.beta_sheet.build_a_sheet_from_a_reference(bended_strand, 'parallel', '+', len(random_perturbed_sheet))
    #bended_sheet = ssg.ca_tracing.beta_sheet.superimpose_sheet(bended_sheet, random_perturbed_sheet)
    #ssg.IO.save_ca_list(bended_sheet, 'bend_sheet.pdb')
    #bended_sheet_bb = ssg.ca_tracing.beta_sheet.thread_backbone_for_sheet(bended_sheet, 'parallel')
    #ssg.IO.save_residue_list(bended_sheet_bb, "bend_sheet_bb.pdb")

    #twisted_strand = ssg.ca_tracing.beta_sheet.twist_strand(random_perturbed_sheet[0], np.radians(5))
    #ssg.IO.save_ca_list(twisted_strand, 'twist_strand.pdb')
    #twisted_sheet = ssg.ca_tracing.beta_sheet.build_a_sheet_from_a_reference(twisted_strand, 'parallel', '+', len(random_perturbed_sheet))
    #twisted_sheet = ssg.ca_tracing.beta_sheet.superimpose_sheet(twisted_sheet, random_perturbed_sheet)
    #ssg.IO.save_ca_list(twisted_sheet, 'twist_sheet.pdb')
    #twisted_sheet_bb = ssg.ca_tracing.beta_sheet.thread_backbone_for_sheet(twisted_sheet, 'parallel')
    #ssg.IO.save_residue_list(twisted_sheet_bb, "twist_sheet_bb.pdb")

