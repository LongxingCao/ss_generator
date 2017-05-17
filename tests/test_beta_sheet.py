#!/usr/bin/env python3

import pytest
import numpy as np
np.seterr(all='raise')

import ss_generator as ssg


def test_build_beta_sheet():
    print("test build beta sheet.")
    
    res_list = ssg.beta_sheet.build_ideal_flat_beta_sheet('parallel', 10, 2)
    #res_list = ssg.beta_sheet.build_ideal_flat_beta_sheet('antiparallel', 41, 2)
    ssg.IO.save_residue_list(res_list, "ideal_flat_sheet.pdb")


    phi1 = -120 
    psi1 = 125
    phi2 = -120
    psi2 = 125

    for i, strand in enumerate(res_list):
        #ssg.basic.change_torsions(strand, 6, np.radians(-120), np.radians(134.6))
        #ssg.basic.change_torsions(strand, 5, np.radians(-153), np.radians(153))
        
        for j in range(0, len(strand), 2):
            ssg.basic.change_torsions(strand, j, np.radians(phi1), np.radians(psi1))
            ssg.basic.change_torsions(strand, j + 1, np.radians(phi2), np.radians(psi2))


    ssg.IO.save_residue_list(res_list, "perturbed_sheet.pdb")


    peptide_dipeptide_transformations = ssg.beta_sheet.load_peptide_dipeptide_transformations("peptide_dipeptide_transformations.json")
    
    #di_pp_direction = ssg.beta_sheet.get_dipeptide_bond_direction(*res_list[0][:3])
    #expected_transformation = ssg.beta_sheet.get_expected_dipeptide_transformation(*res_list[0][:2], *di_pp_direction)
    #print(ssg.beta_sheet.search_peptide_dipeptide_transformations(peptide_dipeptide_transformations, expected_transformation))



    axis = np.array([0, 0, 1])
    theta = np.radians(0)
    M = ssg.geometry.rotation_matrix_from_axis_and_angle(axis, theta)
    pitch_angle = np.radians(45)
    seed = (np.array([np.sin(pitch_angle), 0, np.cos(pitch_angle)]),
            np.array([np.cos(pitch_angle), 0, -np.sin(pitch_angle)]))
    di_pp_directions = [seed]

    for i in range(10):
        v1 = np.dot(M, di_pp_directions[-1][0])
        v2 = np.dot(M, di_pp_directions[-1][1])
        di_pp_directions.append((v1, v2))


    res_list = ssg.beta_sheet.build_beta_strand_from_dipeptide_directions(di_pp_directions, peptide_dipeptide_transformations)
    ssg.IO.save_residue_list(res_list, "dipp_sheet.pdb")

