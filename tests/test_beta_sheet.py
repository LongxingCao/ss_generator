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


    res_list = ssg.beta_sheet.build_beta_barrel('parallel', 12, 20, np.radians(20))
    ssg.IO.save_residue_list(res_list, "beta_barrel.pdb")

def test_beta_sheet_skeleton():
    print("test beta sheet skeleton.")

    topology = [(3, 10), (3, 12), (4, 12), (5, 9)]
    creases = [(np.array([5, 0]), np.array([6.5, 3]), 0),
                (np.array([3.5, 0]), np.array([3.5, 1]), 0)]

    skeleton = ssg.BetaSheetSkeleton(topology, creases)
    
    #print(skeleton.get_skeleton_boundary())
    #print(skeleton.split_boundary_by_crease(creases[0]))
    print(skeleton.point_on_lower_left(np.array([4.5, 1]), creases[0]))
